"""Frozen training/validation pilot for measurement-conditioned internal updates."""
from pathlib import Path
import argparse
import collections
import gc
import json
import os
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import sha256
from src.real_response_data import ResponseCase, save_json, key
from src.visible_ptycho_measurements import VisiblePtychoMeasurements
from src.ptycho_internal_refinement import VisibleIntensityProblem
from src.learned_ptycho_refinement import SpatialPreconditioner, refine_columns


def verify_execution():
    path = ROOT/'protocols/m4_internal_update_execution_v1.json'
    freeze = json.loads(path.read_text())
    for group in ['code_hashes', 'input_hashes']:
        for name, digest in freeze[group].items():
            if sha256(ROOT/name) != digest:
                raise RuntimeError(f'Changed pilot dependency: {name}')
    contract = json.loads((ROOT/'protocols/m4_internal_update_pilot_v1.json').read_text())
    return freeze, contract, sha256(path)


def load_case(scan, fraction, device, role, batch):
    case = ResponseCase(ROOT, scan, fraction, device, role)
    view = VisiblePtychoMeasurements(ROOT, scan, fraction)
    if not np.array_equal(view.visible_ids, case.visible_ids):
        raise RuntimeError('Different visible geometry in the two input views')
    target = torch.tensor(view.frames().astype(np.float32)*case.alpha, device=device)
    problem = VisibleIntensityProblem(case.operator, case.visible_ids, target, batch=batch)
    return case, problem


@torch.no_grad()
def score_columns(case, columns, batch=256):
    error = denominator = 0.
    for start in range(0, len(case.label_ids), batch):
        ids = case.label_ids[start:start+batch]
        fields = case.operator.fields(columns, torch.as_tensor(ids, device=case.device))
        prediction = fields.abs().square().sum(1).double()
        target = case.targets(ids).double()
        error += float((prediction-target).square().sum())
        denominator += float(target.square().sum())
    return dict(relative_intensity_rmse=float(np.sqrt(error/denominator)),
                squared_error=error, target_squared_norm=denominator,
                predicted_frames=len(case.label_ids))


@torch.no_grad()
def validate(model, cases, cfg):
    model.eval()
    rows = []
    for case, problem in cases:
        columns, records = refine_columns(case.object[None], problem, case.fraction,
            steps=cfg['internal_steps'], model=model, differentiable=False)
        final_loss = float(problem.loss(columns))
        expected = records[-1]['visible_loss_after_polynomial']
        if abs(final_loss-expected) > 5e-7+2e-5*abs(expected):
            raise FloatingPointError('Final validation loss does not match its polynomial')
        rows.append(dict(scan=case.scan, fraction=case.fraction,
            **score_columns(case, columns, cfg['query_batch']),
            final_visible_half_mse=final_loss, internal_updates=records))
    model.train()
    return dict(mean_relative_intensity_rmse=float(np.mean([r['relative_intensity_rmse'] for r in rows])), rows=rows)


def train_update(model, optimizer, case, problem, ids, cfg):
    optimizer.zero_grad(set_to_none=True)
    target = case.targets(ids)
    columns, records = refine_columns(case.object[None], problem, case.fraction,
        steps=cfg['internal_steps'], model=model)
    prediction = case.operator.fields(columns, torch.as_tensor(ids, device=case.device)).abs().square().sum(1)
    loss = (prediction-target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite outer training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), gradient_norm=float(norm), internal_updates=records)


def train(args, out, contract, base):
    cfg = contract['method']; opt = contract['optimization']
    if args.kind not in ['pointwise', 'spatial'] or args.seed not in opt['seeds']:
        raise ValueError('Model or seed outside the pilot contract')
    training = {(s, f):load_case(s, f, args.device, 'train', cfg['visible_batch'])
                for s in ['W', 'IC1', 'IC2'] for f in contract['fractions']}
    validation = [load_case(s, f, args.device, 'validation', cfg['visible_batch'])
                  for s in ['TP1', 'TP2'] for f in contract['fractions']]
    rng = np.random.default_rng(args.seed)
    model = SpatialPreconditioner(args.kind).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    save_json(out/'config.json', dict(base, kind=args.kind, seed=args.seed, torch=torch.__version__,
        python=sys.version, host=os.uname().nodename, optimization=opt, method=cfg,
        parameters=sum(p.numel() for p in model.parameters())))
    logs = []; curves = []; best = float('inf'); best_step = None
    cumulative = 0.; branch_counts = collections.Counter()
    checkpoint_hashes = {}; sampled = []
    for step in range(opt['steps']+1):
        if step:
            scan = str(rng.choice(['W', 'IC1', 'IC2'], p=[.5, .25, .25]))
            fraction = float(rng.choice(contract['fractions']))
            case, problem = training[(scan, fraction)]
            ids = rng.choice(case.label_ids, size=cfg['query_batch'], replace=len(case.label_ids)<cfg['query_batch'])
            tick = time.perf_counter()
            result = train_update(model, optimizer, case, problem, ids, cfg)
            torch.cuda.synchronize(args.device)
            cumulative += time.perf_counter()-tick
            branch_counts.update(r['certificate']['branch'] for r in result['internal_updates'])
            sampled.append(dict(step=step, scan=scan, fraction=fraction,
                target_ids=ids.tolist(), **result))
            if step % 25 == 0:
                recent = sampled[-25:]
                logs.append(dict(step=step, mean_training_mse=float(np.mean([r['loss'] for r in recent])),
                    training_seconds=cumulative, branches=dict(branch_counts),
                    last_gradient_norm=result['gradient_norm']))
                save_json(out/'training_curve.json', logs)
                save_json(out/'status.json', dict(base, status='training', step=step,
                    updated=time.time(), best_step=best_step, best_validation=best,
                    training_seconds=cumulative, peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30))
        if step % opt['checkpoints_every'] == 0:
            metrics = validate(model, validation, cfg)
            curves.append(dict(step=step, **metrics))
            checkpoint = out/f'step{step}.pt'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                kind=args.kind, seed=args.seed, numpy_generator=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(args.device),
                validation=metrics, freeze_sha256=base['freeze_sha256']), checkpoint)
            checkpoint_hashes[checkpoint.name] = sha256(checkpoint)
            if metrics['mean_relative_intensity_rmse'] < best:
                best = metrics['mean_relative_intensity_rmse']; best_step = step
            save_json(out/'validation_curve.json', curves)
            save_json(out/'sampling_and_updates.json', sampled)
            print(json.dumps(dict(kind=args.kind, seed=args.seed, step=step,
                validation=metrics['mean_relative_intensity_rmse'], best_step=best_step,
                training_seconds=cumulative)), flush=True)
    selected = torch.load(out/f'step{best_step}.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(selected['model'])
    training_fit = validate(model, list(training.values()), cfg)
    save_json(out/'selected_training_endpoints.json', training_fit)
    record = dict(base, status='completed', kind=args.kind, seed=args.seed,
        selected_step=best_step, selected_validation_relative_rmse=best,
        selected_checkpoint=f'step{best_step}.pt', checkpoint_hashes=checkpoint_hashes,
        training_seconds=cumulative, branches=dict(branch_counts),
        peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
        optimization_steps=opt['steps'], validation_checkpoints=len(curves),
        role_statement='Selected using TP validation; W/IC outer targets trained. No FLY1/LFP/NCM/LCLS labels used.',
        operator_counts_scope='Each internal update: two forward and one adjoint application per visible frame. Autograd work and diagnostic scoring are additional and included in measured runtime.')
    save_json(out/'summary.json', record)
    save_json(out/'status.json', dict(record, updated=time.time()))


@torch.no_grad()
def controls(args, out, contract, base):
    rows = []; schedule = contract['controls']['iterations']
    cfg = contract['method']
    for role, scans in [('train', ['W', 'IC1', 'IC2']), ('validation', ['TP1', 'TP2'])]:
        for scan in scans:
            for fraction in contract['fractions']:
                case, problem = load_case(scan, fraction, args.device, role, cfg['visible_batch'])
                columns = case.object[None].clone()
                trajectory = []; cumulative = 0.
                destination = out/key(scan, fraction); destination.mkdir()
                for step in range(max(schedule)+1):
                    if step:
                        tick = time.perf_counter()
                        columns, trace = refine_columns(columns, problem, fraction, steps=1,
                            model=None, differentiable=False)
                        torch.cuda.synchronize(args.device)
                        cumulative += time.perf_counter()-tick
                        trajectory.append(dict(iteration=step, **trace[0]))
                    if step in schedule:
                        final_loss = float(problem.loss(columns))
                        if step:
                            expected = trajectory[-1]['visible_loss_after_polynomial']
                            if abs(final_loss-expected) > 5e-7+2e-5*abs(expected):
                                raise FloatingPointError('Uniform trajectory polynomial mismatch')
                        path = destination/f'step{step}_object.npy'
                        np.save(path, columns.cpu().numpy(), allow_pickle=False)
                        row = dict(role=role, scan=scan, fraction=fraction, step=step,
                            **score_columns(case, columns, cfg['query_batch']),
                            visible_half_mse=final_loss, iteration_seconds=cumulative,
                            object_file=str(path.relative_to(out)), object_sha256=sha256(path))
                        rows.append(row)
                save_json(destination/'trajectory.json', trajectory)
                save_json(out/'rows.json', rows)
                save_json(out/'status.json', dict(base, status='controls_running', scan=scan,
                    fraction=fraction, completed_conditions=len(rows)//len(schedule), updated=time.time()))
                print(json.dumps(dict(scan=scan, fraction=fraction, completed_conditions=len(rows)//len(schedule))), flush=True)
                del case, problem, columns
                gc.collect(); torch.cuda.empty_cache()
    summary = dict(base, status='completed', rows=len(rows), conditions=len(rows)//len(schedule),
        fixed_update_counts=schedule, query_scope='TP validation and original W/IC training_ids only',
        three_update_control='Matched physical forward/adjoint work; zero/1/2/10/30 shown as fixed additional context')
    save_json(out/'summary.json', summary)
    save_json(out/'status.json', dict(summary, updated=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'controls'], required=True)
    parser.add_argument('--kind', choices=['spatial', 'pointwise'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    freeze, contract, digest = verify_execution()
    name = 'uniform_controls' if args.mode == 'controls' else f'{args.kind}_{args.seed}'
    out = ROOT/'runs/m4_internal_update_pilot_v1'/name
    out.mkdir(parents=True, exist_ok=False)
    base = dict(pid=os.getpid(), host=os.uname().nodename, device=args.device,
        started=time.time(), freeze_sha256=digest, material_or_stress_labels_loaded=False,
        reconstructed_reference_pixels_loaded=False)
    save_json(out/'status.json', dict(base, status='loading'))
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(torch.device(args.device))
        torch.manual_seed(args.seed or 2026110202)
        np.random.seed(args.seed or 2026110202)
        torch.cuda.reset_peak_memory_stats(args.device)
        if args.mode == 'train':
            train(args, out, contract, base)
        else:
            controls(args, out, contract, base)
    except Exception as error:
        save_json(out/'failure.json', dict(base, error=repr(error), failed=time.time()))
        save_json(out/'status.json', dict(base, status='failed', error=repr(error), updated=time.time()))
        raise


if __name__ == '__main__':
    main()
