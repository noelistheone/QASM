"""Prospectively extend every pilot fit with preserved optimizer/RNG state."""
from pathlib import Path
import argparse
import collections
import json
import os
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import sha256
from src.real_response_data import save_json
from src.query_aware_refinement import OrthogonalUpdateNet
from scripts.run_query_aware_pilot import verify_new, update, evaluate
from scripts.run_ptycho_internal_pilot import load_case


def verify_extension():
    original, parent_hash = verify_new()
    path = ROOT/'protocols/m4_query_aware_extension_execution_v1.json'
    frozen = json.loads(path.read_text())
    for group in ['code_hashes', 'input_hashes']:
        for name, digest in frozen[group].items():
            if sha256(ROOT/name) != digest:
                raise RuntimeError(f'Changed extension dependency: {name}')
    contract = json.loads((ROOT/'protocols/m4_query_aware_extension_v1.json').read_text())
    return original, contract, sha256(path), parent_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=['complex_bounded', 'complex_unbounded'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    original, contract, digest, parent_hash = verify_extension()
    assert args.seed in original['optimization']['seeds']
    cfg = original['method']; opt = original['optimization']; bounded = args.kind == 'complex_bounded'
    parent = ROOT/'runs/m4_query_aware_pilot_v1'/f'{args.kind}_{args.seed}'
    out = ROOT/'runs/m4_query_aware_extended_v1'/f'{args.kind}_{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    base = dict(pid=os.getpid(), host=os.uname().nodename, device=args.device, started=time.time(),
        freeze_sha256=digest, parent_freeze_sha256=parent_hash, kind=args.kind, seed=args.seed,
        material_stress_labels_loaded=False, reference_pixels_loaded=False)
    save_json(out/'status.json', dict(base, status='loading'))
    try:
        torch.set_num_threads(4); torch.cuda.set_device(torch.device(args.device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(args.device)
        training = {(s, f):load_case(s, f, args.device, 'train', cfg['visible_batch'])
                    for s in original['roles']['train'] for f in original['fractions']}
        validation = [load_case(s, f, args.device, 'validation', cfg['visible_batch'])
                      for s in original['roles']['validation'] for f in original['fractions']]
        parent_path = parent/'step1000.pt'
        checkpoint = torch.load(parent_path, map_location=args.device, weights_only=False)
        assert checkpoint['step'] == 1000 and checkpoint['freeze_sha256'] == parent_hash
        model = OrthogonalUpdateNet().to(args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        assert all(g['lr'] == opt['learning_rate'] for g in optimizer.param_groups)
        rng = np.random.default_rng(); rng.bit_generator.state = checkpoint['numpy_generator']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(), args.device)
        resumed = evaluate(model, validation, bounded, cfg)
        old_rows = checkpoint['validation']['rows']
        differences = [r['relative_intensity_rmse']-o['relative_intensity_rmse']
                       for r, o in zip(resumed['rows'], old_rows)]
        assert max(abs(v) for v in differences) < 1e-5
        save_json(out/'resume_check.json', dict(parent_checkpoint_sha256=sha256(parent_path),
            differences=differences, metrics=resumed, optimizer_and_rng_restored=True,
            hardware_bitwise_identity_claim=False))
        save_json(out/'config.json', dict(base, torch=torch.__version__, method=cfg,
            original_optimization=opt, continuation=contract, parent_checkpoint_sha256=sha256(parent_path)))
        old_curve = json.loads((parent/'validation_curve.json').read_text())
        selected = min(old_curve, key=lambda r:(r['mean_relative_intensity_rmse'], r['step']))
        best = selected['mean_relative_intensity_rmse']; best_step = selected['step']
        best_path = parent/f'step{best_step}.pt'
        curves = []; sampling = []; logs = []; hashes = {}
        branches = collections.Counter(); boundary_count = 0; cumulative = 0.
        for step in range(1001, contract['total_steps']+1):
            scan = str(rng.choice(['W', 'IC1', 'IC2'], p=[.5, .25, .25]))
            fraction = float(rng.choice(original['fractions']))
            case, problem = training[(scan, fraction)]
            ids = rng.choice(case.label_ids, size=cfg['query_batch'], replace=False)
            tick = time.perf_counter()
            row = update(model, optimizer, case, problem, ids, bounded, cfg)
            torch.cuda.synchronize(args.device); cumulative += time.perf_counter()-tick
            branches.update(r['certificate']['branch'] for r in row['trace'])
            boundary_count += sum(r['query_bound'] is not None and r['query_bound']['cap'] < 1
                                  and r['step'] >= 1-1e-6 for r in row['trace'])
            sampling.append(dict(step=step, scan=scan, fraction=fraction, ids=ids.tolist(), **row))
            if step % 25 == 0:
                logs.append(dict(step=step, mean_loss=float(np.mean([r['loss'] for r in sampling[-25:]])),
                    training_seconds=cumulative, branches=dict(branches), boundary_updates=boundary_count))
                save_json(out/'training_curve.json', logs)
                save_json(out/'status.json', dict(base, status='training', step=step, updated=time.time(),
                    best_step=best_step, best_validation=best, training_seconds=cumulative,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30))
            if step % contract['validation_interval'] == 0:
                metrics = evaluate(model, validation, bounded, cfg)
                curves.append(dict(step=step, **metrics))
                cp = out/f'step{step}.pt'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    kind=args.kind, seed=args.seed, validation=metrics, numpy_generator=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(args.device),
                    freeze_sha256=digest), cp)
                hashes[cp.name] = sha256(cp)
                if metrics['mean_relative_intensity_rmse'] < best:
                    best = metrics['mean_relative_intensity_rmse']; best_step = step; best_path = cp; selected = curves[-1]
                save_json(out/'validation_curve.json', curves)
                save_json(out/'sampling_and_updates.json', sampling)
                print(json.dumps(dict(kind=args.kind, seed=args.seed, step=step,
                    validation=metrics['mean_relative_intensity_rmse'], best_step=best_step,
                    training_seconds=cumulative, boundary_updates=boundary_count)), flush=True)
        selected_checkpoint = torch.load(best_path, map_location=args.device, weights_only=False)
        model.load_state_dict(selected_checkpoint['model'])
        save_json(out/'selected_training_endpoints.json', evaluate(model, list(training.values()), bounded, cfg))
        save_json(out/'selected_validation.json', selected)
        result = dict(base, status='completed', total_steps=contract['total_steps'],
            additional_steps=contract['total_steps']-1000, selected_step=best_step,
            selected_validation_relative_rmse=best, selected_checkpoint=str(best_path.relative_to(ROOT)),
            selected_checkpoint_sha256=sha256(best_path), checkpoint_hashes=hashes,
            additional_validation_checkpoints=len(curves), total_validation_checkpoints=len(curves)+len(old_curve),
            additional_training_seconds=cumulative, branches=dict(branches), boundary_updates=boundary_count,
            peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30)
        save_json(out/'summary.json', result); save_json(out/'status.json', result)
    except Exception as error:
        save_json(out/'failure.json', dict(base, error=repr(error), time=time.time()))
        save_json(out/'status.json', dict(base, status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
