"""Frozen-mean NPPC controls, with explicit sample versus fixed normalization."""
from pathlib import Path
import argparse
import json
import os
import shutil
import socket
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import sha256
from src.mask_conditioned_learning import REMAINING, masks_from_orders, evaluation_masks
from src.mask_learning_context import MaskLearningContext
from src.posterior_components import MaskPosteriorComponents, component_loss


def atomic_json(path, data):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalization', choices=['sample', 'fixed'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mean-run', type=Path,
                        default=ROOT / 'runs/m2_mask_learning_20261101/image_seed20261117')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    context = MaskLearningContext(ROOT, args.mean_run, args.device, profile=args.profile)
    net = MaskPosteriorComponents(args.rank).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(.9, .999))
    rng = np.random.default_rng(args.seed)
    train_ids = np.asarray(context.split['covariance_train'])
    sources = [Path(__file__), ROOT / 'src/mask_learning_context.py', ROOT / 'src/posterior_components.py',
               ROOT / 'src/mask_conditioned_learning.py', ROOT / 'src/response_learning.py',
               ROOT / 'src/born_scattering.py']
    config = dict(normalization=args.normalization, seed=args.seed, steps=args.steps, batch=args.batch,
                  rank=args.rank, device=args.device, host=socket.gethostname(), profile=args.profile,
                  source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                  protocol_sha256=sha256(ROOT / 'protocols/m2_nppc_control_v1.json'),
                  data_manifest_sha256=sha256(ROOT / 'data/m2_mask_features_v1/manifest.json'),
                  mean_checkpoint_sha256=context.mean_checkpoint_hash, mean_step=context.mean_step,
                  mean_is_completed=context.mean_is_completed,
                  parameters=sum(p.numel() for p in net.parameters()))
    snapshot = args.out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name, digest in config['source_hashes'].items():
        destination = snapshot / name.replace('/', '__')
        if not destination.exists():
            shutil.copy2(ROOT / name, destination)
        assert sha256(destination) == digest
    if (args.out / 'summary.json').exists():
        raise RuntimeError('Already completed')
    best, start, elapsed, history = float('inf'), 0, 0., []
    if args.resume:
        previous = json.loads((args.out / 'config.json').read_text())
        for key in ['normalization', 'seed', 'steps', 'batch', 'rank', 'source_hashes',
                    'protocol_sha256', 'data_manifest_sha256', 'mean_checkpoint_sha256']:
            assert previous[key] == config[key], key
        checkpoint = torch.load(args.out / 'last.pt', map_location=args.device, weights_only=True)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, elapsed, history = [checkpoint[k] for k in ['step', 'best', 'seconds', 'history']]
        rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        scale_report = json.loads((args.out / 'training_scale.json').read_text())
    else:
        if (args.out / 'config.json').exists():
            raise RuntimeError('Existing run requires --resume')
        atomic_json(args.out / 'config.json', config)
        scale_report = context.training_energy_scale(args.batch)
        atomic_json(args.out / 'training_scale.json', scale_report)
    assert scale_report['mean_checkpoint_sha256'] == context.mean_checkpoint_hash
    energy_scale = scale_report['fixed_energy_scale']
    setup_seconds = time.perf_counter() - wall_start
    atomic_json(args.out / 'setup_timing.json', dict(seconds=setup_seconds, resumed=args.resume))
    t0 = time.perf_counter()

    def peak_memory():
        return torch.cuda.max_memory_allocated(args.device) / 2**30 if str(args.device).startswith('cuda') else 0.

    def loss_for(batch, weight):
        directions, variances, factors = net(batch['feature'], batch['mean'], batch['mask'])
        loss, metrics = component_loss(directions, variances, batch['truth'] - batch['mean'],
                                       args.normalization, energy_scale, weight)
        return loss, metrics, directions, variances, factors

    @torch.no_grad()
    def evaluate(ids, save=False):
        net.eval()
        by_budget = []
        for budget in [0, 2, 5, 10]:
            arrays = {k: [] for k in ['plugin', 'prediction', 'reference', 'capture_fraction', 'variances',
                                     'projected_energy', 'error_energy', 'image_mse']}
            accumulated = np.zeros(3)
            maximum_orthogonality, maximum_residual, degenerate = 0., 0., 0
            for offset in range(0, len(ids), args.batch):
                idx = ids[offset:offset + args.batch]
                b = context.batch(idx, evaluation_masks(idx, budget))
                loss, metrics, directions, variances, factors = loss_for(b, .1)
                accumulated += np.array([float(loss), float(metrics['direction_loss']),
                                          float(metrics['variance_loss'])]) * len(idx)
                error = (b['truth'] - b['mean']).flatten(1)
                projection = torch.einsum('bkd,bd->bk', directions, error).square()
                plugin = context.physical(b['mean'])
                prediction = plugin + context.factor_queries(factors)
                values = dict(plugin=plugin, prediction=prediction, reference=b['target'],
                              capture_fraction=metrics['captured_energy'] / metrics['error_energy'].clamp_min(1e-12),
                              variances=variances, projected_energy=projection, error_energy=metrics['error_energy'],
                              image_mse=error.square().mean(1))
                for key, value in values.items():
                    arrays[key].append(value.cpu().numpy())
                gram = directions @ directions.transpose(1, 2)
                maximum_orthogonality = max(maximum_orthogonality,
                    float((gram - torch.eye(args.rank, device=args.device)).abs().max()))
                maximum_residual = max(maximum_residual, float(b['cg_residual'].max()))
                degenerate += int((variances < 1e-12).sum())
            data = {k: np.concatenate(v) for k, v in arrays.items()}
            errors = np.abs(data['prediction'] / data['reference'] - 1)
            native = accumulated / len(ids)
            item = dict(additional_actions=budget, objects=len(ids), selection_loss=float(native[0]),
                        direction_loss=float(native[1]), variance_loss=float(native[2]),
                        mean_relative_error=float(errors.mean()),
                        plugin_relative_error=float(np.abs(data['plugin'] / data['reference'] - 1).mean()),
                        normalized_query_mse=float(np.mean(((data['prediction'] - data['reference']) /
                            context.query_scale.cpu().numpy())**2)),
                        residual_energy_capture=float(data['capture_fraction'].mean()),
                        mean_predicted_variance=data['variances'].mean(0).tolist(),
                        mean_projected_energy=data['projected_energy'].mean(0).tolist(),
                        image_raw_mse=float(data['image_mse'].mean()),
                        maximum_orthogonality_error=maximum_orthogonality,
                        maximum_cg_residual=maximum_residual, degenerate_components=degenerate,
                        per_query_relative_error=errors.mean(0).tolist())
            by_budget.append(item)
            if save:
                np.savez(args.out / f'assessment_budget{budget}.npz', ids=np.asarray(ids), **data)
        net.train()
        return dict(selection_loss=float(np.mean([v['selection_loss'] for v in by_budget])), by_budget=by_budget)

    for step in range(start + 1, args.steps + 1):
        ids = rng.choice(train_ids, args.batch, replace=False)
        extra = rng.integers(0, 11, args.batch)
        masks = masks_from_orders([rng.permutation(REMAINING) for _ in ids], extra)
        b = context.batch(ids, masks, training=True)
        weight = .1 * float(np.clip(-1 + 2 * (step - 1) / 200, 1e-6, 1))
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, _, _, _ = loss_for(b, weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite component loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError('Nonfinite component gradient')
        optimizer.step()
        if step == 1 or step % 25 == 0:
            state = dict(status='running', normalization=args.normalization, seed=args.seed, step=step,
                         total_steps=args.steps, loss=float(loss.detach()), variance_weight=weight,
                         direction_loss=float(metrics['direction_loss'].detach()),
                         variance_loss=float(metrics['variance_loss'].detach()), gradient_norm=float(gradient),
                         mean_capture_fraction=float((metrics['captured_energy'] /
                             metrics['error_energy'].clamp_min(1e-12)).mean()),
                         seconds=elapsed + time.perf_counter() - t0, peak_GiB=peak_memory(), pid=os.getpid())
            atomic_json(args.out / 'status.json', state)
            print(json.dumps(state), flush=True)
        if args.profile:
            state['status'] = 'profile_completed'
            atomic_json(args.out / 'profile.json', state)
            atomic_json(args.out / 'status.json', state)
            return
        if step % 400 == 0 or step == args.steps:
            validation = evaluate(context.split['validation'][:256])
            validation.update(step=step, seconds=elapsed + time.perf_counter() - t0)
            history.append(validation)
            improved = validation['selection_loss'] < best
            if improved:
                best = validation['selection_loss']
            checkpoint = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best,
                              seconds=validation['seconds'], history=history, numpy_rng=rng.bit_generator.state,
                              torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            temporary = args.out / 'last.partial.pt'
            torch.save(checkpoint, temporary)
            temporary.replace(args.out / 'last.pt')
            if improved:
                shutil.copy2(args.out / 'last.pt', args.out / 'best.pt')
            atomic_json(args.out / 'validation_history.json', history)
            print(json.dumps(dict(validation=validation, best=best)), flush=True)
    checkpoint = torch.load(args.out / 'best.pt', map_location=args.device, weights_only=True)
    net.load_state_dict(checkpoint['model'])
    report = dict(status='completed', normalization=args.normalization, seed=args.seed, best_step=checkpoint['step'],
                  assessment=evaluate(context.split['assessment'], save=True),
                  seconds=elapsed + time.perf_counter() - t0, peak_GiB=peak_memory(),
                  setup_seconds_current_invocation=setup_seconds,
                  total_seconds_current_invocation=time.perf_counter() - wall_start,
                  mean_checkpoint_sha256=context.mean_checkpoint_hash, covariance_training_objects=4000,
                  official_test_opened=False)
    atomic_json(args.out / 'summary.json', report)
    atomic_json(args.out / 'status.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
