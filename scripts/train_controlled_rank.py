"""Rank ablation with architecture-independent stochastic inputs (conditional M3)."""
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
from src.born_scattering import BatchedBorn, BornConfig, sha256
from src.controlled_query_learning import (batch_from_observation, coherent_response, current_response,
    draw_training_observation, independent_noise_generator, matched_rank_initialization,
    state_digest)
from src.mask_conditioned_learning import REMAINING, evaluation_masks, masks_from_orders
from src.mask_learning_context import MaskLearningContext
from src.response_learning import PhysicalQueries


def save_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, required=True, choices=[1, 2, 4, 8])
    p.add_argument('--readout', choices=['moment', 'coherent'], default='moment')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.readout == 'coherent' and args.rank != 1:
        raise ValueError('The coherent point control uses one field')
    response = current_response if args.readout == 'moment' else coherent_response
    gate_path = ROOT / 'audit/m2_readout_gate_20261101/decision.json'
    if not args.profile:
        gate = json.loads(gate_path.read_text())
        assert gate['status'] == 'completed'
        assert gate['next_action'] == 'run_m3_prediction_robustness'
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'summary.json').exists():
        raise RuntimeError('Completed output must be preserved')
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    c = MaskLearningContext(ROOT, ROOT/'runs/m2_mask_learning_20261101/image_seed20261117', args.device)
    net, init = matched_rank_initialization(args.rank, args.seed)
    net = net.to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(.9, .999))
    rng = np.random.default_rng(args.seed)
    noise_rng = independent_noise_generator(args.seed)
    train = np.asarray(c.split['covariance_train'])
    files = [Path(__file__)] + [ROOT/'src'/s for s in
        ['controlled_query_learning.py', 'posterior_components.py', 'mask_learning_context.py',
         'mask_conditioned_learning.py', 'response_learning.py', 'born_scattering.py']]
    config = dict(readout=args.readout, rank=args.rank, seed=args.seed, steps=4000, batch=128,
        profile=args.profile, device=args.device, host=socket.gethostname(),
        torch_version=torch.__version__, initialization=init,
        parameters=sum(v.numel() for v in net.parameters()),
        mean_checkpoint_sha256=c.mean_checkpoint_hash,
        source_hashes={str(f.relative_to(ROOT)): sha256(f) for f in files},
        protocol_sha256=sha256(ROOT/'protocols/m3_rank_learning_v1.json'),
        data_manifest_sha256=sha256(ROOT/'data/m2_mask_features_v1/manifest.json'),
        gate_sha256=sha256(gate_path) if gate_path.exists() else None)
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name, digest in config['source_hashes'].items():
        destination = snapshot/name.replace('/', '__')
        if not destination.exists():
            shutil.copy2(ROOT/name, destination)
        assert sha256(destination) == digest
    first, best, history, elapsed = 0, float('inf'), [], 0.
    if args.resume:
        old = json.loads((args.out/'config.json').read_text())
        for key in ['readout', 'rank', 'seed', 'steps', 'batch', 'initialization', 'mean_checkpoint_sha256',
                    'source_hashes', 'protocol_sha256', 'data_manifest_sha256', 'gate_sha256']:
            assert old[key] == config[key], key
        cp = torch.load(args.out/'last.pt', map_location=args.device, weights_only=True)
        net.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        first, best, history, elapsed = [cp[k] for k in ['step', 'best', 'history', 'seconds']]
        rng.bit_generator.state = cp['numpy_rng']
        noise_rng.bit_generator.state = cp['noise_rng']
    else:
        if (args.out/'config.json').exists():
            raise RuntimeError('Inspect a partial run before resuming')
        save_json(args.out/'config.json', config)
        torch.save(dict(model=net.state_dict(), initialization=init, rank=args.rank,
                        seed=args.seed, step=0), args.out/'initial.pt')
    setup = time.perf_counter() - started
    t0 = time.perf_counter()

    @torch.no_grad()
    def evaluate(ids, final=False):
        net.eval()
        new = None
        if final:
            new = PhysicalQueries([BatchedBorn(BornConfig(wave=w, numRec=60),
                cache_dir=ROOT/'data/born_cache', device=args.device).matrix for w in [5, 7, 10, 12]])
            new_targets = np.load(ROOT/'data/m2_transfer_queries_v1/queries.npy', mmap_mode='r')
        reports = []
        for budget in ([0, 2, 5, 9, 10] if final else [0, 2, 5, 9]):
            predictions, references = [], []
            for j in range(0, len(ids), 128):
                index = ids[j:j+128]
                b = batch_from_observation(c, index, evaluation_masks(index, budget))
                factors = net.differentiable_factors(b['feature'], b['mean'], b['mask'])
                pred = response(b['mean'], factors, c.physical)
                ref = b['target']
                if final:
                    pred = torch.cat([pred, response(b['mean'], factors, new)], 1)
                    ref = torch.cat([ref, torch.tensor(np.asarray(new_targets[index]).copy(), device=args.device)], 1)
                predictions.append(pred.cpu().numpy())
                references.append(ref.cpu().numpy())
            pred, ref = np.concatenate(predictions), np.concatenate(references)
            assert np.isfinite(pred).all() and (pred >= 0).all()
            err = np.abs(pred/ref - 1)
            row = dict(budget=budget, objects=len(ids), original_mre=float(err[:, :15].mean()),
                zero_predictions=int((pred == 0).sum()),
                per_query_mse=np.mean((pred-ref)**2, axis=0).tolist(),
                per_query_relative_rmse=np.sqrt(np.mean((pred/ref-1)**2, axis=0)).tolist(),
                original_normalized_mse=float(np.mean(((pred[:, :15]-ref[:, :15])/c.query_scale.cpu().numpy())**2)))
            if final:
                row.update(new_mre=float(err[:, 15:].mean()), interpolation_mre=float(err[:, 15:25].mean()),
                           extrapolation_mre=float(err[:, 25:].mean()))
                np.savez(args.out/f'budget{budget}.npz', ids=np.asarray(ids), prediction=pred, reference=ref)
            reports.append(row)
        net.train()
        return reports

    for step in range(first+1, 3 if args.profile else 4001):
        ids = rng.choice(train, 128, replace=False)
        extra = rng.integers(0, 10, 128)
        masks = masks_from_orders([rng.permutation(REMAINING) for _ in ids], extra)
        _ = [rng.choice(np.flatnonzero(v == 0)) for v in masks]
        observation = draw_training_observation(c, ids, noise_rng)
        if step == 1:
            save_json(args.out/'first_input_hashes.json', dict(ids=state_digest({'ids': torch.tensor(ids)}),
                masks=state_digest({'masks': torch.tensor(masks)}),
                observation=state_digest({'observation': torch.tensor(observation)})))
        b = batch_from_observation(c, ids, masks, observation)
        factors = net.differentiable_factors(b['feature'], b['mean'], b['mask'])
        pred = response(b['mean'], factors, c.physical)
        loss = ((pred-b['target'])/c.query_scale).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if step in [1, 2] or step % 25 == 0:
            status = dict(status='running', rank=args.rank, seed=args.seed, step=step,
                loss=float(loss.detach()), gradient=float(gradient), pid=os.getpid(),
                seconds=elapsed+time.perf_counter()-t0, peak_GiB=torch.cuda.max_memory_allocated(args.device)/2**30)
            save_json(args.out/'status.json', status)
            print(json.dumps(status), flush=True)
        if args.profile and step == 2:
            status['status'] = 'profile_completed'
            save_json(args.out/'profile.json', status)
            return
        if step % 400 == 0:
            results = evaluate(c.split['validation'][:256])
            score = float(np.mean([r['original_normalized_mse'] for r in results]))
            history.append(dict(step=step, selection_loss=score, results=results))
            improved = score < best
            best = min(best, score)
            cp = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), step=step,
                best=best, history=history, seconds=elapsed+time.perf_counter()-t0,
                numpy_rng=rng.bit_generator.state, noise_rng=noise_rng.bit_generator.state)
            tmp = args.out/'last.partial.pt'
            torch.save(cp, tmp)
            tmp.replace(args.out/'last.pt')
            if improved:
                shutil.copy2(args.out/'last.pt', args.out/'best.pt')
            save_json(args.out/'validation_history.json', history)
            print(json.dumps(history[-1]), flush=True)
    cp = torch.load(args.out/'best.pt', map_location=args.device, weights_only=True)
    net.load_state_dict(cp['model'])
    summary = dict(status='completed', readout=args.readout, rank=args.rank, seed=args.seed, best_step=cp['step'],
        results=evaluate(c.split['assessment'], final=True), parameters=config['parameters'],
        seconds=elapsed+time.perf_counter()-t0, setup_seconds=setup,
        peak_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
        config_sha256=sha256(args.out/'config.json'), checkpoint_sha256=sha256(args.out/'best.pt'),
        official_test_opened=False)
    save_json(args.out/'summary.json', summary)
    save_json(args.out/'status.json', dict(status='completed', best_step=cp['step']))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
