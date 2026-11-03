"""Build frozen features from observations; never load scoring targets/images."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import sha256
from src.independent_scattering_assessment import (
    NOISE_SEEDS, MULTIPLIERS, atomic_json, load_verified_freeze, mask_channels, masks_for_ids, slots)
from src.scattering_observation_features import FrozenScatteringFeatures


def build_cache(context, ids, observation, directory, batch=16):
    directory = Path(directory)
    directory.mkdir()
    if observation.shape != (len(ids), 7200) or not np.isfinite(observation).all():
        raise ValueError('Invalid field observations')
    records = []
    for slot in slots():
        if slot['alias']:
            records.append(slot)
            continue
        name = slot['name']
        mask = masks_for_ids(ids, slot['budget'], slot['order'])
        visible = np.where(mask_channels(mask), observation, 0.)
        arrays = {label:np.lib.format.open_memmap(directory/f'{name}_{label}.npy', mode='w+',
            dtype=np.float32, shape=(len(ids), 1, 128, 128)) for label in ['feature', 'mean']}
        summaries, responses, residuals = [], [], []
        for start in range(0, len(ids), batch):
            stop = min(start+batch, len(ids))
            values = context(visible[start:stop], mask[start:stop])
            for label in arrays:
                arrays[label][start:stop] = values[label].cpu().numpy()
            summaries.append(values['features66'].cpu().numpy())
            responses.append(values['mean_queries35'].cpu().numpy())
            residuals.append(values['cg_residual'].cpu().numpy())
        for array in arrays.values():
            array.flush()
        del arrays
        np.savez(directory/f'{name}_summary.npz', masks=mask, features66=np.concatenate(summaries),
            mean_queries35=np.concatenate(responses), cg_residual=np.concatenate(residuals))
        files = [directory/f'{name}_{v}.npy' for v in ['feature', 'mean']]+[directory/f'{name}_summary.npz']
        records.append(dict(slot, files={p.name:sha256(p) for p in files}))
    np.save(directory/'ids.npy', ids)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-seed', type=int, choices=NOISE_SEEDS, required=True)
    parser.add_argument('--multiplier', type=float, choices=MULTIPLIERS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    freeze = ROOT/'protocols/m3_official_assessment_frozen_v1.json'
    load_verified_freeze(ROOT, freeze)
    base = ROOT/'data/official_scattering_v1'
    manifest = json.loads((base/'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['freeze_sha256'] != sha256(freeze):
        raise RuntimeError('Acquisition and final freeze disagree')
    record = next(v for v in manifest['observations'] if v['noise_seed'] == args.noise_seed and v['multiplier'] == args.multiplier)
    path = base/record['file']
    if sha256(path) != record['sha256'] or sha256(base/'ids.npy') != manifest['ids_sha256']:
        raise RuntimeError('Observation artifacts changed')
    ids = np.load(base/'ids.npy', allow_pickle=False)
    if not np.array_equal(ids, np.arange(100)):
        raise ValueError('All100 official IDs required')
    observation = np.load(path, allow_pickle=False)
    destination = base/'features'/path.stem
    if destination.exists():
        raise RuntimeError('Preserve completed/partial official features')
    destination.parent.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    context = FrozenScatteringFeatures(ROOT, args.device)
    records = build_cache(context, ids, observation, destination, args.batch)
    atomic_json(destination/'manifest.json', dict(status='completed', objects=len(ids), slots=records,
        freeze_sha256=sha256(freeze), observation_sha256=record['sha256'],
        ids_sha256=sha256(destination/'ids.npy'), noise_seed=args.noise_seed, multiplier=args.multiplier,
        feature_contract=context.metadata, seconds=time.time()-started, device=args.device,
        scoring_arrays_loaded=False))
    print(json.dumps(dict(status='completed', cache=str(destination), seconds=time.time()-started)))


if __name__ == '__main__':
    main()
