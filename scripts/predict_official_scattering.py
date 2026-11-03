"""Save all frozen predictor outputs without importing a scoring reference."""
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
from src.frozen_scattering_predictor import FrozenScatteringPredictor
from src.independent_scattering_assessment import (
    NOISE_SEEDS, MULTIPLIERS, atomic_json, load_verified_freeze, model_specs)
from src.scattering_observation_features import FrozenScatteringFeatures


def predict_one(context, specification, cache, cache_manifest, destination, batch=16):
    cache, destination = Path(cache), Path(destination)
    destination.mkdir()
    ids = np.load(cache/'ids.npy', allow_pickle=False)
    model = FrozenScatteringPredictor(context, specification['kind'],
                                     specification['run'], specification['rank'])
    records = []
    for slot in cache_manifest['slots']:
        if slot['alias']:
            records.append(dict(slot=slot['name'], alias=slot['alias'],
                                order=slot['order'], budget=slot['budget']))
            continue
        name = slot['name']
        feature = np.load(cache/f'{name}_feature.npy', mmap_mode='r', allow_pickle=False)
        mean = np.load(cache/f'{name}_mean.npy', mmap_mode='r', allow_pickle=False)
        with np.load(cache/f'{name}_summary.npz', allow_pickle=False) as file:
            summary = {key:file[key].copy() for key in file.files}
        results = []
        for start in range(0, len(ids), batch):
            end = min(start+batch, len(ids))
            values = {key:torch.tensor(np.asarray(array[start:end]).copy(), device=context.device)
                      for key, array in [('feature', feature), ('mean', mean),
                          ('masks', summary['masks']), ('features66', summary['features66']),
                          ('mean_queries35', summary['mean_queries35'])]}
            results.append(model(values, slot['budget']).cpu().numpy())
        prediction = np.concatenate(results)
        if prediction.shape != (len(ids), 35) or not np.isfinite(prediction).all():
            raise FloatingPointError('Invalid predictions; no objects may be dropped')
        path = destination/f'{name}.npz'
        np.savez(path, ids=ids, prediction=prediction)
        records.append(dict(slot=name, order=slot['order'], budget=slot['budget'],
                            file=path.name, sha256=sha256(path),
                            nonpositive_predictions=int((prediction <= 0).sum())))
    return dict(specification=specification, artifact=model.metadata, predictions=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-seed', type=int, choices=NOISE_SEEDS, required=True)
    parser.add_argument('--multiplier', type=float, choices=MULTIPLIERS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    freeze = ROOT/'protocols/m3_official_assessment_frozen_v1.json'
    load_verified_freeze(ROOT, freeze)
    key = f'noise{args.noise_seed}_x{args.multiplier:g}'
    cache = ROOT/'data/official_scattering_v1/features'/key
    manifest_path = cache/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'completed' or manifest['freeze_sha256'] != sha256(freeze):
        raise RuntimeError('Cache is not complete under this freeze')
    if manifest['noise_seed'] != args.noise_seed or manifest['multiplier'] != args.multiplier:
        raise RuntimeError('Wrong observation setting')
    if sha256(cache/'ids.npy') != manifest['ids_sha256']:
        raise RuntimeError('Object identities changed')
    for slot in manifest['slots']:
        for name, expected in slot.get('files', {}).items():
            if sha256(cache/name) != expected:
                raise RuntimeError(f'Feature cache changed: {name}')
    out = ROOT/'runs/m3_official_assessment_v1'/key
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    context = FrozenScatteringFeatures(ROOT, args.device)
    for spec in model_specs():
        destination = out/spec['name']
        if destination.exists():
            raise RuntimeError(f'Preserve existing/partial predictions: {destination}')
        started = time.time()
        result = predict_one(context, spec, cache, manifest, destination, args.batch)
        result.update(status='completed', cache_manifest_sha256=sha256(manifest_path),
            freeze_sha256=sha256(freeze), noise_seed=args.noise_seed, multiplier=args.multiplier,
            device=args.device, torch_version=torch.__version__, batch=args.batch,
            completed_time=time.time(), seconds=time.time()-started, scoring_arrays_loaded=False)
        atomic_json(destination/'summary.json', result)
        print(json.dumps(dict(model=spec['name'], status='completed', seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
