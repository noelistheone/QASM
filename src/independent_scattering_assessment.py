"""Data-independent contracts for the prepared official scattering assessment.

These helpers do not download/open a test set, train or choose models.
"""
from pathlib import Path
import json

import numpy as np

from src.born_scattering import sha256
from src.mask_conditioned_learning import INITIAL, REMAINING


SEEDS = (20261126, 20261127, 20261128)
NOISE_SEEDS = (20270111, 20270112, 20270113)
MULTIPLIERS = (.5, 1., 2.)
BUDGETS = (0, 2, 5, 10)
NAMESPACE = 20270101
SLICES = {'original': slice(0, 15), 'interpolation': slice(15, 25),
          'extrapolation': slice(25, 35), 'new_combined': slice(15, 35)}


def model_specs():
    result = []
    for seed in SEEDS:
        result.append(dict(name=f'primary_moment8_seed{seed}', family='primary_moment8',
            kind='moment', rank=8, seed=seed,
            run=f'runs/m2_random_current_20261101/random_current_seed{seed}'))
    for kind, rank in [('moment', 1), ('moment', 2), ('moment', 4), ('moment', 8), ('coherent', 1)]:
        family = f'matched_{kind}{rank}'
        for seed in SEEDS:
            result.append(dict(name=f'{family}_seed{seed}', family=family, kind=kind,
                rank=rank, seed=seed,
                run=f'runs/m3_rank_learning_20261101/{kind}_rank{rank}_seed{seed}'))
    configs = [('multiplicative', 'original', 'm2_direct_query_cnn_20261101', 'direct'),
               ('additive', 'original', 'm2_direct_query_cnn_additive_20261101', 'direct'),
               ('multiplicative', 'headrate', 'm2_direct_query_headrate_20261101', 'multiplicative'),
               ('additive', 'headrate', 'm2_direct_query_headrate_20261101', 'additive')]
    for kind, rate, folder, prefix in configs:
        family = f'direct_{kind}_{rate}'
        for seed in SEEDS:
            result.append(dict(name=f'{family}_seed{seed}', family=family, kind=kind,
                rank=8, seed=seed, run=f'runs/{folder}/{prefix}_seed{seed}'))
    for seed in [20261117, 20261118, 20261119]:
        result.append(dict(name=f'joint_seed{seed}', family='joint', kind='joint',
            rank=8, seed=seed, run=f'runs/m2_mask_learning_20261101/joint_seed{seed}'))
    for name in ['image_only', 'ridge_budget', 'ridge_pooled']:
        result.append(dict(name=name, family=name, kind=name, rank=8, seed=None, run=None))
    assert len(result) == len({v['name'] for v in result}) == 36
    return result


def masks_for_ids(ids, budget, order):
    ids = np.asarray(ids)
    if budget not in BUDGETS or order not in [0, 1]:
        raise ValueError('Undeclared mask or measurement budget')
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
        raise ValueError('Nonnegative integer object IDs required')
    masks = np.zeros((len(ids), 30), dtype=np.float32)
    masks[:, list(INITIAL)] = 1
    for row, obj in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([NAMESPACE, 902+order, int(obj)]))
        masks[row, rng.permutation(REMAINING)[:budget]] = 1
    return masks


def observation_noise(ids, seed, multiplier):
    if seed not in NOISE_SEEDS or multiplier not in MULTIPLIERS:
        raise ValueError('Undeclared official noise setting')
    values = []
    for obj in ids:
        if int(obj) != obj or obj < 0:
            raise ValueError('Nonnegative integer object IDs required')
        rng = np.random.default_rng(np.random.SeedSequence([NAMESPACE, seed, int(obj)]))
        values.append(rng.standard_normal(7200).astype(np.float32))
    return np.stack(values)*(multiplier*1e-4/np.sqrt(2))


def mask_channels(masks):
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim != 2 or masks.shape[1] != 30 or not np.isin(masks, [0, 1]).all():
        raise ValueError('Expected binary receiver-block masks')
    return np.broadcast_to(np.repeat(masks, 6, axis=1)[:, None, :, None],
                           (len(masks), 20, 180, 2)).reshape(len(masks), 7200).astype(bool)


def slots():
    return [dict(name=f'mask{o}_budget{b}', order=o, budget=b,
                 alias='mask0_budget0' if o == 1 and b == 0 else None)
            for o in [0, 1] for b in BUDGETS]


def load_verified_freeze(root, path):
    root, path = Path(root), Path(path)
    freeze = json.loads(path.read_text())
    if freeze.get('status') != 'frozen_before_official_acquisition':
        raise RuntimeError('Final assessment freeze is absent or incomplete')
    if freeze.get('official_test_downloaded') is not False or freeze.get('official_test_opened') is not False:
        raise RuntimeError('This is not a pre-acquisition freeze')
    if freeze['models'] != model_specs():
        raise RuntimeError('Frozen comparator set changed')
    for label in ['code_hashes', 'artifact_hashes', 'prerequisite_hashes']:
        for relative, expected in freeze[label].items():
            if sha256(root/relative) != expected:
                raise RuntimeError(f'Frozen assessment dependency changed: {relative}')
    return freeze


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def paired_summary(left, right, draws):
    difference = np.asarray(left)-np.asarray(right)
    if difference.ndim != 1:
        raise ValueError('Exactly one aggregated error per object required')
    bootstrap = difference[draws].mean(1)
    return dict(mean_difference=float(difference.mean()),
                object_bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
                objects_improved=int((difference < 0).sum()))
