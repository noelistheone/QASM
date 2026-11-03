"""Score only after every declared frozen prediction has been saved.

Model errors are averaged across training/noise seeds, not their predictions.
Uncertainty resamples100 object IDs; repeated observations do not expand n.
"""
from pathlib import Path
import csv
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import sha256
from src.independent_scattering_assessment import (
    BUDGETS, MULTIPLIERS, NOISE_SEEDS, SLICES, atomic_json, load_verified_freeze,
    model_specs, paired_summary)


def collect_predictions(root, specifications, expected_ids, freeze_hash):
    """Load/verify prediction-only files before any target file is opened."""
    records, hashes = {}, {}
    for spec in specifications:
        for seed in NOISE_SEEDS:
            for multiplier in MULTIPLIERS:
                folder = root/f'runs/m3_official_assessment_v1/noise{seed}_x{multiplier:g}'/spec['name']
                summary_path = folder/'summary.json'
                summary = json.loads(summary_path.read_text())
                if (summary['status'] != 'completed' or summary['specification'] != spec
                        or summary['freeze_sha256'] != freeze_hash or summary['scoring_arrays_loaded']):
                    raise RuntimeError(f'Invalid completed prediction summary: {summary_path}')
                hashes[str(summary_path.relative_to(root))] = sha256(summary_path)
                slot_records = {r['slot']:r for r in summary['predictions']}
                for order in [0, 1]:
                    for budget in BUDGETS:
                        name = f'mask{order}_budget{budget}'
                        item = slot_records[name]
                        if item.get('alias'):
                            item = slot_records[item['alias']]
                        path = folder/item['file']
                        if sha256(path) != item['sha256']:
                            raise RuntimeError(f'Prediction changed after completion: {path}')
                        with np.load(path, allow_pickle=False) as data:
                            if set(data.files) != {'ids', 'prediction'}:
                                raise ValueError('Prediction artifact contains undeclared arrays')
                            ids, prediction = data['ids'], data['prediction']
                        if not np.array_equal(ids, expected_ids) or prediction.shape != (len(ids), 35):
                            raise ValueError('All declared objects and queries must be preserved')
                        if not np.isfinite(prediction).all():
                            raise ValueError('Nonfinite predictor; retain failure rather than dropping objects')
                        records[(spec['name'], seed, multiplier, order, budget)] = prediction
                        hashes[str(path.relative_to(root))] = item['sha256']
    return records, hashes


def aggregate(records, specifications, reference, draws):
    groups = {}
    for spec in specifications:
        groups.setdefault(spec['family'], []).append(spec)
    rows, individual, object_errors = [], [], {}
    for family, models in groups.items():
        for multiplier in MULTIPLIERS:
            for order in [0, 1]:
                for budget in BUDGETS:
                    relative_by_seed = []
                    for model in models:
                        noise_errors = []
                        for seed in NOISE_SEEDS:
                            prediction = records[(model['name'], seed, multiplier, order, budget)]
                            relative = prediction/reference-1
                            noise_errors.append(np.abs(relative))
                            individual.append(dict(model=model['name'], family=family, seed=model['seed'],
                                noise_seed=seed, multiplier=multiplier, mask_order=order, budget=budget,
                                **{f'{name}_mre':float(np.abs(relative[:,section]).mean()) for name,section in SLICES.items()},
                                per_query_mse=np.mean((prediction-reference)**2, axis=0).tolist(),
                                per_query_relative_rmse=np.sqrt(np.mean(relative**2, axis=0)).tolist(),
                                nonpositive_predictions=int((prediction <= 0).sum())))
                        relative_by_seed.append(noise_errors)
                    errors = np.asarray(relative_by_seed)  # model,noise,object,query
                    for query, section in SLICES.items():
                        per_model_noise_object = errors[:,:,:,section].mean(-1)
                        per_object = per_model_noise_object.mean((0, 1))
                        training_means = per_model_noise_object.mean((1, 2))
                        noise_means = per_model_noise_object.mean((0, 2))
                        object_errors[(family,multiplier,order,budget,query)] = per_object
                        rows.append(dict(family=family, multiplier=multiplier, mask_order=order,
                            budget=budget, queries=query, mean=float(per_object.mean()), objects=len(reference),
                            training_seed_means=training_means.tolist(), noise_seed_means=noise_means.tolist(),
                            training_seed_sd=float(training_means.std(ddof=1)) if len(models)>1 else None,
                            object_bootstrap95=np.quantile(per_object[draws].mean(1),[.025,.975]).tolist()))
    comparisons = []
    for family in groups:
        if family == 'primary_moment8':
            continue
        for multiplier in MULTIPLIERS:
            for order in [0, 1]:
                for budget in BUDGETS:
                    for query in SLICES:
                        key = (multiplier,order,budget,query)
                        comparisons.append(dict(left='primary_moment8', right=family,
                            multiplier=multiplier, mask_order=order, budget=budget, queries=query,
                            **paired_summary(object_errors[('primary_moment8',*key)],
                                             object_errors[(family,*key)],draws)))
    return dict(groups=rows, comparisons=comparisons, all_replicates=individual)


def main():
    freeze_path = ROOT/'protocols/m3_official_assessment_frozen_v1.json'
    freeze = load_verified_freeze(ROOT, freeze_path)
    out = ROOT/'analysis/m3_official_assessment_v1'
    if out.exists():
        raise RuntimeError('Preserve existing complete/partial official scoring')
    started = time.time()
    specifications = model_specs()
    ids = np.arange(100)
    records, prediction_hashes = collect_predictions(ROOT, specifications, ids, sha256(freeze_path))
    # Every prediction has now been verified; only here may scoring read truth.
    base = ROOT/'data/official_scattering_v1'
    manifest = json.loads((base/'manifest.json').read_text())
    reference_path = base/'scoring/targets35.npy'
    if (sha256(reference_path) != manifest['scoring_hashes']['targets35.npy']
            or manifest['freeze_sha256'] != sha256(freeze_path)):
        raise RuntimeError('Scoring reference and frozen preparation disagree')
    reference = np.load(reference_path, allow_pickle=False)
    if reference.shape != (100,35) or not (np.isfinite(reference).all() and (reference>0).all()):
        raise ValueError('All100 positive-response references required for frozen relative metrics')
    draws = np.random.default_rng(freeze['bootstrap_seed']).integers(0, 100, (freeze['bootstrap_draws'],100))
    result = aggregate(records, specifications, reference, draws)
    result.update(status='completed', freeze_sha256=sha256(freeze_path), objects=100,
        predictor_count=len(specifications), reference_sha256=sha256(reference_path),
        prediction_hashes=prediction_hashes, seconds=time.time()-started,
        primary='new_combined;nominalnoise;mask0;allfourbudgets;primary_moment8',
        interval_scope='Conditional object-paired pointwise bootstrap; no multiple-comparison correction; training seeds and noise are averaged errors, not extra objects.',
        model_selection_performed=False)
    out.mkdir(parents=True)
    atomic_json(out/'summary.json', result)
    for name, rows in [('groups',result['groups']),('comparisons',result['comparisons'])]:
        with (out/f'{name}.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='completed',objects=100,groups=len(result['groups']),contrasts=len(result['comparisons']))))


if __name__ == '__main__':
    main()
