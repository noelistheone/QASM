"""Score completed fixed/gauge controls and retain ungauged outcomes separately."""
from pathlib import Path
import csv
import itertools
import json
import sys
import time

import h5py
import numpy as np
from scipy.special import xlogy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_broadband_extended import digest, pool, save

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol_path = ROOT / 'protocols/m4_broadband_strengthened_scoring_v1.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['script_sha256'] == digest(Path(__file__))
    initial = json.loads((ROOT / 'protocols/m4_broadband_transfer_v1.json').read_text())
    extended_path = ROOT / 'protocols/m4_broadband_extended_v1.json'
    gauge_path = ROOT / 'protocols/m4_broadband_gauge_v1.json'
    assert digest(extended_path) == protocol['fixed_execution_sha256']
    assert digest(gauge_path) == protocol['gauge_execution_sha256']
    extended = json.loads(extended_path.read_text())
    families = ['uniform_unbounded3', 'uniform_bounded3'] + [x['name'] for x in initial['models']]
    families += [f'smooth_{s:g}' for s in extended['smoothing_sigmas']] + ['poisson_gain', 'l2_gain']
    expected = {f'native_{e}' for e in extended['native_checkpoints']}
    expected |= {f'{m}_{e}' for m in families for e in extended['refinement_checkpoints']}
    assert len(expected) == 30
    experiments, inventory, outcomes = [], [], []
    for cohort, probes in [('primary', ['fixed', 'free']), ('original_ungauged', ['free'])]:
        for energy, count, probe in itertools.product([8, 12], [60, 150, 300], probes):
            gauge = cohort == 'primary' and probe == 'free'
            phase = 'm4_broadband_gauge_v1' if gauge else 'm4_broadband_extended_v1'
            folder = ROOT / 'runs' / phase / f'e{energy}_n{count}_{probe}'
            status = json.loads((folder / 'status.json').read_text())
            assert status['status'] in ['completed', 'failed'], (folder, status)
            report_path = folder / 'report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            identity = dict(cohort=cohort, energy=energy, visible_count=count,
                            probe=probe, gauge_fixed=gauge)
            entries = []
            if report is not None:
                assert report['status'] == 'completed' and report['native_epochs'] == 5000
                assert not report['query_values_read']
                assert report['protocol_sha256'] == digest(gauge_path if gauge else extended_path)
                entries = report['predictions']
                assert {p['method'] for p in entries} == expected
                for entry in entries:
                    assert digest(ROOT / entry['path']) == entry['sha256']
            else:
                assert cohort == 'original_ungauged' and status['status'] == 'failed'
                for path in sorted(folder.glob('*.npy')):
                    assert path.stem in expected
                    entries.append(dict(method=path.stem, path=str(path.relative_to(ROOT)), sha256=digest(path)))
            for entry in entries:
                array = np.load(ROOT / entry['path'], mmap_mode='r', allow_pickle=False)
                assert array.shape == (120, 256, 256) and np.isfinite(array).all() and (array >= 0).all()
                inventory.append(dict(**identity, **entry))
            missing = sorted(expected - {p['method'] for p in entries})
            if cohort == 'primary':
                assert not missing
            outcome = dict(**identity, status=status['status'], predictions=len(entries),
                           missing_methods=missing, error=status.get('error', ''), folder=str(folder.relative_to(ROOT)))
            outcomes.append(outcome)
            checkpoints = report['checkpoints'] if report else [json.loads(p.read_text()) for p in sorted(folder.glob('checkpoint_*.json'))]
            experiments.append(dict(**identity, predictions=entries, checkpoints=checkpoints,
                                    refinements=report['refinements'] if report else [], host=status['host']))
    assert sum(x['cohort'] == 'primary' for x in inventory) == 360
    out = ROOT / 'analysis/m4_broadband_strengthened_v1'
    out.mkdir(exist_ok=False)
    save(out / 'verified_predictions.json', dict(created=time.time(), predictions=inventory,
        scoring_sha256=digest(protocol_path), query_previously_observed=True))
    save(out / 'trajectory_outcomes.json', outcomes)
    targets = {}
    with h5py.File(ROOT / initial['raw_path'], 'r') as h:
        for energy in [8, 12]:
            counts = h['data/frames'][energy, 1, np.asarray(initial['query_ids']), :, :]
            counts = counts[:, ::-1, ::-1][:, :256, 8:264].astype(np.float64)
            targets[energy] = {k: pool(counts, k) for k in [1, 4, 16]}
    rows = []
    for experiment in experiments:
        energy = experiment['energy']
        native_cost = {c['epoch']: c['native_seconds'] for c in experiment['checkpoints']}
        update_cost = {c['method']: c['algorithm_seconds'] for c in experiment['refinements']}
        path_by_method = {p['method']: ROOT / p['path'] for p in experiment['predictions']}
        references = {}
        for epoch in [1000, 5000]:
            key = f'native_{epoch}'
            if key in path_by_method:
                array = np.load(path_by_method[key], allow_pickle=False).astype(np.float64)
                references[epoch] = {k: pool(array, k) for k in [1, 4, 16]}
        for entry in experiment['predictions']:
            method = entry['method']
            epoch = int(method.rsplit('_', 1)[1])
            array = np.load(ROOT / entry['path'], allow_pickle=False).astype(np.float64)
            for k in [1, 4, 16]:
                prediction, target = pool(array, k), targets[energy][k]
                error = (prediction - target) ** 2
                row = {key: experiment[key] for key in ['cohort', 'energy', 'visible_count', 'probe', 'gauge_fixed']}
                row.update(method=method, epoch=epoch, pool=k,
                    relative_intensity_rmse=float(np.sqrt(error.sum() / (target * target).sum())),
                    signed_poisson_corrected_mse=float(np.mean(error - target)),
                    poisson_deviance_per_count=float(2 * np.sum(prediction-target+xlogy(target, target/np.maximum(prediction, 1e-12))) / target.sum()),
                    predicted_to_observed_fluence=float(prediction.sum() / target.sum()),
                    native_seconds_to_checkpoint=native_cost.get(epoch, ''),
                    extra_update_seconds=update_cost.get(method, ''), host=experiment['host'])
                for reference_epoch in [1000, 5000]:
                    diff_key = f'paired_mse_difference_to_native{reference_epoch}'
                    se_key = f'conditional_poisson_noise_se_vs_native{reference_epoch}'
                    row[diff_key], row[se_key] = '', ''
                    if reference_epoch in references:
                        reference = references[reference_epoch][k]
                        delta = prediction - reference
                        row[diff_key] = float(np.mean(error - (reference-target)**2))
                        row[se_key] = float(np.sqrt(np.sum(4*delta*delta*target)) / target.size)
                rows.append(row)
        print(json.dumps({key: experiment[key] for key in ['cohort', 'energy', 'visible_count', 'probe']}), flush=True)
    primary = [r for r in rows if r['cohort'] == 'primary']
    secondary = [r for r in rows if r['cohort'] == 'original_ungauged']
    assert len(primary) == 1080
    write_csv(out / 'all_metrics.csv', primary)
    write_csv(out / 'original_ungauged_metrics.csv', secondary)
    save(out / 'report.json', dict(status='completed', primary_fits=12, primary_predictions=360,
        primary_rows=len(primary), original_ungauged_rows=len(secondary), outcomes=outcomes,
        protocol_sha256=digest(protocol_path),
        scope='Retrospective strengthening on previously scored view 1. Six unchanged fixed-probe trajectories and six native power-gauge joint-probe trajectories; all frozen neural variants and controls retained.',
        se_scope='Conditional independent-Poisson pixel noise for fixed predictions; not independent sample/acquisition variability.',
        cost_scope='Within-trajectory native-plus-update timing, excluding I/O and readout; original ungauged diagnostic-overlap timings are not isolated benchmarks.'))
    print(json.dumps(dict(status='completed',primary_rows=len(primary),secondary_rows=len(secondary))))


if __name__ == '__main__':
    main()
