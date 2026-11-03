"""Condense frozen results for the self-contained main paper; no model fitting."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / name).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    source = 'analysis/m7_nppc_v1/groups.csv'
    nppc = read(source)
    official = read('analysis/m3_official_assessment_v1/groups.csv')
    comparisons = read('analysis/m3_official_assessment_v1/comparisons.csv')
    budgets = [0, 2, 5, 10]
    lookup = {(r['family'], int(r['budget'])): float(r['mre'])
              for r in nppc if r['queries'] == 'new'}
    rows = []
    for variant, label in [('raw', 'Native image objective'),
                           ('rescaled', 'Rescaled variance'),
                           ('constant', 'Constant variance'),
                           ('rescaled_tail', 'Rescaled + isotropic tail'),
                           ('constant_tail', 'Constant + isotropic tail')]:
        values = [min(lookup[f'nppc_{mode}_{variant}', b] for mode in ['sample', 'fixed'])
                  for b in budgets]
        rows.append(dict(variant=variant, label=label, mre=values))
    response = [next(float(r['mean']) for r in official
                     if r['family'] == 'primary_moment8' and int(r['budget']) == b
                     and r['multiplier'] == '1.0' and r['mask_order'] == '0'
                     and r['queries'] == 'new_combined') for b in budgets]
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Readout & 30 & 42 & 60 & 90 \\', r'\midrule']
    for row in rows:
        lines.append(row['label'] + ' & ' + ' & '.join(f'{100*x:.2f}' for x in row['mre']) + r' \\')
    lines += [r'\midrule', 'Response moments & ' + ' & '.join(f'{100*x:.2f}' for x in response) + r' \\',
              r'\bottomrule', r'\end{tabular}']
    (ROOT / 'paper/tables/variance_calibration.tex').write_text('\n'.join(lines) + '\n')
    robustness = []
    for baseline in ['ridge_budget', 'ridge_pooled', 'direct_additive_original',
                     'direct_multiplicative_original', 'direct_additive_headrate',
                     'direct_multiplicative_headrate', 'joint']:
        selected = [r for r in comparisons if r['left'] == 'primary_moment8'
                    and r['right'] == baseline and r['queries'] == 'new_combined'
                    and not (r['mask_order'] == '1' and r['budget'] == '0')]
        assert len(selected) == 21
        gain = [-100*float(r['mean_difference']) for r in selected]
        robustness.append(dict(baseline=baseline, settings=len(selected),
            improvements=sum(x > 0 for x in gain),
            pointwise_intervals_above_zero=sum(json.loads(r['object_bootstrap95'])[1] < 0 for r in selected),
            gain_percentage_points=[min(gain), max(gain)]))
    familiar_vs_new = {}
    for family in ['ridge_budget', 'primary_moment8', 'direct_additive_original', 'direct_additive_headrate']:
        familiar_vs_new[family] = {q: next(float(r['mean']) for r in official
            if r['family'] == family and r['queries'] == q and r['budget'] == '0'
            and r['multiplier'] == '1.0' and r['mask_order'] == '0')
            for q in ['original', 'new_combined']}
    seed_sd = {b: next(float(r['training_seed_sd']) for r in official
        if r['family'] == 'primary_moment8' and r['queries'] == 'new_combined'
        and int(r['budget']) == b and r['multiplier'] == '1.0' and r['mask_order'] == '0') for b in budgets}
    morphology_path = 'analysis/m7_morphology_v1/groups.csv'
    morphology = read(morphology_path)
    scope_morphology = {cohort: {family: next(float(r['mre']) for r in morphology
        if r['cohort'] == cohort and r['family'] == family and r['budget'] == '0'
        and r['queries'] == 'new') for family in ['primary_moment8', 'image_only']}
        for cohort in ['inclusions', 'perforated']}
    real_path = 'analysis/m7_real_moments_v1/metrics.csv'
    real = read(real_path); scope_real = {}
    for probe in ['fixed', 'free']:
        scope_real[probe] = {}
        for kind in ['moment', 'moment_no_factors']:
            values = [float(r['rmse']) for r in real if r['probe'] == probe
                      and r['kind'] == kind and r['pool'] == '1']
            assert len(values) == 18
            scope_real[probe][kind] = statistics.mean(values)
    paths = [source, 'analysis/m3_official_assessment_v1/groups.csv',
             'analysis/m3_official_assessment_v1/comparisons.csv', morphology_path, real_path]
    out = ROOT / 'analysis/m10_main_evidence_v1'; out.mkdir(exist_ok=True)
    data = dict(calibration_rows=rows, response_mre=response, receiver_counts=[30, 42, 60, 90],
        calibration_summary='Per-budget minimum over the two normalization variants; an optimistic comparison envelope, not a deployable test-selected model.',
        robustness=robustness,
        familiar_vs_new=familiar_vs_new, training_seed_sd=seed_sd,
        scope_morphology=scope_morphology, scope_real=scope_real,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (out/'evidence.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(dict(table='paper/tables/variance_calibration.tex', robustness=robustness), indent=2))


if __name__ == '__main__':
    main()
