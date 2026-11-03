"""Freeze executable artifacts only after complete M3 scientific review.

This command never downloads or opens the official archive. It must be run after
the M3 gate has been written from the complete analyses and isolated costs.
"""
from pathlib import Path
import ast
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import sha256
from src.independent_scattering_assessment import model_specs


def source_closure(initial):
    pending, visited = list(initial), set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) else (
                [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name and name.startswith('src.'):
                    pending.append(ROOT/(name.replace('.', '/')+'.py'))
    return sorted(visited)


def main():
    out = ROOT/'protocols/m3_official_assessment_frozen_v1.json'
    if out.exists():
        raise RuntimeError('Preserve the existing final assessment freeze')
    gate_path = ROOT/'audit/m3_prediction_gate_20261101/decision.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('status') != 'completed' or gate.get('next_action') != 'run_independent_official_assessment':
        raise RuntimeError('Full M3 scientific review is required')
    if not gate.get('scientific_review'):
        raise RuntimeError('The gate must record interpretation of complete evidence')
    complete_path = ROOT/'runs/m3_completion_20261101/status.json'
    complete = json.loads(complete_path.read_text())
    if complete['status'] != 'completed_artifacts_ready_for_scientific_review':
        raise RuntimeError('M3 completion queue has not finished')
    prereqs = [gate_path, complete_path,
               ROOT/'protocols/m3_official_assessment_preparation_v1.json',
               ROOT/'audit/m2_readout_gate_20261101/decision.json',
               ROOT/'analysis/m2_readout_gate_20261101/summary.json']
    for name, path in [('rank', ROOT/'analysis/m3_rank_20261101/summary.json'),
                       ('robustness', ROOT/'analysis/m3_robustness_20261101/summary.json'),
                       ('isolated_cost', ROOT/'runs/m3_inference_cost_20261101/summary.json')]:
        value = json.loads(path.read_text())
        if value['status'] != 'completed' or value.get('pending'):
            raise RuntimeError(f'Incomplete prerequisite: {name}')
        expected = complete['completed_steps'][name]['summary_sha256']
        if sha256(path) != expected or gate['reviewed_summary_hashes'][name] != expected:
            raise RuntimeError(f'Reviewed evidence changed: {name}')
        if name == 'isolated_cost' and len(value['results']) != 88:
            raise RuntimeError('All88 timing cells are required')
        prereqs.append(path)
    preparation = json.loads(prereqs[2].read_text())
    archive = ROOT/'sources/resources/official_assessment_v1/inv-scatter-test.zip'
    if archive.exists() or (ROOT/'data/official_scattering_v1').exists():
        raise RuntimeError('Official acquisition already exists; audit before claiming pre-acquisition freeze')
    specs = model_specs()
    artifacts = [ROOT/'data/m2_mask_features_v1/manifest.json',
                 ROOT/'runs/m2_mask_learning_20261101/image_seed20261117/best.pt',
                 ROOT/'runs/m2_mask_learning_20261101/image_seed20261117/config.json',
                 ROOT/'runs/m2_mask_learning_20261101/image_seed20261117/summary.json']
    for spec in specs:
        if spec['run']:
            run = ROOT/spec['run']
            if json.loads((run/'summary.json').read_text())['status'] != 'completed':
                raise RuntimeError(f'Incomplete predictor: {spec["name"]}')
            artifacts.extend(run/name for name in ['best.pt', 'config.json', 'summary.json'])
    artifacts.extend((ROOT/'runs/m2_mask_learning_20261101/mean_readout_controls').glob('combined35_*_coefficients.npz'))
    sources = source_closure([ROOT/'scripts'/name for name in [
        'freeze_official_scattering_assessment.py', 'prepare_official_scattering.py',
        'cache_official_scattering_features.py', 'predict_official_scattering.py',
        'score_official_scattering.py']])
    record = dict(status='frozen_before_official_acquisition', created=time.time(),
        official_test_downloaded=False, official_test_opened=False, models=specs,
        protocol=preparation, object_handling='ExactlyLMDBkeys0–99,float32(1,128,128),physicalcontrastf, noresize/clipping; responseusesf,notnormalized2f-1.',
        randomness='SeedSequence[20270101,noise_seed,objectID]; masksSeedSequence[20270101,902+order,objectID].',
        bootstrap_seed=20270131, bootstrap_draws=2000,
        code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        artifact_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(artifacts))},
        prerequisite_hashes={str(p.relative_to(ROOT)):sha256(p) for p in prereqs})
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(path=str(out), models=len(specs), code_files=len(sources), sha256=sha256(out))))


if __name__ == '__main__':
    main()
