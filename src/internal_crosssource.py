"""Input-only inference helpers for the internal-update benchmark."""
from pathlib import Path
import json
import time
import numpy as np
import torch

from src.born_scattering import sha256
from src.query_aware_refinement import OrthogonalUpdateNet, query_aware_refine
from src.query_response_trust import QueryResponseBound
from src.learned_ptycho_refinement import SpatialPreconditioner, refine_columns


def load_freeze(root):
    root = Path(root)
    path = root/'protocols/m4_internal_crosssource_frozen_v1.json'
    data = json.loads(path.read_text())
    assert data['status'] == 'frozen_before_new_predictions'
    for group in ['code_hashes', 'artifact_hashes']:
        for name, digest in data[group].items():
            if sha256(root/name) != digest:
                raise RuntimeError(f'Changed assessment dependency: {name}')
    return data, sha256(path)


def load_model(root, specification, device):
    family = specification['family']
    if family.startswith('uniform_'):
        return None
    if family.startswith('complex_'):
        model = OrthogonalUpdateNet().to(device)
    elif family in ['internal_pointwise', 'internal_spatial']:
        model = SpatialPreconditioner(family.replace('internal_', '')).to(device)
    else:
        raise ValueError(f'Unknown method: {family}')
    path = Path(root)/specification['checkpoint']
    assert sha256(path) == specification['checkpoint_sha256']
    saved = torch.load(path, map_location=device, weights_only=False)
    assert saved['step'] == specification['selected_step'] and saved['seed'] == specification['seed']
    expected_kind = family.replace('internal_', '')
    assert saved['kind'] == expected_kind
    model.load_state_dict(saved['model'], strict=True); model.eval()
    return model


@torch.no_grad()
def infer_columns(case, problem, specification, model):
    if case.role is not None or case._raw is not None:
        raise PermissionError('An input-only case is required')
    family = specification['family']
    if family in ['internal_pointwise', 'internal_spatial']:
        return refine_columns(case.object[None], problem, case.fraction, steps=3,
                              model=model, differentiable=False)
    bounded = family in ['complex_bounded', 'uniform_bounded3']
    bound = QueryResponseBound(case.operator, case.query_ids, batch=256) if bounded else None
    return query_aware_refine(case.object[None], problem, case.fraction, model=model,
        bound=bound, radius=.1 if bounded else None, steps=3, differentiable=False)


@torch.no_grad()
def export_predictions(case, columns, path, batch=256):
    """Export raw archived-unit predictions; never read a measurement value."""
    path = Path(path)
    if case.role is not None or case._raw is not None:
        raise PermissionError('Query labels must be unavailable')
    if path.exists():
        raise FileExistsError(path)
    if not np.isfinite(case.alpha) or case.alpha <= 0:
        raise ValueError('A positive input-only normalization is required')
    ids_all = np.asarray(case.query_ids)
    if ids_all.ndim != 1 or len(ids_all) == 0 or len(np.unique(ids_all)) != len(ids_all):
        raise ValueError('Unique nonempty query coordinates required')
    result = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                      shape=(len(ids_all), 64, 64))
    start = time.perf_counter()
    for first in range(0, len(ids_all), batch):
        ids = torch.as_tensor(ids_all[first:first+batch], device=case.device)
        prediction = case.operator.fields(columns, ids).abs().square().sum(1)/case.alpha
        if prediction.shape != (len(ids), 64, 64) or not torch.isfinite(prediction).all() or (prediction < 0).any():
            raise FloatingPointError('Invalid physical prediction')
        result[first:first+len(ids)] = prediction.cpu().numpy().astype(np.float32)
    if torch.device(case.device).type == 'cuda': torch.cuda.synchronize(case.device)
    result.flush(); del result
    return dict(sha256=sha256(path), frames=len(ids_all), export_seconds=time.perf_counter()-start)
