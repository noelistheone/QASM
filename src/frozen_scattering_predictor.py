"""One observation-only inference interface for existing frozen query models.

No fitting, selection or scoring occurs here. The wrapper reuses each model's
unchanged prediction formula and records artifact hashes for later blind runs.
"""
from pathlib import Path
import json

import numpy as np
import torch

from src.born_scattering import sha256
from src.cached_query_readouts import direct_from_cache, ridge_from_cache
from src.controlled_query_learning import current_response, coherent_response
from src.direct_query_cnn import DirectQueryCNN
from src.direct_query_cnn_additive import AdditiveDirectQueryCNN
from src.mask_conditioned_learning import MaskUNet
from src.posterior_components import MaskPosteriorComponents


class CombinedQueryOperators:
    def __init__(self, context):
        self.original, self.transfer = context.original, context.transfer
        self.matrices = list(context.original.matrices)+list(context.transfer.matrices)
        self.weights = context.original.weights

    def __call__(self, image):
        return torch.cat([self.original(image), self.transfer(image)], 1)


class FrozenScatteringPredictor:
    def __init__(self, context, kind, run=None, rank=8):
        allowed = {'moment', 'coherent', 'joint', 'multiplicative', 'additive',
                   'image_only', 'ridge_budget', 'ridge_pooled'}
        if kind not in allowed:
            raise ValueError('Unknown frozen predictor family')
        if kind == 'coherent' and rank != 1:
            raise ValueError('The coherent control has one correction field')
        self.context, self.kind = context, kind
        self.physical = CombinedQueryOperators(context)
        self.network = None
        self.coefficients = {}
        self.metadata = dict(kind=kind, rank=rank if kind in ['moment', 'coherent'] else None,
            frozen_mean_sha256=context.mean_checkpoint_hash, prediction_queries=35,
            accepts_reference=False, fits_parameters=False)
        if kind in ['image_only', 'ridge_budget', 'ridge_pooled']:
            if run is not None:
                raise ValueError('Calibration/mean models do not take a neural run')
            if kind != 'image_only':
                suffixes = ['pooled'] if kind == 'ridge_pooled' else [f'budget{b}' for b in [0, 2, 5, 10]]
                base = context.root/'runs/m2_mask_learning_20261101/mean_readout_controls'
                hashes = {}
                for suffix in suffixes:
                    path = base/f'combined35_{suffix}_coefficients.npz'
                    with np.load(path, allow_pickle=False) as data:
                        self.coefficients[suffix] = {key: data[key].copy() for key in data.files}
                    hashes[str(path.relative_to(context.root))] = sha256(path)
                self.metadata['coefficient_hashes'] = hashes
            return
        if run is None:
            raise ValueError('A completed neural run is required')
        path = context.root/Path(run)
        summary = json.loads((path/'summary.json').read_text())
        config = json.loads((path/'config.json').read_text())
        if summary.get('status') != 'completed':
            raise RuntimeError('The declared predictor is not complete')
        if kind != 'joint' and config['mean_checkpoint_sha256'] != context.mean_checkpoint_hash:
            raise RuntimeError('Predictor and mean artifacts disagree')
        if kind in ['moment', 'coherent']:
            net = MaskPosteriorComponents(rank)
        elif kind == 'joint':
            net = MaskUNet()
        elif kind == 'multiplicative':
            net = DirectQueryCNN(np.zeros(66), np.ones(66), 0., 1.)
        else:
            net = AdditiveDirectQueryCNN(np.zeros(66), np.ones(66), 0., 1., context.query_scale)
        checkpoint = path/'best.pt'
        checkpoint_hash = sha256(checkpoint)
        if 'checkpoint_sha256' in summary and summary['checkpoint_sha256'] != checkpoint_hash:
            raise RuntimeError('Checkpoint does not match its completed summary')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net.load_state_dict(state['model'], strict=True)
        self.network = net.to(context.device).eval().requires_grad_(False)
        self.metadata.update(run=str(path.relative_to(context.root)), checkpoint_sha256=checkpoint_hash,
            run_config_sha256=sha256(path/'config.json'), best_step=state.get('step'),
            parameters=sum(v.numel() for v in net.parameters()))

    @torch.no_grad()
    def __call__(self, features, budget):
        if any(name in features for name in ['truth', 'target', 'reference']):
            raise ValueError('Scoring targets are not accepted by inference')
        if budget not in [0, 2, 5, 10]:
            raise ValueError('Budget outside the declared assessment')
        f, mean, mask = [features[k] for k in ['feature', 'mean', 'masks']]
        if not bool((mask.sum(1) == 5+budget).all()):
            raise ValueError('Mask cardinality does not match the budget')
        if self.kind in ['moment', 'coherent']:
            factors = self.network(f, mean, mask)[2]
            readout = current_response if self.kind == 'moment' else coherent_response
            result = readout(mean, factors, self.physical)
        elif self.kind == 'joint':
            result = self.physical(self.network(f, mask))
        elif self.kind in ['multiplicative', 'additive']:
            result = direct_from_cache(self.network, f, mean, mask,
                features['features66'], features['mean_queries35'])
        elif self.kind == 'image_only':
            result = features['mean_queries35']
        else:
            suffix = 'pooled' if self.kind == 'ridge_pooled' else f'budget{budget}'
            value = ridge_from_cache(self.coefficients[suffix], features['features66'].cpu().numpy(),
                                     features['mean_queries35'].cpu().numpy())
            result = torch.tensor(value, device=self.context.device, dtype=torch.float64)
        if result.shape != (len(f), 35) or not bool(torch.isfinite(result).all()):
            raise FloatingPointError('Invalid query prediction; no clipping or dropping is performed')
        return result
