"""Shared, observation-only inputs for frozen-mean covariance experiments."""
from pathlib import Path
import json
import numpy as np
import torch
from src.born_scattering import BornConfig, BatchedBorn, sha256
from src.response_learning import PhysicalQueries
from src.mask_conditioned_learning import MaskUNet, masked_cg_feature, evaluation_masks


class MaskLearningContext:
    def __init__(self, root, mean_run, device, profile=False):
        self.root = Path(root)
        self.device = torch.device(device)
        self.mean_run = Path(mean_run)
        if not profile and not (self.mean_run / 'summary.json').exists():
            raise RuntimeError('The declared mean-predictor run must finish before covariance fitting')
        if (self.mean_run / 'summary.json').exists():
            assert json.loads((self.mean_run / 'summary.json').read_text())['status'] == 'completed'
        config = json.loads((self.mean_run / 'config.json').read_text())
        assert config['mode'] == 'image' and config['seed'] == 20261117
        self.mean_checkpoint_hash = sha256(self.mean_run / 'best.pt')
        checkpoint = torch.load(self.mean_run / 'best.pt', map_location=self.device, weights_only=True)
        self.mean_step = checkpoint['step']
        self.mean = MaskUNet().to(self.device)
        self.mean.load_state_dict(checkpoint['model'])
        self.mean.eval().requires_grad_(False)
        # A profile may use a running checkpoint; formal experiments may not.
        self.mean_is_completed = (self.mean_run / 'summary.json').exists()
        data = self.root / 'data/m2_mask_features_v1'
        parent = self.root / 'data/m2_response_pilot_v1'
        self.meta = json.loads((data / 'manifest.json').read_text())
        assert self.meta['status'] == 'completed'
        self.split = json.loads((data / 'split.json').read_text())
        groups = [set(self.split[k]) for k in ['mean_train', 'covariance_train', 'validation', 'assessment']]
        for i in range(len(groups)):
            for j in range(i):
                assert not groups[i].intersection(groups[j])
        self.images = np.load(parent / 'images.npy', mmap_mode='r')
        self.targets = np.load(parent / 'queries.npy', mmap_mode='r')
        self.signals = np.load(data / 'clean_signals.npy', mmap_mode='r')
        self.observations = np.load(self.root / 'data/m2_gaussian_actions_v1/measurements.npy', mmap_mode='r')
        cache = self.root / 'data/born_cache'
        self.matrix = BatchedBorn(BornConfig(numRec=180), cache_dir=cache, device=self.device).matrix
        self.regularization = self.meta['ridge_info']['lambda']
        matrices = [BatchedBorn(BornConfig(wave=w, numRec=60), cache_dir=cache,
                               device=self.device).matrix for w in [4, 6, 8]]
        self.physical = PhysicalQueries(matrices).to(self.device)
        self.query_scale = torch.tensor(self.meta['query_scale'], device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def batch(self, indices, masks, training=False):
        source = self.signals if training else self.observations
        observation = torch.tensor(np.asarray(source[indices].copy()), device=self.device)
        if training:
            observation += torch.randn_like(observation) * (1e-4 / np.sqrt(2))
        mask = torch.tensor(masks, device=self.device)
        feature, residual = masked_cg_feature(observation, mask, self.matrix, self.regularization)
        if not bool(torch.isfinite(feature).all()):
            raise FloatingPointError('Nonfinite input feature')
        mean = self.mean(feature, mask)
        truth = torch.tensor(np.asarray(self.images[indices].copy()), device=self.device)
        target = torch.tensor(np.asarray(self.targets[indices].copy()), device=self.device)
        return dict(feature=feature, mask=mask, mean=mean, truth=truth, target=target,
                    observation=observation, cg_residual=residual)

    @torch.no_grad()
    def training_energy_scale(self, batch_size=128):
        ids = self.split['covariance_train'][:512]
        total = 0.
        count = 0
        by_budget = []
        for budget in [0, 2, 5, 10]:
            subtotal = 0.
            for start in range(0, len(ids), batch_size):
                subset = ids[start:start + batch_size]
                b = self.batch(subset, evaluation_masks(subset, budget))
                energy = (b['truth'] - b['mean']).square().flatten(1).sum(1)
                subtotal += float(energy.sum())
                count += len(subset)
            total += subtotal
            by_budget.append(dict(additional_actions=budget, mean_energy=subtotal / len(ids)))
        return dict(fixed_energy_scale=total / count, objects=len(ids), evaluations=count,
                    by_budget=by_budget, mean_checkpoint_sha256=self.mean_checkpoint_hash)

    def factor_queries(self, factors):
        # PhysicalQueries is homogeneous quadratic: sum_k q(L_k)=tr(Q LL^T).
        b, k, d = factors.shape
        if d != 128 * 128:
            raise ValueError('Image-grid mismatch')
        return self.physical(factors.reshape(b * k, 1, 128, 128)).reshape(b, k, 15).sum(1)
