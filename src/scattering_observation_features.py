"""Frozen inference features without any image/target dataset handle.

This adapter prepares a later independent assessment. It preserves the existing
CG30, MaskUNet and query summaries, and is not substituted into running M3 jobs.
"""
from pathlib import Path
import json

import numpy as np
import torch

from src.born_scattering import BatchedBorn, BornConfig, sha256
from src.mask_conditioned_learning import MaskUNet, masked_cg_feature
from src.response_learning import PhysicalQueries


class FrozenScatteringFeatures:
    def __init__(self, root, device='cpu'):
        self.root = Path(root)
        self.device = torch.device(device)
        run = self.root/'runs/m2_mask_learning_20261101/image_seed20261117'
        if json.loads((run/'summary.json').read_text())['status'] != 'completed':
            raise RuntimeError('The frozen mean checkpoint is incomplete')
        config = json.loads((run/'config.json').read_text())
        if config['mode'] != 'image' or config['seed'] != 20261117:
            raise RuntimeError('Unexpected mean checkpoint')
        self.mean_checkpoint_hash = sha256(run/'best.pt')
        cp = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        self.mean = MaskUNet().to(self.device)
        self.mean.load_state_dict(cp['model'])
        self.mean.eval().requires_grad_(False)
        meta_path = self.root/'data/m2_mask_features_v1/manifest.json'
        meta = json.loads(meta_path.read_text())
        if meta['status'] != 'completed':
            raise RuntimeError('Training preprocessing manifest is incomplete')
        self.regularization = meta['ridge_info']['lambda']
        self.query_scale = np.asarray(meta['query_scale'], dtype=np.float32)
        cache = self.root/'data/born_cache'
        self.matrix = BatchedBorn(BornConfig(numRec=180), cache_dir=cache, device=self.device).matrix
        matrices = [BatchedBorn(BornConfig(wave=w, numRec=60), cache_dir=cache,
                               device=self.device).matrix for w in [4, 6, 8, 5, 7, 10, 12]]
        self.original = PhysicalQueries(matrices[:3]).to(self.device)
        self.transfer = PhysicalQueries(matrices[3:]).to(self.device)
        angles = np.abs((np.linspace(0, 359, 20)[:, None]-np.linspace(0, 359, 180)[None, :]+180) % 360-180)
        self.bands = torch.tensor(np.stack([
            (angles >= lo) & ((angles < hi) if hi < 180 else (angles <= hi))
            for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 90), (90, 180)]]),
            device=self.device, dtype=torch.float32)
        self.metadata = dict(mean_checkpoint_sha256=self.mean_checkpoint_hash,
            training_preprocessing_manifest_sha256=sha256(meta_path),
            regularization=self.regularization, CG_iterations=30,
            observed_power_subtraction=1e-8,
            query_waves=[4, 6, 8, 5, 7, 10, 12],
            dataset_handles=[], accepts_truth=False)

    @torch.no_grad()
    def __call__(self, observation, masks):
        obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        mask = torch.as_tensor(masks, device=self.device, dtype=torch.float32)
        if obs.ndim != 2 or obs.shape[1] != 7200 or mask.shape != (len(obs), 30) or len(obs) == 0:
            raise ValueError('Expected field channels (batch,7200) and masks (batch,30)')
        if not bool(torch.isfinite(mask).all() and ((mask == 0) | (mask == 1)).all() and (mask.sum(1) > 0).all()):
            raise ValueError('Expected nonempty binary receiver masks')
        channels = mask.repeat_interleave(6, 1)[:, None, :, None].expand(-1, 20, -1, 2).reshape(-1, 7200).bool()
        if not bool(torch.isfinite(obs[channels]).all()):
            raise ValueError('An observed field channel is nonfinite')
        # Missing measurements can be absent/NaN; remove them before every
        # operation, including the power summary, not just the CG right side.
        obs = torch.where(channels, obs, 0.)
        feature, residual = masked_cg_feature(obs, mask, self.matrix, self.regularization)
        mean = self.mean(feature, mask)
        if not bool(torch.isfinite(feature).all() and torch.isfinite(mean).all()):
            raise FloatingPointError('Nonfinite frozen reconstruction feature')
        mean_original = self.original(mean)
        mean_queries = torch.cat([mean_original, self.transfer(mean)], 1)
        feature_original = self.original(feature)
        acquired = mask.repeat_interleave(6, 1)[:, None, :]
        weights = self.bands[None]*acquired[:, None]
        denominators = weights.sum((-1, -2))
        if not bool((denominators > 0).all()):
            raise ValueError('A requested measured-power band is entirely absent')
        power = obs.reshape(-1, 20, 180, 2).square().sum(-1)-1e-8
        measured = (power[:, None]*weights).sum((-1, -2))/denominators
        summaries = torch.cat([mean_original, feature_original, measured, mask, mask.mean(1, keepdim=True)], 1)
        return dict(feature=feature, mean=mean, masks=mask, features66=summaries,
                    mean_queries35=mean_queries, cg_residual=residual,
                    visible_observation=obs)
