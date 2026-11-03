"""Reproducible inputs and current-query readout for the planned rank study.

This module does not alter the frozen M2 implementations. Observation noise has
its own NumPy generator, independent of architecture initialization and device.
The readout is the diagonal contraction already used in M2, with no Gaussian
conditioning computation when the loss contains only current responses.
"""
import copy
import hashlib

import numpy as np
import torch

from src.mask_conditioned_learning import masked_cg_feature
from src.posterior_components import MaskPosteriorComponents


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def matched_rank_initialization(rank, seed, max_rank=8):
    """Identical trunk and prefix of one common output head for every rank."""
    if rank not in [1, 2, 4, 8] or rank > max_rank:
        raise ValueError('Rank outside the declared ablation')
    # CPU initialization is independent of the training-device RNG state.
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA generators. Only the CPU generator
        # belongs to this fork; leave all device streams untouched.
        torch.random.default_generator.manual_seed(seed)
        common = MaskPosteriorComponents(max_rank)
        common_state = copy.deepcopy(common.state_dict())
        common_hash = state_digest(common_state)
        if rank == max_rank:
            model = common
        else:
            model = MaskPosteriorComponents(rank)
            for name in ['trunk.head.weight', 'trunk.head.bias']:
                common_state[name] = common_state[name][:rank].clone()
            model.load_state_dict(common_state)
    backbone = {k: v for k, v in model.state_dict().items()
                if not k.startswith('trunk.head.')}
    return model, dict(common_rank8_state_sha256=common_hash,
                       backbone_sha256=state_digest(backbone),
                       initial_state_sha256=state_digest(model.state_dict()))


def independent_noise_generator(seed):
    return np.random.default_rng(np.random.SeedSequence([int(seed), 20261201]))


def draw_training_observation(context, ids, noise_rng):
    clean = np.asarray(context.signals[ids]).copy()
    noise = noise_rng.standard_normal(clean.shape, dtype=np.float32)
    return clean + noise * np.float32(1e-4 / np.sqrt(2))


def fixed_object_noise(ids, channels, seed, complex_std=1e-4):
    """Object identity fixes noise across masks, batches and model executions."""
    values = []
    for obj in ids:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(obj)]))
        values.append(rng.standard_normal(channels, dtype=np.float32))
    return np.stack(values) * np.float32(complex_std / np.sqrt(2))


@torch.no_grad()
def batch_from_observation(context, ids, masks, observation=None):
    """The caller declares observed data; truth is returned only as a loss label."""
    if observation is None:
        observation = np.asarray(context.observations[ids]).copy()
    if observation.shape != (len(ids), context.matrix.shape[0]):
        raise ValueError('Unexpected observation shape')
    obs = torch.tensor(observation, device=context.device)
    mask = torch.tensor(masks, device=context.device)
    feature, residual = masked_cg_feature(obs, mask, context.matrix,
                                          context.regularization)
    if not bool(torch.isfinite(feature).all()):
        raise FloatingPointError('Nonfinite feature')
    mean = context.mean(feature, mask)
    truth = torch.tensor(np.asarray(context.images[ids]).copy(), device=context.device)
    target = torch.tensor(np.asarray(context.targets[ids]).copy(), device=context.device)
    return dict(feature=feature, mask=mask, mean=mean, truth=truth,
                target=target, observation=obs, cg_residual=residual)


def current_response(mean, factors, physical):
    """q(mean) + sum_k q(L_k), matching M2's float64 diagonal contraction."""
    batch, rank, dimension = factors.shape
    if mean.flatten(1).shape != (batch, dimension):
        raise ValueError('Mean/factor dimensions differ')
    weights = physical.weights.repeat_interleave(2, dim=0).to(torch.float64)
    responses = []
    for matrix in physical.matrices:
        mean_field = (mean.flatten(1) @ matrix.T).to(torch.float64)
        fields = (factors.reshape(batch * rank, dimension) @ matrix.T)
        fields = fields.reshape(batch, rank, -1).to(torch.float64)
        responses.append(mean_field.square() @ weights
                         + fields.square().sum(1) @ weights)
    return torch.cat(responses, 1)


def coherent_response(mean, factors, physical):
    """Matched one-field point reconstruction: q(mean + delta), including cross terms."""
    if factors.shape[1] != 1 or mean.flatten(1).shape != factors[:, 0].shape:
        raise ValueError('The point-reconstruction control has exactly one spatial field')
    image = mean.flatten(1) + factors[:, 0]
    weights = physical.weights.repeat_interleave(2, dim=0).to(torch.float64)
    return torch.cat([(image @ matrix.T).to(torch.float64).square() @ weights
                      for matrix in physical.matrices], 1)
