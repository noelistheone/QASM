"""CPU checks before the conditional M3 rank experiment; no model fitting."""
from pathlib import Path
import datetime
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import BatchedBorn, BornConfig, sha256
from src.controlled_query_learning import (coherent_response, current_response,
    draw_training_observation, fixed_object_noise, independent_noise_generator,
    matched_rank_initialization)
from src.low_rank_physical_update import project_queries, quadratic_readout
from src.response_learning import PhysicalQueries


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20261201)
    before = torch.get_rng_state().clone()
    models, records = {}, {}
    for rank in [1, 2, 4, 8]:
        with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA RNG must not be reseeded')):
            models[rank], records[rank] = matched_rank_initialization(rank, 20261126)
        assert torch.equal(before, torch.get_rng_state())
    assert len({v['backbone_sha256'] for v in records.values()}) == 1
    assert len({v['common_rank8_state_sha256'] for v in records.values()}) == 1
    for rank, model in models.items():
        for suffix in ['weight', 'bias']:
            key = f'trunk.head.{suffix}'
            assert torch.equal(model.state_dict()[key], models[8].state_dict()[key][:rank])
    context = SimpleNamespace(signals=np.zeros((7, 7200), np.float32))
    streams = [independent_noise_generator(20261126) for _ in models]
    first = [draw_training_observation(context, [1, 5], g) for g in streams]
    assert all(np.array_equal(v, first[0]) for v in first)
    rng_state = streams[0].bit_generator.state
    next_obs = draw_training_observation(context, [0, 3], streams[0])
    restored = independent_noise_generator(0)
    restored.bit_generator.state = rng_state
    assert np.array_equal(next_obs, draw_training_observation(context, [0, 3], restored))
    all_noise = fixed_object_noise([2, 8, 15], 7200, 20261211)
    assert np.array_equal(all_noise[[2, 0]], fixed_object_noise([15, 2], 7200, 20261211))
    assert np.array_equal(2 * all_noise, fixed_object_noise([2, 8, 15], 7200, 20261211, 2e-4))
    del models
    matrices = [BatchedBorn(BornConfig(wave=w, numRec=60),
                            cache_dir=ROOT / 'data/born_cache', device='cpu').matrix
                for w in [4, 6, 8]]
    physical = PhysicalQueries(matrices)
    mean = torch.rand(2, 1, 128, 128)
    factors = (torch.randn(2, 4, 128 * 128) * .01).requires_grad_()
    old = quadratic_readout(project_queries(mean, factors, physical))
    new = current_response(mean, factors, physical)
    rel = float(((old - new).norm() / old.norm()).detach())
    scale = old.detach().mean(0)
    g_old, = torch.autograd.grad((old / scale).square().mean(), factors)
    g_new, = torch.autograd.grad((new / scale).square().mean(), factors)
    grad_rel = float((g_old - g_new).norm() / g_old.norm())
    assert rel < 1e-12 and grad_rel < 2e-6
    one = factors.detach()[:, :1].clone().requires_grad_()
    projection = project_queries(mean, one, physical)
    expanded = (projection['constant'] + 2*projection['linear'][...,0]
                + projection['quadratic'][...,0,0])
    coherent = coherent_response(mean, one, physical)
    coherent_rel = float(((expanded-coherent).norm()/expanded.norm()).detach())
    g_expanded, = torch.autograd.grad((expanded/scale).square().mean(), one)
    g_coherent, = torch.autograd.grad((coherent/scale).square().mean(), one)
    coherent_grad_rel = float((g_expanded-g_coherent).norm()/g_expanded.norm())
    # Matrix products are float32; expanding (mu+delta) before that product changes rounding.
    assert coherent_rel < 2e-6 and coherent_grad_rel < 2e-5
    out = ROOT / 'audit/m3_preparation_20261101'
    out.mkdir(exist_ok=True)
    report = dict(status='passed', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Preparation only; no M3 training, evaluation-grid generation or model selection.',
        rank_initializations=records, readout_relative_error=rel,
        factor_gradient_relative_error=grad_rel,
        coherent_cross_term_relative_error=coherent_rel,
        coherent_factor_gradient_relative_error=coherent_grad_rel,
        checks=['common trunk', 'common head prefix', 'CPU initialization RNG unchanged',
                'initialization never reseeds a CUDA generator',
                'architecture-independent training-noise arrays', 'noise-stream exact resume',
                'object-noise batch/order invariance', 'noise-scale coupling',
                'actual original-query operators: readout equality', 'readout gradient equality',
                'coherent point control includes exact quadratic cross term', 'coherent gradient equality'],
        seconds=time.perf_counter()-started,
        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in
            [Path(__file__), ROOT/'src/controlled_query_learning.py']})
    (out/'controlled_input_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
