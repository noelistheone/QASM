"""Freeze public development inputs and check batched algebra against upstream."""
from pathlib import Path
import argparse
import json
import sys
import time

import lmdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/InverseBench')]
from src.born_scattering import BornConfig, BatchedBorn, sha256
from inverse_problems.inverse_scatter import InverseScatter
from algo.daps import LangevinDynamics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT/'audit/gpu_execution_20261101'
    out.mkdir(parents=True, exist_ok=True)
    data_out = ROOT/'data/inv_scatter_dev'
    data_out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'sources/resources/validation_extracted/inv-scatter-val'
    env = lmdb.open(str(source), readonly=True, lock=False, create=False)
    with env.begin() as tx:
        images = np.stack([np.frombuffer(tx.get(str(i).encode()), dtype=np.float32).reshape(1,128,128)
                           for i in range(10)])
    env.close()
    np.save(data_out/'images.npy', images)
    (data_out/'manifest.json').write_text(json.dumps({
        'split': 'official validation (development only)', 'ids': list(range(10)),
        'truth_used_for': 'simulation and evaluation; never reconstruction or acquisition decisions',
        'normalization': 'x=2*f-1', 'shape': list(images.shape),
        'images_sha256': sha256(data_out/'images.npy'),
        'source_sha256': sha256(source/'data.mdb'), 'official_test_opened': False}, indent=2))
    checks = []

    def check(name, error, tolerance):
        row = dict(name=name, error=float(error), tolerance=tolerance, passed=bool(error <= tolerance))
        checks.append(row)
        print(json.dumps(row), flush=True)
        if not row['passed']:
            (out/'born_checks_FAILED.json').write_text(json.dumps(checks, indent=2))
            raise AssertionError(name)

    gen = torch.Generator(device=args.device).manual_seed(20261101)
    for n in [16, 128]:
        c = BornConfig(Nx=n, Ny=n, numRec=13, numTrans=3)
        op = BatchedBorn(c, device=args.device, dtype=torch.float64)
        ref = InverseScatter(Nx=n, Ny=n, numRec=13, numTrans=3, svd=False,
                             unnorm_shift=1., unnorm_scale=.5, device=args.device)
        x = torch.randn(3,1,n,n,device=args.device,dtype=torch.float64,generator=gen)
        pred = op.as_complex(op.forward(x))
        reference = torch.cat([ref.forward(x[i:i+1]) for i in range(3)])
        check(f'forward_double_N{n}', (pred-reference).norm()/reference.norm(), 2e-12)
        observed = op.forward(x) + .001*torch.randn(pred.numel()*2//3, device=args.device,
            dtype=torch.float64,generator=gen)[None, :]
        grad = op.gradient(x, observed)
        gradient_ref = torch.cat([ref.gradient(x[i:i+1],op.as_complex(observed[i:i+1]))
                                  for i in range(3)])
        check(f'gradient_double_N{n}', (grad-gradient_ref).norm()/gradient_ref.norm(), 2e-11)
        v = torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=gen)
        y = torch.randn(observed.shape,device=x.device,dtype=x.dtype,generator=gen)
        lhs = (op.forward(v,unnormalize=False)*.5*y).sum()
        rhs = (v*op.adjoint(y)).sum()
        check(f'adjoint_N{n}', abs(lhs-rhs)/(abs(lhs)+abs(rhs)), 1e-12)
        eps = 1e-4
        fd = (op.loss(x+eps*v,observed).sum()-op.loss(x-eps*v,observed).sum())/(2*eps)
        check(f'loss_directional_N{n}', abs(fd-(grad*v).sum())/(abs(fd)+1e-20), 1e-8)
        # A fresh off-truth batch must be evaluated independently, not broadcast from x[0].
        check(f'batch_independence_N{n}', (op.forward(x)[2]-op.forward(x[2:])[0]).abs().max(), 1e-12)

    timing = []
    for receivers in [60, 180, 360]:
        t0 = time.perf_counter()
        c = BornConfig(numRec=receivers)
        op = BatchedBorn(c,cache_dir=ROOT/'data/born_cache',device=args.device)
        ref = InverseScatter(numRec=receivers,numTrans=20,svd=False,
                             unnorm_shift=1.,unnorm_scale=.5,device=args.device)
        x = torch.as_tensor(2*images[:2]-1,device=args.device)
        y_ref = torch.cat([ref.forward(x[i:i+1]) for i in range(2)])
        y = op.as_complex(op.forward(x))
        check(f'full_float32_R{receivers}',(y-y_ref).norm()/y_ref.norm(),2e-6)
        check(f'float32_absolute_R{receivers}',(y-y_ref).abs().max(),1e-7)
        observation = op.forward(x) + 1e-4*torch.randn(2*y.numel(),device=args.device,generator=gen).reshape(2,-1)
        grad = op.gradient(x,observation)
        gref = torch.cat([ref.gradient(x[i:i+1],op.as_complex(observation[i:i+1])) for i in range(2)])
        check(f'full_gradient_float32_R{receivers}',(grad-gref).norm()/gref.norm(),2e-4)
        norm2 = op.spectral_norm_squared()
        lr = {360:4e-5,180:8e-5,60:2e-4}[receivers]
        timing.append(dict(receivers=receivers,cache_seconds=time.perf_counter()-t0,
                           matrix_norm_squared=norm2,
                           largest_data_hessian_step=lr*norm2*.25/1e-8))
        del op, ref
        torch.cuda.empty_cache()

    # Match one Langevin chain with the upstream optimizer for a fixed RNG stream.
    op = BatchedBorn(BornConfig(Nx=16,Ny=16,numRec=13,numTrans=3),device=args.device)
    x = torch.randn(2,1,16,16,device=args.device,generator=gen)
    observation = op.forward(x)
    torch.manual_seed(123)
    reference = LangevinDynamics(4,1e-5,tau=.01,lr_min_ratio=.5).sample(x,op,observation,.3,.2)
    torch.manual_seed(123)
    new = x.clone()
    lr = 1e-5*(1+.2*(.5-1))
    for _ in range(4):
        new = new - lr*(op.gradient(new,observation)/(2*.01**2)+(new-x)/.3**2)
        new = new + np.sqrt(2*lr)*torch.randn_like(new)
    check('Langevin_update_parity', (reference-new).abs().max(), 2e-7)
    report = dict(checks=checks,timing=timing,passed=all(c['passed'] for c in checks),
                  upstream_commit=BornConfig().source_commit,
                  float_precision='float32 main execution; independently checked complex128 reference',
                  scope='Batched linear inverse-scattering operator, not real ptychographic data')
    (out/'born_operator_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':len(checks),'cache_timing':timing}),flush=True)


if __name__ == '__main__':
    main()
