"""Nonnegative FISTA-TV on the fixed DAPS observations.

The published InverseBench tau=5e-7 is always reported. Neighboring values are
development controls, retained together. This is an independent implementation
of the stated convex objective, not a claim of bitwise MATLAB reproduction.
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import BornConfig,BatchedBorn,sha256
from src.tv_prox import tv_prox,total_variation,proximal_certificate
spec=importlib.util.spec_from_file_location('born_metrics',ROOT/'scripts/run_born_daps.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def main():
    p=argparse.ArgumentParser();p.add_argument('--daps-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--maxiter',type=int,default=2000);p.add_argument('--prox-iterations',type=int,default=50)
    p.add_argument('--precision',choices=['float32','float64'],default='float32')
    p.add_argument('--tau',type=float,nargs='+',default=[5e-8,1.5e-7,5e-7,1.5e-6,5e-6])
    p.add_argument('--resume',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((args.daps_run/'protocol.json').read_text())
    dtype=getattr(torch,args.precision)
    operator=BatchedBorn(BornConfig(**cfg['operator']),cache_dir=ROOT/'data/born_cache',device=args.device,dtype=dtype)
    observations=torch.tensor(np.load(args.daps_run/'measurements_real.npy'),device=args.device,dtype=dtype)
    truth=torch.tensor(np.load(ROOT/'data/inv_scatter_dev/images.npy')[cfg['objects']],device=args.device,dtype=dtype)
    n=len(observations);y=observations.repeat(len(args.tau),1)
    tau=torch.tensor(args.tau,device=args.device,dtype=dtype).repeat_interleave(n)
    norm2=operator.spectral_norm_squared(steps=100);step_size=.99/norm2
    x=torch.zeros(len(y),128,128,device=args.device,dtype=dtype);extrapolated=x.clone();dual=None;momentum=1.
    start=0;history=[];previous_seconds=0.;checkpoint=args.out/'checkpoint.pt'
    protocol=dict(operator=cfg['operator'],objects=cfg['objects'],tau=args.tau,
        objective='0.5*sum_all_complex_measurements |Af-y|^2 + tau*sum_pixel isotropic_TV(f); f>=0',
        data_normalization='no averaging by receiver, transmitter or pixel',
        boundary='replicated edge / zero last forward difference',initialization='zero object',
        published_tau=5e-7,selection='all development candidates retained; no official test accessed',
        maxiter=args.maxiter,prox_iterations=args.prox_iterations,norm_squared=norm2,precision=args.precision,
        initial_step=step_size,backtracking='smooth majorization; step halves on failure',
        measurement_sha256=sha256(args.daps_run/'measurements_real.npy'),
        sources={str(f.relative_to(ROOT)):sha256(f) for f in [Path(__file__),ROOT/'src/tv_prox.py',ROOT/'src/born_scattering.py']},
        reference='InverseBench Appendix B.7.1; Sun et al. 2019, doi:10.1109/TCI.2019.2893568',
        reference_code_commit='88468d889c0e4e8f93fbc6f244e8ff2479a618d6')
    if args.resume:
        old=json.loads((args.out/'protocol.json').read_text())
        if old!=protocol:raise ValueError('Resume protocol mismatch')
        state=torch.load(checkpoint,map_location=args.device,weights_only=True)
        x,extrapolated,dual=state['x'],state['extrapolated'],state['dual']
        momentum,step_size,start=state['momentum'],state['step_size'],state['iteration']
        history=state['history'];previous_seconds=state['seconds']
    else:
        if checkpoint.exists() or (args.out/'summary.json').exists():raise FileExistsError(args.out)
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    t0=time.perf_counter();backtracks=0
    def forward(z):return operator.forward(z[:,None],unnormalize=False)
    def elapsed():return previous_seconds+time.perf_counter()-t0
    with torch.no_grad():
        for iteration in range(start,args.maxiter):
            residual=forward(extrapolated)-y
            grad=operator.adjoint(residual,normalized=False)[:,0]
            value=.5*residual.square().sum(1)
            for trial in range(20):
                next_x,next_dual=tv_prox(extrapolated-step_size*grad,step_size*tau,dual,
                                         iterations=args.prox_iterations)
                displacement=next_x-extrapolated
                next_residual=forward(next_x)-y
                next_value=.5*next_residual.square().sum(1)
                upper=value+(grad*displacement).flatten(1).sum(1)+displacement.square().flatten(1).sum(1)/(2*step_size)
                tolerance=2e-6*(next_value.abs()+value.abs()).clamp_min(1e-12)
                if bool((next_value<=upper+tolerance).all()):break
                step_size*=.5;backtracks+=1
            else:raise RuntimeError('Smooth-majorization backtracking failed')
            if not bool(torch.isfinite(next_x).all()):raise FloatingPointError('nonfinite TV iterate')
            delta=(next_x-x).flatten(1).norm(dim=1)/(1+x.flatten(1).norm(dim=1))
            next_momentum=(1+math.sqrt(1+4*momentum**2))/2
            extrapolated=next_x+((momentum-1)/next_momentum)*(next_x-x)
            x,dual,momentum=next_x,next_dual,next_momentum
            if (iteration+1)%50==0 or iteration+1==args.maxiter:
                row=dict(iteration=iteration+1,seconds=elapsed(),step_size=step_size,
                    max_relative_iterate_change=float(delta.max()),
                    mean_objective=float((next_value+tau*total_variation(x)).mean()),backtracks=backtracks)
                history.append(row);print(json.dumps(row),flush=True)
                (args.out/'status.json').write_text(json.dumps(dict(status='running',**row),indent=2))
                temp=checkpoint.with_suffix('.partial.pt')
                torch.save(dict(x=x,extrapolated=extrapolated,dual=dual,momentum=momentum,step_size=step_size,
                    iteration=iteration+1,history=history,seconds=elapsed()),temp);temp.replace(checkpoint)
        grad=operator.adjoint(forward(x)-y,normalized=False)[:,0]
        prox_input=x-step_size*grad
        prox,certificate_dual=tv_prox(prox_input,step_size*tau,dual,iterations=500)
        certificate=proximal_certificate(prox_input,prox,step_size*tau,certificate_dual)
        fixed_residual=(x-prox).flatten(1).norm(dim=1)/(1+x.flatten(1).norm(dim=1))
        scored=mod.metrics(x[:,None],truth.repeat(len(args.tau),1,1,1));records=[]
        objectives=.5*(forward(x)-y).square().sum(1)+tau*total_variation(x)
        for a,v in enumerate(args.tau):
            for i,obj in enumerate(cfg['objects']):
                j=a*n+i
                records.append(dict(tau=v,object_id=obj,psnr=float(scored['psnr'][j]),ssim=float(scored['ssim'][j]),
                    raw_rmse=float(scored['raw_rmse'][j]),objective=float(objectives[j]),
                    proximal_fixed_point_relative_residual=float(fixed_residual[j]),
                    prox_duality_gap=float(certificate['gap'][j])))
        with (args.out/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        np.save(args.out/'predictions.npy',x.cpu().numpy().reshape(len(args.tau),n,1,128,128))
        grouped=[dict(tau=v,mean_psnr=float(scored['psnr'].reshape(len(args.tau),n)[j].mean()),
                    mean_ssim=float(scored['ssim'].reshape(len(args.tau),n)[j].mean()),
                    max_fixed_point_residual=float(fixed_residual.reshape(len(args.tau),n)[j].max()))
                    for j,v in enumerate(args.tau)]
        summary=dict(status='completed',seconds=elapsed(),device=args.device,n_objects=n,
            grouped=grouped,history=history,measurement_sha256=protocol['measurement_sha256'],
            matrix_calls=operator.calls,max_fixed_point_residual=float(fixed_residual.max()),
            max_prox_gap=float(certificate['gap'].max()),published_tau=5e-7,
            selection='grid evaluated on development only; final test untouched',backtracks=backtracks)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
        (args.out/'status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
