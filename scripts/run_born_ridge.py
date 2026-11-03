"""Classical Tikhonov control on exactly the DAPS development measurements.

All regularization values are retained. Validation-dependent best values are
explicitly development selections and must be frozen before a test evaluation.
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import BornConfig,BatchedBorn,sha256
spec=importlib.util.spec_from_file_location('born_metrics',ROOT/'scripts/run_born_daps.py')
metrics_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics_module)


def main():
    p=argparse.ArgumentParser();p.add_argument('--daps-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--maxiter',type=int,default=500)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=8)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    cfg=json.loads((args.daps_run/'protocol.json').read_text())
    c=BornConfig(**cfg['operator'])
    operator=BatchedBorn(c,cache_dir=ROOT/'data/born_cache',device=args.device)
    y=torch.tensor(np.load(args.daps_run/'measurements_real.npy'),device=args.device)
    truth=torch.tensor(np.load(ROOT/'data/inv_scatter_dev/images.npy')[cfg['objects']],device=args.device)
    alphas=np.logspace(-6,0,7)
    norm2=operator.spectral_norm_squared()
    reg=torch.tensor(alphas*norm2,device=args.device,dtype=torch.float32).repeat_interleave(len(y))
    expanded_y=y.repeat(len(alphas),1)
    target=truth.repeat(len(alphas),1,1,1)
    rhs=operator.adjoint(expanded_y,normalized=False).flatten(1)
    diagonal=operator.matrix.square().sum(0)[None,:]+reg[:,None]
    solution=torch.zeros_like(rhs);residual=rhs.clone();z=residual/diagonal;direction=z.clone()
    rz=(residual*z).sum(1)
    denom_ref=rhs.norm(dim=1).clamp_min(1e-30)
    active=torch.ones(len(rhs),device=args.device,dtype=torch.bool)
    converged_at=torch.full((len(rhs),),-1,device=args.device,dtype=torch.int32)
    t0=time.perf_counter()
    history=[]
    with torch.no_grad():
        for step in range(args.maxiter):
            image=direction.reshape(-1,1,128,128)
            product=operator.adjoint(operator.forward(image,unnormalize=False),normalized=False).flatten(1)
            product+=reg[:,None]*direction
            denom=(direction*product).sum(1)
            coefficient=torch.where(active,rz/denom.clamp_min(1e-30),0)
            solution+=coefficient[:,None]*direction
            residual-=coefficient[:,None]*product
            relative=residual.norm(dim=1)/denom_ref
            newly=active & (relative<1e-5)
            converged_at[newly]=step+1
            active &= ~newly
            newz=residual/diagonal
            rz_new=(residual*newz).sum(1)
            beta=torch.where(active,rz_new/rz.clamp_min(1e-30),0)
            direction=newz+beta[:,None]*direction
            direction[~active]=0
            rz=rz_new
            if (step+1)%25 == 0 or not bool(active.any()):
                row={'step':step+1,'active':int(active.sum()),'max_relative_residual':float(relative.max()),
                     'seconds':time.perf_counter()-t0}
                history.append(row);print(json.dumps(row),flush=True)
                (args.out/'status.json').write_text(json.dumps(dict(status='running',**row),indent=2))
            if not bool(active.any()):break
        predicted=solution.reshape(-1,1,128,128)
        scored=metrics_module.metrics(predicted,target)
        # Independently recompute the final residual; the recurrence can drift.
        recomputed=operator.adjoint(operator.forward(predicted,unnormalize=False),normalized=False).flatten(1)
        recomputed+=reg[:,None]*solution
        final_relative=(recomputed-rhs).norm(dim=1)/denom_ref
        records=[]
        for a,alpha in enumerate(alphas):
            for i,obj in enumerate(cfg['objects']):
                j=a*len(y)+i
                records.append(dict(relative_alpha=float(alpha),absolute_lambda=float(reg[j]),object_id=obj,
                    psnr=float(scored['psnr'][j]),ssim=float(scored['ssim'][j]),raw_rmse=float(scored['raw_rmse'][j]),
                    cg_steps=int(converged_at[j]),cg_recomputed_relative_residual=float(final_relative[j])))
        with (args.out/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        np.save(args.out/'predictions.npy',predicted.cpu().numpy().reshape(len(alphas),len(y),1,128,128))
        grouped=[dict(relative_alpha=float(alpha),mean_psnr=float(scored['psnr'].reshape(len(alphas),len(y))[j].mean()),
                       mean_ssim=float(scored['ssim'].reshape(len(alphas),len(y))[j].mean())) for j,alpha in enumerate(alphas)]
        summary={'status':'completed','measurement_sha256':sha256(args.daps_run/'measurements_real.npy'),
            'objective':'min ||A f-y||^2 + lambda ||f||^2','n_objects':len(y),
            'regularization_grid':alphas.tolist(),'matrix_norm_squared':norm2,'grouped':grouped,
            'validation_best_psnr_alpha':max(grouped,key=lambda x:x['mean_psnr'])['relative_alpha'],
            'selection_note':'Development selection only; all seven configurations retained',
            'cg_converged':int((converged_at>0).sum()),'cg_total':len(rhs),
            'cg_max_recomputed_relative_residual':float(final_relative.max()),
            'seconds':time.perf_counter()-t0,'device':args.device,'history':history}
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
        (args.out/'status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
