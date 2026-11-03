"""Refine flagged Tikhonov systems with double precision; preserve originals."""
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
from src.born_scattering import BornConfig,BatchedBorn
spec=importlib.util.spec_from_file_location('born_metrics',ROOT/'scripts/run_born_daps.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def main():
    p=argparse.ArgumentParser();p.add_argument('--receivers',type=int,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--maxiter',type=int,default=5000)
    p.add_argument('--out-tag',default='numerical_refinement_gpu');args=p.parse_args()
    torch.set_num_threads(4)
    base=ROOT/'runs/m1_baselines_20261101';out=base/f'ridge_r{args.receivers}'
    source=list(csv.DictReader((out/'metrics.csv').open()))
    ids=[j for j,r in enumerate(source) if int(r['cg_steps'])<0 or float(r['cg_recomputed_relative_residual'])>1e-5]
    if not ids:
        print('No flagged systems');return
    config=json.loads((base/f'daps_r{args.receivers}_s32/protocol.json').read_text())
    op=BatchedBorn(BornConfig(**config['operator']),cache_dir=ROOT/'data/born_cache',device=args.device,dtype=torch.float64)
    original=np.load(out/'predictions.npy').reshape(-1,1,128,128)
    indices=np.array([j%10 for j in ids])
    observations=np.load(base/f'daps_r{args.receivers}_s32/measurements_real.npy')[indices]
    warm=out/'numerical_refinement.npz'
    if warm.exists():
        previous=np.load(warm)
        original=original.astype(np.float64)
        original[previous['flat_indices']]=previous['predictions']
    y=torch.tensor(observations,dtype=torch.float64,device=args.device)
    x=torch.tensor(original[ids].reshape(len(ids),-1),dtype=torch.float64,device=args.device)
    lam=torch.tensor([float(source[j]['absolute_lambda']) for j in ids],dtype=torch.float64,device=args.device)
    def system(z):
        zimg=z.reshape(-1,1,128,128)
        return op.adjoint(op.forward(zimg,unnormalize=False),normalized=False).flatten(1)+lam[:,None]*z
    rhs=op.adjoint(y,normalized=False).flatten(1);r=rhs-system(x);d=r.clone();rr=(r*r).sum(1)
    norm=rhs.norm(dim=1);t0=time.perf_counter()
    active=r.norm(dim=1)/norm>=1e-9;d[~active]=0
    for step in range(args.maxiter):
        if not bool(active.any()):break
        ad=system(d);alpha=torch.where(active,rr/(d*ad).sum(1).clamp_min(1e-300),0)
        x+=alpha[:,None]*d;r-=alpha[:,None]*ad
        new=(r*r).sum(1);active &= r.norm(dim=1)/norm>=1e-9
        beta=torch.where(active,new/rr.clamp_min(1e-300),0)
        d=r+beta[:,None]*d;d[~active]=0;rr=new
        if (step+1)%100==0:
            print(json.dumps({'step':step+1,'relative_residual':(r.norm(dim=1)/norm).tolist()}),flush=True)
        if bool((r.norm(dim=1)/norm<1e-9).all()):break
    check=(rhs-system(x)).norm(dim=1)/norm
    prediction=x.reshape(-1,1,128,128)
    truth=torch.tensor(np.load(ROOT/'data/inv_scatter_dev/images.npy')[indices],dtype=torch.float64,device=args.device)
    score=mod.metrics(prediction,truth)
    records=[]
    for i,j in enumerate(ids):
        records.append(dict(flat_index=j,object_id=int(source[j]['object_id']),
            relative_alpha=float(source[j]['relative_alpha']),absolute_lambda=float(lam[i]),
            original_psnr=float(source[j]['psnr']),psnr=float(score['psnr'][i]),ssim=float(score['ssim'][i]),
            relative_residual=float(check[i]),passed=bool(check[i]<1e-8),steps=step+1,
            original_relative_residual=float(source[j]['cg_recomputed_relative_residual'])))
    np.savez(out/(args.out_tag+'.npz'),flat_indices=np.array(ids),predictions=prediction.cpu().numpy())
    report=dict(records=records,seconds=time.perf_counter()-t0,all_passed=all(r['passed'] for r in records),
        same_operator='same float32 matrix cache promoted to double; exact same objective',
        originals_preserved=True,device=args.device,warm_start_from_cpu_attempt=warm.exists())
    (out/(args.out_tag+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    assert report['all_passed']


if __name__=='__main__':main()
