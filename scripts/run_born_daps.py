"""Development reproduction of DAPS with independent batched posterior draws.

Updates follow pinned InverseBench/algo/daps.py (MIT license in vendor).
Task-specific hyperparameters are Table 12 of the ICLR 2025 paper, not the
generic repository defaults. Official test data are not accessed by this file.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import datetime
import json
import os
import platform
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/InverseBench')]
from models.precond import EDMPrecond
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
from src.born_scattering import BornConfig, BatchedBorn, sha256


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(p)


def atomic_checkpoint(path, value):
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    torch.save(value,tmp)
    tmp.replace(p)


def load_prior(device):
    model = EDMPrecond(img_resolution=128,img_channels=1,label_dim=0,
        model_type='DhariwalUNet',model_channels=128,channel_mult=[1,1,1,2,2],
        attn_resolutions=[16],num_blocks=1,dropout=0.)
    state = torch.load(ROOT/'checkpoints/inv-scatter-5m.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['ema'])
    return model.eval().requires_grad_(False).to(device)


class DenoiserBatches:
    """Inference-only memory partition; no cross-sample normalization in prior."""
    def __init__(self, net, microbatch):
        self.net, self.microbatch = net, microbatch

    def __call__(self, x, sigma):
        return torch.cat([self.net(part,sigma) for part in x.split(self.microbatch)],dim=0)


def metrics(prediction, truth):
    """Physical-domain clipped PSNR/SSIM; raw RMSE is recorded separately."""
    x, y = prediction.clamp(0,1), truth.clamp(0,1)
    mse = (x-y).square().flatten(1).mean(1)
    psnr = -10*torch.log10(mse.clamp_min(1e-30))
    grid = torch.arange(11,device=x.device,dtype=x.dtype)-5
    g = torch.exp(-grid.square()/(2*1.5**2)); g /= g.sum()
    kernel = (g[:,None]*g[None,:])[None,None]
    mx, my = F.conv2d(x,kernel), F.conv2d(y,kernel)
    vx = F.conv2d(x*x,kernel)-mx*mx
    vy = F.conv2d(y*y,kernel)-my*my
    cov = F.conv2d(x*y,kernel)-mx*my
    ssim = ((2*mx*my+.01**2)*(2*cov+.03**2)/
            ((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).flatten(1).mean(1)
    return {'psnr':psnr,'ssim':ssim,
            'raw_rmse':(prediction-truth).square().flatten(1).mean(1).sqrt()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receivers',type=int,choices=[60,180,360],required=True)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--microbatch',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261101)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--diffusion-steps',type=int,default=10)
    p.add_argument('--langevin-steps',type=int,default=50)
    p.add_argument('--objects',type=int,default=10)
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--profile-only',action='store_true')
    args=p.parse_args()
    if args.objects < 1 or args.objects > 10 or args.samples < 1:
        p.error('Use 1–10 development objects and at least one sample')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg=BornConfig(numRec=args.receivers)
    lr={360:4e-5,180:8e-5,60:2e-4}[args.receivers]
    min_ratio={360:1.,180:1.,60:.5}[args.receivers]
    weights=ROOT/'checkpoints/inv-scatter-5m.pt'
    images_path=ROOT/'data/inv_scatter_dev/images.npy'
    frozen=dict(operator=asdict(cfg),samples_per_object=args.samples,objects=list(range(args.objects)),
        denoiser_microbatch=args.microbatch,
        seed=args.seed,annealing_steps=args.steps,diffusion_steps=args.diffusion_steps,
        langevin_steps=args.langevin_steps,langevin_lr=lr,lr_min_ratio=min_ratio,tau=1e-4,
        annealing_sigma_max=100.,annealing_sigma_min=.1,diffusion_sigma_min=.01,
        noise_real_std=cfg.sigma_noise/np.sqrt(2),split='official validation/development only',
        prior_sha256=sha256(weights),images_sha256=sha256(images_path),
        source_commit=cfg.source_commit,precision='float32; TF32 off',
        reference='InverseBench ICLR2025 Table12; sigma schedules from pinned code',
        samples_role='approximate posterior draws, not independent ground-truth objects',
        changes_from_upstream=['vectorized independently across objects/draws',
          'float32 forward/analytic adjoint after float64 parity checks',
          'explicit nonfinite failure instead of silent zero output',
          'measurement generator independent of posterior-sampling RNG'],
        script_sha256=sha256(Path(__file__)),operator_sha256=sha256(ROOT/'src/born_scattering.py'))
    config_path=args.out/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text()) != frozen:
        raise RuntimeError('Run protocol differs: use a distinct run directory')
    atomic_json(config_path,frozen)
    environment=dict(host=socket.gethostname(),pid=os.getpid(),python=platform.python_version(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cwd=str(ROOT),device=str(device))
    atomic_json(args.out/'environment.json',environment)
    t0=time.perf_counter()
    operator=BatchedBorn(cfg,cache_dir=ROOT/'data/born_cache',device=device)
    net=DenoiserBatches(load_prior(device),args.microbatch)
    truth=torch.tensor(np.load(images_path)[:args.objects],device=device)
    measurement_generator=torch.Generator(device=device).manual_seed(90210)
    observation=operator.noisy(truth*2-1,measurement_generator)
    np.save(args.out/'measurements_real.npy',observation.cpu().numpy())
    observation=observation.repeat_interleave(args.samples,dim=0)
    batch=args.samples*args.objects
    outer=Scheduler(num_steps=args.steps,sigma_max=100.,sigma_min=.1,sigma_final=0,
                    schedule='linear',timestep='poly-7')
    xt=torch.randn(batch,1,128,128,device=device)*outer.sigma_max
    torch.cuda.reset_peak_memory_stats(device)
    if args.profile_only:
        with torch.no_grad():
            for _ in range(2):
                q=net(xt,torch.tensor(1.,device=device))
                grad=operator.gradient(q,observation)
        torch.cuda.synchronize()
        report=dict(batch=batch,seconds=time.perf_counter()-t0,
            peak_GiB=torch.cuda.max_memory_allocated(device)/2**30,
            reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30,
            free_GiB=torch.cuda.mem_get_info(device)[0]/2**30,
            finite=bool(torch.isfinite(q).all() and torch.isfinite(grad).all()))
        atomic_json(args.out/'profile.json',report)
        print(json.dumps(report),flush=True)
        return
    cp=args.out/'checkpoint.pt'
    start_step=0
    elapsed_before=0.
    if args.resume and cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        xt=saved['xt'].to(device)
        start_step=saved['step']
        elapsed_before=saved['elapsed_seconds']
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'],device)
    elif cp.exists():
        raise RuntimeError('Existing checkpoint; use --resume')
    t_sample=time.perf_counter()
    atomic_json(args.out/'status.json',dict(status='running',step=start_step,total_steps=args.steps,
                                           batch=batch,host=environment['host'],pid=os.getpid()))
    with torch.no_grad():
        for step in range(start_step,args.steps):
            sigma=outer.sigma_steps[step]
            schedule=Scheduler(num_steps=args.diffusion_steps,sigma_max=sigma,sigma_min=.01,
                               sigma_final=0,schedule='linear',timestep='poly-7')
            x0=DiffusionSampler(schedule).sample(net,xt,SDE=False,verbose=False)
            x=x0.clone()
            step_lr=lr*(1+(step/args.steps)*(min_ratio-1))
            for _ in range(args.langevin_steps):
                gradient=operator.gradient(x,observation)/(2*1e-8)+(x-x0)/sigma**2
                x-=step_lr*gradient
                x+=np.sqrt(2*step_lr)*torch.randn_like(x)
            xt=x+torch.randn_like(x)*outer.sigma_steps[step+1]
            if not bool(torch.isfinite(xt).all()):
                atomic_json(args.out/'status.json',dict(status='failed_nonfinite',step=step,
                    finite_sample_count=int(torch.isfinite(xt).flatten(1).all(1).sum()),batch=batch))
                raise FloatingPointError(f'Nonfinite DAPS state at step {step}')
            if (step+1)%args.checkpoint_every == 0 or step+1 == args.steps:
                torch.cuda.synchronize()
                elapsed=elapsed_before+time.perf_counter()-t_sample
                status=dict(status='running',step=step+1,total_steps=args.steps,
                    batch=batch,elapsed_seconds=elapsed,
                    expected_remaining_seconds=elapsed/(step+1)*(args.steps-step-1),
                    peak_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                    data_rmse=float(operator.loss(xt,observation).mean().sqrt()/np.sqrt(observation.shape[1])),
                    range=[float(xt.min()),float(xt.max())],host=environment['host'],pid=os.getpid())
                atomic_json(args.out/'status.json',status)
                with (args.out/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(status)+'\n')
                atomic_checkpoint(cp,dict(xt=xt.cpu(),step=step+1,elapsed_seconds=elapsed,
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)))
                print(json.dumps(status),flush=True)
        raw=(xt+1)*.5
        targets=truth.repeat_interleave(args.samples,dim=0)
        scored=metrics(raw,targets)
        residual=(operator.forward(xt)-observation).square().mean(1).sqrt()
        rows=[]
        for j in range(batch):
            row=dict(object_id=j//args.samples,draw=j%args.samples,
                     **{key:float(value[j]) for key,value in scored.items()},measurement_rmse=float(residual[j]))
            rows.append(row)
        with (args.out/'metrics_per_draw.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        array=raw.cpu().numpy().reshape(args.objects,args.samples,1,128,128)
        np.save(args.out/'posterior_draws.npy',array)
        mean_prediction=raw.reshape(args.objects,args.samples,1,128,128).mean(1)
        mean_metrics=metrics(mean_prediction,truth)
        summary=dict(status='completed',n_objects=args.objects,draws_per_object=args.samples,
            sample_average={key:float(value.mean()) for key,value in scored.items()},
            posterior_mean={key:float(value.mean()) for key,value in mean_metrics.items()},
            objects=[dict(object_id=i,**{key:float(value.reshape(args.objects,args.samples)[i].mean())
                                       for key,value in scored.items()}) for i in range(args.objects)],
            measurement_rmse=float(residual.mean()),
            sampling_seconds=elapsed_before+time.perf_counter()-t_sample,
            setup_seconds=t_sample-t0,peak_GiB=torch.cuda.max_memory_allocated(device)/2**30,
            operator_calls=operator.calls,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(args.out/'summary.json',summary)
        atomic_json(args.out/'status.json',summary)
        print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            out.mkdir(parents=True,exist_ok=True)
            atomic_json(out/'failure.json',dict(status='failed',type=type(error).__name__,
                message=str(error),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise
