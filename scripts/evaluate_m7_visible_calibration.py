"""Parameter-free calibration of factor energy using acquired power only."""
from pathlib import Path
import argparse,csv,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256
from src.scattering_observation_features import FrozenScatteringFeatures
from src.frozen_scattering_predictor import CombinedQueryOperators
from src.posterior_components import MaskPosteriorComponents
from src.independent_scattering_assessment import masks_for_ids, observation_noise
from scripts.evaluate_m7_morphology import generate


@torch.no_grad()
def main(args,config):
    torch.set_num_threads(4);torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'runs/m7_visible_calibration_v1'/args.cohort/f'noise{args.noise_seed}'
    out.mkdir(parents=True,exist_ok=False)
    context=FrozenScatteringFeatures(ROOT,args.device);physical=CombinedQueryOperators(context)
    if args.cohort=='official':
        cache=ROOT/f'data/official_scattering_v1/features/noise{args.noise_seed}_x1'
        inputs={}
        for b in config['budgets']:
            pref=f'mask0_budget{b}'
            with np.load(cache/f'{pref}_summary.npz') as z:summary={k:z[k].copy() for k in z.files}
            inputs[b]={key:torch.tensor(np.load(cache/f'{pref}_{key}.npy'),device='cpu') for key in ['feature','mean']}
            inputs[b].update({key:torch.tensor(summary[key]) for key in ['features66','masks','mean_queries35']})
    else:
        morph=json.loads((ROOT/'protocols/m7_morphology_v1.json').read_text())
        images,labels=generate(morph);ids=np.arange(len(images))+morph['id_offset']
        clean=torch.tensor(images,device=args.device).flatten(1)@context.matrix.T
        obs=clean+torch.tensor(observation_noise(ids,args.noise_seed,1.),device=args.device)
        inputs={}
        for b in config['budgets']:
            masks=masks_for_ids(ids,b,0);parts=[]
            for start in range(0,len(ids),32):
                parts.append({k:v.cpu() for k,v in context(obs[start:start+32],masks[start:start+32]).items()})
            inputs[b]={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    # Fixed training response scales for lambda/dx=6, five angular bands.
    scale=torch.tensor(context.query_scale[5:10],device=args.device,dtype=torch.float64)
    specs=[('response',s,f'runs/m2_random_current_20261101/random_current_seed{s}/best.pt')
           for s in [20261126,20261127,20261128]]
    specs += [(f'nppc_{m}',s,f'runs/m2_nppc_20261101/{m}_seed{s}/best.pt')
              for m in ['sample','fixed'] for s in [20261120,20261121,20261122]]
    records=[];began=time.time()
    for family,seed,path in specs:
        assert sha256(ROOT/path)==config['checkpoint_hashes'][path]
        net=MaskPosteriorComponents(8).to(args.device)
        net.load_state_dict(torch.load(ROOT/path,map_location=args.device,weights_only=True)['model']);net.eval()
        for budget,allf in inputs.items():
            predictions=[];gains=[];gained_means=[];visible=[]
            for start in range(0,len(allf['mean']),16):
                f={k:v[start:start+16].to(args.device) for k,v in allf.items()}
                L=net(f['feature'],f['mean'],f['masks'])[2];n,r,d=L.shape
                weights=context.bands[None]*f['masks'].repeat_interleave(6,1)[:,None,None,:]
                weights=weights.double()/weights.sum((-1,-2),keepdim=True)
                pm=(f['mean'].flatten(1)@context.matrix.T).double().reshape(n,20,180,2).square().sum(-1)
                pl=(L.reshape(n*r,d)@context.matrix.T).double().reshape(n,r,20,180,2).square().sum((-1,1))
                p=(pm[:,None]*weights).sum((-1,-2));h=(pl[:,None]*weights).sum((-1,-2))
                s=f['features66'][:,30:35].double()
                residual=(s-p)/scale;basis=h/scale
                den=basis.square().sum(1)
                gamma=torch.where(den>0,(residual*basis).sum(1)/den,torch.zeros_like(den)).clamp_min(0)
                image_basis=p/scale;den_image=image_basis.square().sum(1)
                gain=torch.where(den_image>0,(image_basis*(s/scale)).sum(1)/den_image,torch.zeros_like(den_image)).clamp_min(0)
                qL=physical(L.reshape(n*r,1,128,128)).reshape(n,r,35).sum(1).double()
                qmean=f['mean_queries35'].double()
                predictions.append((qmean+gamma[:,None]*qL).cpu().numpy())
                gained_means.append((gain[:,None]*qmean).cpu().numpy());gains.append(gamma.cpu().numpy())
                before=((p+h-s)/scale).square().sum(1);after=((p+gamma[:,None]*h-s)/scale).square().sum(1)
                assert bool((after<=before+1e-7*(1+before)).all())
                visible.append(torch.stack([before,after],1).cpu().numpy())
            dest=out/f'{family}_{seed}_b{budget}.npz'
            np.savez_compressed(dest,prediction=np.concatenate(predictions),gamma=np.concatenate(gains),visible_loss=np.concatenate(visible))
            records.append(dict(family=family,seed=seed,budget=budget,file=str(dest.relative_to(ROOT)),sha256=sha256(dest)))
            if family=='response' and seed==20261126:
                dest=out/f'image_gain_b{budget}.npz';np.savez_compressed(dest,prediction=np.concatenate(gained_means))
                records.append(dict(family='image_gain',seed=None,budget=budget,file=str(dest.relative_to(ROOT)),sha256=sha256(dest)))
        print(args.cohort,family,seed,round(time.time()-began,2),flush=True)
    (out/'report.json').write_text(json.dumps(dict(status='completed',records=records,
        protocol_sha256=sha256(ROOT/'protocols/m7_visible_calibration_v1.json'),targets_used=False),indent=2)+'\n')


def score(config):
    out=ROOT/'analysis/m7_visible_calibration_v1';out.mkdir(parents=True,exist_ok=False)
    rows=[];rng=np.random.default_rng(2026110211)
    for cohort in config['cohorts']:
        if cohort=='official':
            targets=np.load(ROOT/'data/official_scattering_v1/scoring/targets35.npy');labels=np.repeat('official',len(targets))
        else:
            z=np.load(ROOT/'analysis/m7_morphology_v1/objects_and_targets.npz');targets=z['targets'];labels=z['labels']
        arrays={};factors={}
        for noise in config['noise_seeds']:
            report=json.loads((ROOT/f'runs/m7_visible_calibration_v1/{cohort}/noise{noise}/report.json').read_text())
            assert report['status']=='completed'
            for rec in report['records']:
                assert sha256(ROOT/rec['file'])==rec['sha256'];z=np.load(ROOT/rec['file'])
                key=rec['family'],rec['budget'];arrays.setdefault(key,[]).append(np.abs(z['prediction']/targets-1))
                if 'gamma' in z:factors.setdefault(key,[]).append(z['gamma'])
        for subcohort in np.unique(labels):
            take=labels==subcohort;n=int(take.sum());draws=rng.integers(0,n,(2000,n))
            for (family,budget),values in arrays.items():
                errors=np.mean(values,0)[take,15:35].mean(1);ci=np.quantile(errors[draws].mean(1),[.025,.975])
                rec=dict(cohort=subcohort,family=family,budget=budget,mre=float(errors.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]))
                if (family,budget) in factors:
                    g=np.stack(factors[family,budget])[:,take];rec.update(gamma_mean=float(g.mean()),gamma_median=float(np.median(g)),fraction_zero=float((g==0).mean()))
                rows.append(rec)
    (out/'summary.json').write_text(json.dumps(dict(status='completed',groups=rows),indent=2)+'\n')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'groups.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for r in rows:print(r['cohort'],r['family'],r['budget'],round(r['mre']*100,3),r.get('gamma_mean'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--cohort',choices=['official','morphology'])
    p.add_argument('--noise-seed',type=int);p.add_argument('--score',action='store_true');args=p.parse_args()
    config=json.loads((ROOT/'protocols/m7_visible_calibration_v1.json').read_text())
    assert sha256(Path(__file__))==config['script_sha256']
    if args.score:score(config)
    else:
        assert args.noise_seed in config['noise_seeds'];main(args,config)
