"""Frozen nearest-neighbor controls on the existing independent object test."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256
from src.scattering_observation_features import FrozenScatteringFeatures
from src.frozen_scattering_predictor import CombinedQueryOperators
from src.posterior_components import MaskPosteriorComponents


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--noise-seed',type=int,required=True)
    p.add_argument('--device',required=True);a=p.parse_args()
    contract=json.loads((ROOT/'protocols/m7_focused_mechanism_v1.json').read_text())
    assert a.noise_seed in contract['born']['noise_seeds']
    torch.set_num_threads(4);torch.cuda.set_device(torch.device(a.device))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'runs/m7_nppc_v1'/f'noise{a.noise_seed}';out.mkdir(parents=True,exist_ok=False)
    context=FrozenScatteringFeatures(ROOT,a.device);physical=CombinedQueryOperators(context)
    weights=physical.weights.repeat_interleave(2,dim=0).double()
    traces=torch.cat([matrix.double().square().sum(-1)@weights for matrix in physical.matrices])
    cache=ROOT/'data/official_scattering_v1/features'/f'noise{a.noise_seed}_x1'
    inputs={}
    for budget in contract['born']['budgets']:
        prefix=f'mask0_budget{budget}'
        arrays=[np.load(cache/(prefix+'_'+name+'.npy'),allow_pickle=False) for name in ['feature','mean']]
        with np.load(cache/(prefix+'_summary.npz'),allow_pickle=False) as z:masks=z['masks']
        inputs[budget]=arrays+[masks]
    records=[];started=time.time()
    for mode in ['sample','fixed']:
        for seed in contract['born']['seeds']:
            run=ROOT/'runs/m2_nppc_20261101'/f'{mode}_seed{seed}'
            path=run/'best.pt';assert sha256(path)==contract['models'][str(path.relative_to(ROOT))]['sha256']
            calibration=json.loads((run/'calibration/summary.json').read_text())
            assert calibration['checkpoint_sha256']==sha256(path)
            net=MaskPosteriorComponents(8).to(a.device)
            net.load_state_dict(torch.load(path,map_location=a.device,weights_only=True)['model']);net.eval()
            for budget,(features,means,masks) in inputs.items():
                stat=calibration['statistics'][str(budget)]
                scale=torch.tensor(stat['multiplicative_scale'],device=a.device)
                constant=torch.tensor(stat['mean_projected_energy'],device=a.device)
                pieces={k:[] for k in contract['born']['variants']}
                for start in range(0,len(masks),16):
                    f,m,mask=[torch.tensor(v[start:start+16],device=a.device) for v in [features,means,masks]]
                    directions,variance,_=net(f,m,mask);batch,rank,dim=directions.shape
                    plugin=[];dq=[]
                    for matrix in physical.matrices:
                        plugin.append((m.flatten(1)@matrix.T).double().square()@weights)
                        field=(directions.reshape(batch*rank,dim)@matrix.T).double()
                        dq.append((field.square()@weights).reshape(batch,rank,5))
                    plugin=torch.cat(plugin,1);dq=torch.cat(dq,2)
                    tail=traces[None]-dq.sum(1)
                    assert float(tail.min())>-1e-10
                    raw=plugin+(variance[:,:,None]*dq).sum(1)
                    rescaled=plugin+((variance*scale)[:,:,None]*dq).sum(1)
                    const=plugin+(constant[None,:,None]*dq).sum(1)
                    values=dict(raw=raw,rescaled=rescaled,constant=const,
                        rescaled_tail=rescaled+stat['isotropic_complement_variance']*tail,
                        constant_tail=const+stat['isotropic_complement_variance']*tail)
                    for k,v in values.items():pieces[k].append(v.cpu().numpy())
                for variant,parts in pieces.items():
                    pred=np.concatenate(parts);assert pred.shape==(100,35) and np.isfinite(pred).all()
                    dest=out/f'{mode}_{seed}_{variant}_budget{budget}.npz'
                    np.savez_compressed(dest,prediction=pred)
                    records.append(dict(mode=mode,seed=seed,variant=variant,budget=budget,
                                        file=str(dest.relative_to(ROOT)),sha256=sha256(dest)))
            print(json.dumps(dict(mode=mode,seed=seed,seconds=time.time()-started)),flush=True)
    (out/'report.json').write_text(json.dumps(dict(status='completed',predictions=records,
        noise_seed=a.noise_seed,seconds=time.time()-started,scoring_targets_loaded=False),indent=2)+'\n')


if __name__=='__main__':main()
