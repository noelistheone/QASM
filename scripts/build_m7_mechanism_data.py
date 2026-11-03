"""Fixed-index spatial illustrations and rank-one objective check."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.posterior_components import MaskPosteriorComponents


@torch.no_grad()
def main():
    out=ROOT/'analysis/m7_mechanism_v1';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False
    cache=ROOT/'data/official_scattering_v1/features/noise20270111_x1'
    f=torch.tensor(np.load(cache/'mask0_budget0_feature.npy')[:1],device='cuda:0')
    m=torch.tensor(np.load(cache/'mask0_budget0_mean.npy')[:1],device='cuda:0')
    mask=torch.tensor(np.load(cache/'mask0_budget0_summary.npz')['masks'][:1],device='cuda:0')
    maps=dict(mean=m[0,0].cpu().numpy(),object=np.load(ROOT/"data/official_scattering_v1/scoring/objects.npy",mmap_mode="r")[0,0].copy())
    for name,path in [('nppc','runs/m2_nppc_20261101/sample_seed20261120/best.pt'),
                      ('response','runs/m2_random_current_20261101/random_current_seed20261126/best.pt')]:
        net=MaskPosteriorComponents(8).cuda();net.load_state_dict(torch.load(ROOT/path,map_location='cuda:0',weights_only=True)['model']);net.eval()
        factors=net(f,m,mask)[2][0].cpu().numpy().reshape(8,128,128)
        maps[name+'_factor_norm']=np.sqrt((factors*factors).sum(0))
    np.savez_compressed(out/'fixed_index_maps.npz',**maps)
    target=np.load(ROOT/'data/official_scattering_v1/scoring/targets35.npy')
    errors={'NPPC':[],'Response-trained':[],'Mean image':[]}
    for noise in [20270111,20270112,20270113]:
        cache=ROOT/f'data/official_scattering_v1/features/noise{noise}_x1'
        pred=np.load(cache/'mask0_budget0_summary.npz')['mean_queries35']
        errors['Mean image'].append(np.abs(pred/target-1).mean(0))
        for seed in [20261120,20261121,20261122]:
            pred=np.load(ROOT/f'runs/m7_nppc_v1/noise{noise}/sample_{seed}_raw_budget0.npz')['prediction']
            errors['NPPC'].append(np.abs(pred/target-1).mean(0))
        for seed in [20261126,20261127,20261128]:
            pred=np.load(ROOT/f'runs/m3_official_assessment_v1/noise{noise}_x1/primary_moment8_seed{seed}/mask0_budget0.npz')['prediction']
            errors['Response-trained'].append(np.abs(pred/target-1).mean(0))
    (out/'wavelength_errors.json').write_text(json.dumps({k:np.mean(v,axis=0).reshape(7,5).mean(1).tolist() for k,v in errors.items()},indent=2)+'\n')
    # Exact scalar minimizer for a fixed rank-one direction, including the
    # nonnegative boundary; numerical derivative and candidate checks.
    rng=np.random.default_rng(2026110208);checks=[]
    for _ in range(20):
        b=rng.normal(size=(5,4,4));q=np.einsum('kji,kjl->kil',b,b)
        v=rng.normal(size=4);v/=np.linalg.norm(v);h=np.einsum('i,kij,j->k',v,q,v)
        d=rng.normal(size=5);w=np.exp(rng.normal(size=5))
        numerator=np.sum(w*d*h);denominator=np.sum(w*h*h);s=max(numerator,0)/denominator
        loss=lambda t:float(np.sum(w*(t*h-d)**2))
        assert loss(s)<=loss(0)+1e-12
        derivative=2*np.sum(w*h*(s*h-d))
        assert abs(derivative)<1e-10 if s>0 else derivative>=-1e-10
        checks.append(dict(scale=s,stationarity=float(derivative),loss=loss(s)))
    (out/'rank_one_checks.json').write_text(json.dumps(dict(status='passed',checks=checks,
        counterexample=dict(covariance_diagonal=[9,1],query_diagonal=[0,1],image_rank1_query=0,response_rank1_query=1,true_query=1)),indent=2)+'\n')


if __name__=='__main__':main()
