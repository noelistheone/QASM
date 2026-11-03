"""Train-only Gaussian PCA surrogate for the frozen nested acquisition rig."""
from pathlib import Path
import argparse,json,sys,time,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import BornConfig,BatchedBorn,sha256
from src.response_learning import angular_weights,angular_power

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'data/m2_response_pilot_v1';out=ROOT/'data/m2_gaussian_actions_v1';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Prepared dataset exists')
    contract=ROOT/'protocols/m2_gaussian_action_reference_v1.json';protocol=json.loads(contract.read_text())
    split=json.loads((root/'split.json').read_text());images=np.load(root/'images.npy',mmap_mode='r')
    train=torch.tensor(np.asarray(images[split['train']].copy()),device=args.device).flatten(1)
    t0=time.perf_counter();mean=train.mean(0);train-=mean
    torch.manual_seed(protocol['pca_seed']);left,s,v=torch.pca_lowrank(train,q=160,center=False,niter=4)
    variance=float(train.square().sum())/(len(train)-1);eigenvalues=s[:128].square()/(len(train)-1)
    v=v[:,:128];basis=v*torch.sqrt(eigenvalues)[None,:]
    del left,train
    np.savez(out/'prior.npz',mean=mean.cpu().numpy(),basis=basis.cpu().numpy(),eigenvalues=eigenvalues.cpu().numpy(),
        total_training_variance=variance,explained_fraction=eigenvalues.cumsum(0).cpu().numpy()/variance)
    rig=BatchedBorn(BornConfig(numRec=180),cache_dir=ROOT/'data/born_cache',device=args.device).matrix
    h=rig@basis;center=rig@mean
    measurements=np.lib.format.open_memmap(out/'measurements.npy',mode='w+',dtype=np.float32,shape=(10000,7200))
    projections=np.lib.format.open_memmap(out/'latent_scores.npy',mode='w+',dtype=np.float32,shape=(10000,128))
    generator=torch.Generator(device=args.device).manual_seed(protocol['noise_seed'])
    train_mask=np.zeros(10000,dtype=bool);train_mask[split['train']]=True
    tail_sums={r:torch.zeros(7200,device=args.device,dtype=torch.double) for r in protocol['ranks']}
    for start in range(0,10000,128):
        stop=min(start+128,10000);x=torch.tensor(np.asarray(images[start:stop].copy()),device=args.device).flatten(1)
        z=((x-mean)@v)/torch.sqrt(eigenvalues);signal=x@rig.T
        noisy=signal+torch.randn(signal.shape,generator=generator,device=args.device)*(1e-4/np.sqrt(2))
        measurements[start:stop]=noisy.cpu().numpy();projections[start:stop]=z.cpu().numpy()
        mask=torch.tensor(train_mask[start:stop],device=args.device)
        for rank in protocol['ranks']:
            residual=signal-center-z[:,:rank]@h[:,:rank].T
            tail_sums[rank]+=residual[mask].double().square().sum(0)
    measurements.flush();projections.flush();np.save(out/'observation_matrix.npy',h.cpu().numpy());np.save(out/'observation_center.npy',center.cpu().numpy())
    weights=angular_weights(args.device);query_parts={r:[] for r in protocol['ranks']};tail_query={r:[] for r in protocol['ranks']}
    for wave in [4,6,8]:
        a=BatchedBorn(BornConfig(wave=wave,numRec=60),cache_dir=ROOT/'data/born_cache',device=args.device).matrix
        b=a@basis;m=a@mean;w=weights.repeat_interleave(2,0)
        for rank in protocol['ranks']:
            br=b[:,:rank].double();md=m.double();wd=w.double()
            qm=torch.einsum('mi,mq,mj->qij',br,wd,br)
            linear=torch.einsum('mi,mq,m->qi',br,wd,md)
            const=torch.einsum('m,mq,m->q',md,wd,md)
            query_parts[rank].append((qm.cpu().numpy(),linear.cpu().numpy(),const.cpu().numpy()))
            accum=torch.zeros(5,device=args.device,dtype=torch.double)
            for j in range(0,len(split['train']),128):
                idx=split['train'][j:j+128];x=torch.tensor(np.asarray(images[idx].copy()),device=args.device).flatten(1)
                z=torch.tensor(np.asarray(projections[idx,:rank].copy()),device=args.device)
                residual=x@a.T-m-z@b[:,:rank].T
                accum+=angular_power(residual,weights).double().sum(0)
            tail_query[rank].append((accum/len(split['train'])).cpu().numpy())
    checks=[]
    for rank in protocol['ranks']:
        q,s,c=[np.concatenate([part[j] for part in query_parts[rank]]) for j in range(3)]
        noise=(tail_sums[rank]/len(split['train'])).cpu().numpy()+1e-8/2
        assert np.min(noise)>0 and np.min(np.linalg.eigvalsh(q))>=-1e-12
        zcheck=torch.tensor(np.random.default_rng(20261115).normal(size=(3,rank)),device=args.device,dtype=torch.float32)
        xcheck=mean+zcheck@basis[:,:rank].T
        exact=[]
        for wave in [4,6,8]:
            a=BatchedBorn(BornConfig(wave=wave,numRec=60),cache_dir=ROOT/'data/born_cache',device=args.device).matrix
            exact.append(angular_power(xcheck@a.T,weights).cpu().numpy())
        znp=zcheck.cpu().numpy().astype(float)
        latent=np.einsum('ni,qij,nj->nq',znp,q,znp)+2*znp@s.T+c
        exact=np.concatenate(exact,axis=1);err=float(np.max(np.abs(latent-exact)/np.maximum(exact,1e-20)))
        assert err<1e-4
        checks.append(dict(rank=rank,affine_query_identity_max_relative_error=err))
        np.savez(out/f'rank{rank}.npz',Q=q,linear=s,constant=c,tail_query=np.concatenate(tail_query[rank]),
            noise_variance=noise,eigenvalues=eigenvalues[:rank].cpu().numpy())
    report=dict(status='completed',seconds=time.perf_counter()-t0,protocol_sha256=sha256(contract),
        data_manifest_sha256=sha256(root/'manifest.json'),training_objects=len(split['train']),
        explained_variance={r:float(eigenvalues[:r].sum()/variance) for r in protocol['ranks']},checks=checks,
        hashes={p.name:sha256(p) for p in out.glob('*.np*')},
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/born_scattering.py',ROOT/'src/response_learning.py']})
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in report['source_hashes']:shutil.copy2(ROOT/name,snap/name.replace('/','__'))
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
