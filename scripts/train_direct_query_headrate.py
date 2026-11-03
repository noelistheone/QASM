"""Readout-rate sensitivity for the unchanged matched scalar-query CNNs."""
from pathlib import Path
import argparse,json,os,shutil,socket,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256,BornConfig,BatchedBorn
from src.mask_learning_context import MaskLearningContext
from src.mask_conditioned_learning import REMAINING,masks_from_orders,evaluation_masks
from src.direct_query_cnn_additive import AdditiveDirectQueryCNN
from src.direct_query_cnn import DirectQueryCNN
from src.response_learning import PhysicalQueries


def save_json(path,d):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(d,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--readout',choices=['multiplicative','additive'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();torch.set_num_threads(4)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    c=MaskLearningContext(ROOT,ROOT/'runs/m2_mask_learning_20261101/image_seed20261117',a.device)
    data=ROOT/'data/m2_frozen_amplitudes_v1';frozen=np.load(data/'train.npz');x=frozen['features'][:,:66];logs=np.log(frozen['mean_query'])
    fm=x.mean(0);fs=np.maximum(x.std(0),1e-12);lm=float(logs.mean());ls=max(float(logs.std()),1e-12);del frozen,x,logs
    net=(AdditiveDirectQueryCNN(fm,fs,lm,ls,c.query_scale.cpu().numpy()) if a.readout=='additive' else DirectQueryCNN(fm,fs,lm,ls)).to(a.device)
    optimizer=torch.optim.Adam([dict(params=list(net.spatial.parameters()),lr=1e-4),dict(params=list(net.head.parameters())+list(net.norm.parameters()),lr=1e-3)],betas=(.9,.999))
    rng=np.random.default_rng(a.seed);train=np.asarray(c.split['covariance_train'])
    files=[Path(__file__)]+[ROOT/'src'/f for f in ['direct_query_cnn.py','direct_query_cnn_additive.py','posterior_components.py','mask_learning_context.py','mask_conditioned_learning.py','response_learning.py','born_scattering.py']]
    config=dict(readout=a.readout,backbone_lr=1e-4,head_lr=1e-3,seed=a.seed,device=a.device,host=socket.gethostname(),steps=4000,batch=128,profile=a.profile,
        parameters=sum(v.numel() for v in net.parameters()),mean_checkpoint_sha256=c.mean_checkpoint_hash,
        source_hashes={str(f.relative_to(ROOT)):sha256(f) for f in files},protocol_sha256=sha256(ROOT/'protocols/m2_direct_query_headrate_v1.json'),
        feature_statistics_source_sha256=sha256(data/'train.npz'),torch_version=torch.__version__)
    snapshot=a.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,digest in config['source_hashes'].items():
        dest=snapshot/name.replace('/','__')
        if not dest.exists():shutil.copy2(ROOT/name,dest)
        assert sha256(dest)==digest
    if (a.out/'summary.json').exists():raise RuntimeError('Completed run')
    first=0;best=float('inf');history=[];elapsed=0.
    if a.resume:
        old=json.loads((a.out/'config.json').read_text())
        for key in ['readout','backbone_lr','head_lr','seed','steps','batch','source_hashes','protocol_sha256','feature_statistics_source_sha256']:assert old[key]==config[key],key
        checkpoint=torch.load(a.out/'last.pt',map_location=a.device,weights_only=True)
        net.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        first,best,history,elapsed=[checkpoint[k] for k in ['step','best','history','seconds']]
        rng.bit_generator.state=checkpoint['numpy_rng'];torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
    else:
        if (a.out/'config.json').exists():raise RuntimeError('Inspect partial fit before resume')
        save_json(a.out/'config.json',config)
    setup=time.perf_counter()-started;t0=time.perf_counter()

    @torch.no_grad()
    def evaluate(ids,final=False):
        net.eval();rows=[]
        if final:
            new=PhysicalQueries([BatchedBorn(BornConfig(wave=w,numRec=60),cache_dir=ROOT/'data/born_cache',device=a.device).matrix for w in [5,7,10,12]]).to(a.device)
            new_targets=np.load(ROOT/'data/m2_transfer_queries_v1/queries.npy',mmap_mode='r')
        else:new=None
        for budget in ([0,2,5,9,10] if final else [0,2,5,9]):
            predictions=[];references=[]
            for j in range(0,len(ids),128):
                index=ids[j:j+128];b=c.batch(index,evaluation_masks(index,budget));pred=net(b,c.physical,new)
                target=b['target']
                if final:target=torch.cat([target,torch.tensor(np.asarray(new_targets[index]),device=a.device)],1)
                predictions.append(pred.cpu().numpy());references.append(target.cpu().numpy())
            pred=np.concatenate(predictions);ref=np.concatenate(references);err=np.abs(pred/ref-1)
            row=dict(budget=budget,objects=len(ids),original_mre=float(err[:,:15].mean()),
                original_normalized_mse=float(np.mean(((pred[:,:15]-ref[:,:15])/c.query_scale.cpu().numpy())**2)),nonpositive_predictions=int((pred<=0).sum()))
            if final:
                row.update(transfer_mre=float(err[:,15:].mean()),interpolation_mre=float(err[:,15:25].mean()),extrapolation_mre=float(err[:,25:].mean()))
                np.savez(a.out/f'budget{budget}.npz',ids=np.asarray(ids),prediction=pred,reference=ref)
            rows.append(row)
        net.train();return rows

    for step in range(first+1,3 if a.profile else 4001):
        ids=rng.choice(train,128,replace=False);extra=rng.integers(0,10,128)
        mask=masks_from_orders([rng.permutation(REMAINING) for _ in ids],extra)
        _=[rng.choice(np.flatnonzero(v==0)) for v in mask] # Match current/future mask RNG advancement.
        b=c.batch(ids,mask,training=True);pred=net(b,c.physical);loss=((pred-b['target'])/c.query_scale).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite direct-query loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not bool(torch.isfinite(gradient)):raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if step in [1,2] or step%25==0:
            status=dict(status='running',seed=a.seed,step=step,total_steps=4000,loss=float(loss.detach()),gradient=float(gradient),
                seconds=elapsed+time.perf_counter()-t0,peak_GiB=torch.cuda.max_memory_allocated(a.device)/2**30,pid=os.getpid())
            save_json(a.out/'status.json',status);print(json.dumps(status),flush=True)
        if a.profile and step==2:
            status['status']='profile_completed';save_json(a.out/'profile.json',status);return
        if step%400==0:
            rows=evaluate(c.split['validation'][:256]);score=float(np.mean([v['original_normalized_mse'] for v in rows]))
            history.append(dict(step=step,selection_loss=score,results=rows));improved=score<best
            if improved:best=score
            checkpoint=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,history=history,seconds=elapsed+time.perf_counter()-t0,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            temp=a.out/'last.partial.pt';torch.save(checkpoint,temp);temp.replace(a.out/'last.pt')
            if improved:shutil.copy2(a.out/'last.pt',a.out/'best.pt')
            save_json(a.out/'validation_history.json',history);print(json.dumps(history[-1]),flush=True)
    checkpoint=torch.load(a.out/'best.pt',map_location=a.device,weights_only=True);net.load_state_dict(checkpoint['model'])
    report=dict(status='completed',readout=a.readout,seed=a.seed,best_step=checkpoint['step'],results=evaluate(c.split['assessment'],True),
        seconds=elapsed+time.perf_counter()-t0,setup_seconds=setup,parameters=config['parameters'],peak_GiB=torch.cuda.max_memory_allocated(a.device)/2**30,
        config_sha256=sha256(a.out/'config.json'),checkpoint_sha256=sha256(a.out/'best.pt'),official_test_opened=False)
    save_json(a.out/'summary.json',report);save_json(a.out/'status.json',dict(status='completed',best_step=checkpoint['step']));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
