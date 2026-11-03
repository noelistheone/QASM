"""Mean reconstruction on the actual nested action rig, with explicit masks."""
from pathlib import Path
import argparse,csv,json,math,os,shutil,socket,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import BornConfig,BatchedBorn,sha256
from src.response_learning import PhysicalQueries
from src.mask_conditioned_learning import MaskUNet,REMAINING,masks_from_orders,evaluation_masks,masked_cg_feature

def atomic_json(path,data):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['image','joint'],required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--seed',type=int,default=20261117)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--profile',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    root=ROOT/'data/m2_mask_features_v1';parent=ROOT/'data/m2_response_pilot_v1'
    meta=json.loads((root/'manifest.json').read_text());assert meta['status']=='completed'
    split=json.loads((root/'split.json').read_text());train_ids=np.asarray(split['mean_train'])
    assert len(set(train_ids)&set(split['covariance_train']))==0
    images=np.load(parent/'images.npy',mmap_mode='r');targets=np.load(parent/'queries.npy',mmap_mode='r')
    signals=np.load(root/'clean_signals.npy',mmap_mode='r');observations=np.load(ROOT/'data/m2_gaussian_actions_v1/measurements.npy',mmap_mode='r')
    observation_matrix=BatchedBorn(BornConfig(numRec=180),cache_dir=ROOT/'data/born_cache',device=args.device).matrix
    regularization=meta['ridge_info']['lambda']
    scale=torch.tensor(meta['query_scale'],device=args.device,dtype=torch.float32);variance=meta['image_variance']
    matrices=[BatchedBorn(BornConfig(wave=w,numRec=60),cache_dir=ROOT/'data/born_cache',device=args.device).matrix for w in [4,6,8]]
    physical=PhysicalQueries(matrices).to(args.device);del matrices
    net=MaskUNet().to(args.device);optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=np.random.default_rng(args.seed)
    src=[Path(__file__),ROOT/'src/mask_conditioned_learning.py',ROOT/'src/response_learning.py',ROOT/'src/born_scattering.py']
    config=dict(mode=args.mode,seed=args.seed,steps=args.steps,batch=args.batch,device=args.device,host=socket.gethostname(),
        source_hashes={str(s.relative_to(ROOT)):sha256(s) for s in src},data_manifest_sha256=sha256(root/'manifest.json'),
        protocol_sha256=sha256(ROOT/'protocols/m2_mask_conditioned_learning_v2.json'),
        base_protocol_sha256=sha256(ROOT/'protocols/m2_mask_conditioned_learning_v1.json'),
        parameters=sum(p.numel() for p in net.parameters()))
    snapshot=args.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,value in config['source_hashes'].items():
        dst=snapshot/name.replace('/','__')
        if not dst.exists():shutil.copy2(ROOT/name,dst)
        assert sha256(dst)==value
    if (args.out/'summary.json').exists():raise RuntimeError('Already completed')
    start=0;best=float('inf');elapsed=0.;history=[]
    if args.resume:
        old=json.loads((args.out/'config.json').read_text())
        for key in ['mode','seed','steps','batch','source_hashes','data_manifest_sha256','protocol_sha256']:assert old[key]==config[key],key
        cp=torch.load(args.out/'last.pt',map_location=args.device,weights_only=True)
        net.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);start=cp['step'];best=cp['best'];elapsed=cp['seconds'];history=cp['history']
        rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
    else:
        if (args.out/'config.json').exists():raise RuntimeError('Existing run requires --resume')
        atomic_json(args.out/'config.json',config)
    t0=time.perf_counter()
    def batch(indices,mask_array,training):
        source=signals if training else observations
        y=torch.tensor(np.asarray(source[indices].copy()),device=args.device)
        if training:y=y+torch.randn_like(y)*(1e-4/np.sqrt(2))
        mask=torch.tensor(mask_array,device=args.device)
        with torch.no_grad():feature,cgres=masked_cg_feature(y,mask,observation_matrix,regularization)
        if not bool(torch.isfinite(feature).all()):raise FloatingPointError('Nonfinite CG feature')
        batch.last_cg_max=float(cgres.max())
        truth=torch.tensor(np.asarray(images[indices].copy()),device=args.device)
        qt=torch.tensor(np.asarray(targets[indices].copy()),device=args.device)
        return feature,mask,truth,qt
    def losses(feature,mask,truth,qt,training):
        image=net(feature,mask);li=(image-truth).square().mean()/variance
        if training and args.mode=='image':
            with torch.no_grad():q=physical(image)
        else:q=physical(image)
        lq=((q-qt)/scale).square().mean();loss=li+lq if args.mode=='joint' else li
        return loss,li,lq,image,q
    @torch.no_grad()
    def evaluate(ids,save=False):
        net.eval();by_budget=[]
        for budget in [0,2,5,10]:
            all_loss=[];qpred=[];qbase=[];refs=[];predimgs=[];mse=[];psnr=[];bmse=[]
            for i in range(0,len(ids),args.batch):
                idx=ids[i:i+args.batch];feature,mask,truth,qt=batch(idx,evaluation_masks(idx,budget),False)
                loss,li,lq,img,q=losses(feature,mask,truth,qt,False)
                all_loss.append([float(v)*len(idx) for v in [loss,li,lq]])
                qpred.append(q.cpu().numpy());qbase.append(physical(feature).cpu().numpy());refs.append(qt.cpu().numpy())
                mse.extend((img-truth).square().flatten(1).mean(1).cpu().tolist())
                bmse.extend((feature-truth).square().flatten(1).mean(1).cpu().tolist())
                psnr.extend((-10*torch.log10((img.clamp(0,1)-truth).square().flatten(1).mean(1).clamp_min(1e-30))).cpu().tolist())
                if save:predimgs.append(img.cpu().numpy())
            qpred=np.concatenate(qpred);qb=np.concatenate(qbase);qt=np.concatenate(refs);vals=np.asarray(all_loss).sum(0)/len(ids)
            err=np.abs(qpred/qt-1)
            item=dict(additional_actions=budget,objects=len(ids),selection_loss=float(vals[0]),normalized_image_mse=float(vals[1]),
                normalized_query_mse=float(vals[2]),mean_relative_error=float(err.mean()),image_psnr_clipped=float(np.mean(psnr)),
                image_raw_mse=float(np.mean(mse)),feature_raw_mse=float(np.mean(bmse)),feature_query_relative_error=float(np.abs(qb/qt-1).mean()),
                per_query_relative_error=err.mean(0).tolist())
            by_budget.append(item)
            if save:
                np.savez(args.out/f'assessment_budget{budget}.npz',ids=np.asarray(ids),prediction=qpred,reference=qt,feature_query=qb,
                    image_mse=np.asarray(mse),feature_mse=np.asarray(bmse))
                np.save(args.out/f'assessment_images_budget{budget}.npy',np.concatenate(predimgs))
        net.train();return dict(selection_loss=float(np.mean([v['selection_loss'] for v in by_budget])),by_budget=by_budget)
    for step in range(start+1,args.steps+1):
        lr=1e-5+.5*(1e-3-1e-5)*(1+math.cos(math.pi*(step-1)/max(args.steps-1,1)))
        for group in optimizer.param_groups:group['lr']=lr
        ids=rng.choice(train_ids,args.batch,replace=False);extra=rng.integers(0,11,size=args.batch)
        orders=[rng.permutation(REMAINING) for _ in ids];mask=masks_from_orders(orders,extra)
        feature,mt,truth,qt=batch(ids,mask,True);optimizer.zero_grad(set_to_none=True)
        loss,li,lq,_,_=losses(feature,mt,truth,qt,True)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);optimizer.step()
        if step==1 or step%25==0:
            state=dict(status='running',mode=args.mode,seed=args.seed,step=step,total_steps=args.steps,loss=float(loss.detach()),
                image_loss=float(li.detach()),query_loss=float(lq.detach()),gradient_norm=float(grad),lr=lr,
                seconds=elapsed+time.perf_counter()-t0,peak_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,pid=os.getpid(),
                maximum_relative_feature_normal_residual=batch.last_cg_max)
            atomic_json(args.out/'status.json',state);print(json.dumps(state),flush=True)
        if args.profile:atomic_json(args.out/'profile.json',state);return
        if step%400==0 or step==args.steps:
            val=evaluate(split['validation'][:256]);val.update(step=step,seconds=elapsed+time.perf_counter()-t0)
            history.append(val);improved=val['selection_loss']<best
            if improved:best=val['selection_loss']
            cp=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,seconds=val['seconds'],history=history,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            tmp=args.out/'last.partial.pt';torch.save(cp,tmp);tmp.replace(args.out/'last.pt')
            if improved:shutil.copy2(args.out/'last.pt',args.out/'best.pt')
            atomic_json(args.out/'validation_history.json',history);print(json.dumps(dict(validation=val,best=best)),flush=True)
    cp=torch.load(args.out/'best.pt',map_location=args.device,weights_only=True);net.load_state_dict(cp['model'])
    report=dict(status='completed',mode=args.mode,seed=args.seed,best_step=cp['step'],assessment=evaluate(split['assessment'],save=True),
        seconds=elapsed+time.perf_counter()-t0,peak_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
        mean_training_objects=4000,reserved_covariance_training_objects=4000,official_test_opened=False)
    atomic_json(args.out/'summary.json',report);atomic_json(args.out/'status.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
