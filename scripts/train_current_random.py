"""Random-initialization control for the unchanged current-query objective."""
from pathlib import Path
import argparse,json,os,shutil,socket,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256
from src.mask_learning_context import MaskLearningContext
from src.mask_conditioned_learning import REMAINING,masks_from_orders,evaluation_masks
from src.posterior_components import MaskPosteriorComponents
from src.future_update_objective import update_objective
from src.low_rank_physical_update import project_queries,quadratic_readout
from src.gaussian_actions import action_rows


def atomic_json(path,data):
    temporary=path.with_suffix('.partial.json');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['current'],default='current')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--profile',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    mean_run=ROOT/'runs/m2_mask_learning_20261101/image_seed20261117'
    context=MaskLearningContext(ROOT,mean_run,a.device)
    net=MaskPosteriorComponents(8).to(a.device)
    initial_path=a.out/'initial.pt'
    if not initial_path.exists():torch.save(dict(model=net.state_dict(),seed=a.seed,step=0),initial_path)
    initial=torch.load(initial_path,map_location=a.device,weights_only=True)
    assert initial['seed']==a.seed and initial['step']==0
    net.load_state_dict(initial['model']);del initial
    initial_step=0;completed=True;energy_scale=1.0 # unused by the current-query objective
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-4,betas=(.9,.999))
    rng=np.random.default_rng(a.seed);train=np.asarray(context.split['covariance_train'])
    rows=torch.tensor(action_rows(),device=a.device)
    files=[Path(__file__)]+[ROOT/'src'/f for f in ['future_update_objective.py','low_rank_physical_update.py',
        'posterior_components.py','mask_learning_context.py','mask_conditioned_learning.py','response_learning.py',
        'gaussian_actions.py','born_scattering.py']]
    config=dict(mode=a.mode,seed=a.seed,steps=a.steps,batch=a.batch,device=a.device,host=socket.gethostname(),
        initialization_sha256=sha256(initial_path),initialization_completed=completed,initialization_step=initial_step,initialization_kind='random_without_NPPC',
        mean_checkpoint_sha256=context.mean_checkpoint_hash,profile=a.profile,energy_scale=energy_scale,
        source_hashes={str(f.relative_to(ROOT)):sha256(f) for f in files},
        protocol_sha256=sha256(ROOT/'protocols/m2_current_random_initialization_v1.json'),
        data_manifest_sha256=sha256(ROOT/'data/m2_mask_features_v1/manifest.json'))
    snap=a.out/'source_snapshot';snap.mkdir(exist_ok=True)
    for name,digest in config['source_hashes'].items():
        dest=snap/name.replace('/','__')
        if not dest.exists():shutil.copy2(ROOT/name,dest)
        assert sha256(dest)==digest
    if (a.out/'summary.json').exists():raise RuntimeError('Already completed')
    step_start=0;best=float('inf');elapsed=0.;history=[]
    if a.resume:
        old=json.loads((a.out/'config.json').read_text())
        for key in ['mode','seed','steps','batch','initialization_sha256','mean_checkpoint_sha256','source_hashes',
                    'protocol_sha256','data_manifest_sha256']:assert config[key]==old[key],key
        cp=torch.load(a.out/'last.pt',map_location=a.device,weights_only=True)
        net.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
        step_start,best,elapsed,history=[cp[k] for k in ['step','best','seconds','history']]
        rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
    else:
        if (a.out/'config.json').exists():raise RuntimeError('Existing run requires inspected resume')
        atomic_json(a.out/'config.json',config)
    setup_seconds=time.perf_counter()-started;t0=time.perf_counter()

    @torch.no_grad()
    def evaluate(ids,save=False):
        net.eval();reports=[]
        for budget in [0,2,5,9]:
            accum=np.zeros(3);parts={k:[] for k in ['current','future','reference','mahalanobis','nll','image_mse','factor_energy']}
            for start in range(0,len(ids),a.batch):
                index=ids[start:start+a.batch];mask=evaluation_masks(index,budget)
                action=np.argmax(evaluation_masks(index,budget+1)-mask,axis=1)
                b=context.batch(index,mask)
                result=update_objective(net,b,context,a.mode,torch.tensor(action,device=a.device),rows,energy_scale)
                accum+=np.asarray([float(result[k])*len(index) for k in ['loss','current_loss','future_loss']])
                values=dict(current=result['current'],future=result['future'],reference=b['target'],
                            mahalanobis=result['update']['predictive_mahalanobis'],nll=result['update']['predictive_nll'],
                            image_mse=(b['mean']-b['truth']).square().flatten(1).mean(1),
                            factor_energy=result['factors'].square().sum(-1))
                for key,value in values.items():parts[key].append(value.cpu().numpy())
            arrays={k:np.concatenate(v) for k,v in parts.items()};values=accum/len(ids)
            report=dict(additional_actions=budget,objects=len(ids),selection_loss=float(values[0]),
                        current_query_mse=float(values[1]),future_query_mse=float(values[2]),
                        current_mre=float(np.abs(arrays['current']/arrays['reference']-1).mean()),
                        future_mre=float(np.abs(arrays['future']/arrays['reference']-1).mean()),
                        mean_innovation_mahalanobis_per_channel=float(arrays['mahalanobis'].mean()/240),
                        mean_predictive_nll=float(arrays['nll'].mean()),
                        image_raw_mse=float(arrays['image_mse'].mean()),
                        mean_factor_energy=arrays['factor_energy'].mean(0).tolist(),
                        nonpositive_queries=int((arrays['current']<=0).sum()+(arrays['future']<=0).sum()))
            reports.append(report)
            if save:np.savez(a.out/f'assessment_one_step_budget{budget}.npz',ids=np.asarray(ids),**arrays)
        net.train();return dict(selection_loss=float(np.mean([v['selection_loss'] for v in reports])),by_budget=reports)

    @torch.no_grad()
    def evaluate_current(ids):
        net.eval();reports=[]
        for budget in [0,2,5,10]:
            predictions=[];targets=[]
            for start in range(0,len(ids),a.batch):
                index=ids[start:start+a.batch];b=context.batch(index,evaluation_masks(index,budget))
                # The stop-gradient choice changes backward paths, not forward factors.
                factors=net(b['feature'],b['mean'],b['mask'])[2]
                prediction=quadratic_readout(project_queries(b['mean'],factors,context.physical))
                predictions.append(prediction.cpu().numpy());targets.append(b['target'].cpu().numpy())
            prediction=np.concatenate(predictions);reference=np.concatenate(targets);error=np.abs(prediction/reference-1)
            report=dict(additional_actions=budget,objects=len(ids),mean_relative_error=float(error.mean()),
                        normalized_query_mse=float(np.mean(((prediction-reference)/context.query_scale.cpu().numpy())**2)),
                        per_query_relative_error=error.mean(0).tolist())
            reports.append(report);np.savez(a.out/f'assessment_budget{budget}.npz',ids=np.asarray(ids),prediction=prediction,reference=reference)
        return reports

    for step in range(step_start+1,a.steps+1):
        ids=rng.choice(train,a.batch,replace=False);extra=rng.integers(0,10,a.batch)
        mask=masks_from_orders([rng.permutation(REMAINING) for _ in ids],extra)
        actions=np.array([rng.choice(np.flatnonzero(v==0)) for v in mask])
        b=context.batch(ids,mask,training=True);optimizer.zero_grad(set_to_none=True)
        result=update_objective(net,b,context,a.mode,torch.tensor(actions,device=a.device),rows,energy_scale,
                                diagnostics=a.mode!='continue_fixed')
        loss=result['loss']
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not bool(torch.isfinite(gradient)):raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if step==1 or step%25==0:
            state=dict(status='running',mode=a.mode,seed=a.seed,step=step,total_steps=a.steps,loss=float(loss.detach()),
                       gradient_norm=float(gradient),seconds=elapsed+time.perf_counter()-t0,
                       peak_GiB=torch.cuda.max_memory_allocated(a.device)/2**30,pid=os.getpid())
            if a.mode!='continue_fixed':
                state.update(current_query_loss=float(result['current_loss'].detach()),future_query_loss=float(result['future_loss'].detach()),
                             mahalanobis_per_channel=float(result['update']['predictive_mahalanobis'].detach().mean()/240))
            atomic_json(a.out/'status.json',state);print(json.dumps(state),flush=True)
        if a.profile:
            state['status']='profile_completed';atomic_json(a.out/'profile.json',state);atomic_json(a.out/'status.json',state);return
        if step%400==0 or step==a.steps:
            validation=evaluate(context.split['validation'][:256]);validation.update(step=step,seconds=elapsed+time.perf_counter()-t0)
            history.append(validation);improved=validation['selection_loss']<best
            if improved:best=validation['selection_loss']
            cp=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,seconds=validation['seconds'],history=history,
                    numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            temp=a.out/'last.partial.pt';torch.save(cp,temp);temp.replace(a.out/'last.pt')
            if improved:shutil.copy2(a.out/'last.pt',a.out/'best.pt')
            atomic_json(a.out/'validation_history.json',history);print(json.dumps(dict(validation=validation,best=best)),flush=True)
    cp=torch.load(a.out/'best.pt',map_location=a.device,weights_only=True);net.load_state_dict(cp['model'])
    report=dict(status='completed',mode=a.mode,seed=a.seed,best_step=cp['step'],
                assessment=evaluate_current(context.split['assessment']),one_step=evaluate(context.split['assessment'],save=True),
                seconds=elapsed+time.perf_counter()-t0,setup_seconds_current_invocation=setup_seconds,
                total_seconds_current_invocation=time.perf_counter()-started,peak_GiB=torch.cuda.max_memory_allocated(a.device)/2**30,
                mean_checkpoint_sha256=context.mean_checkpoint_hash,initialization_sha256=config['initialization_sha256'],
                covariance_training_objects=4000,official_test_opened=False)
    atomic_json(a.out/'summary.json',report);atomic_json(a.out/'status.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
