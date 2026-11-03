"""Frozen development study of complex updates with/without query bounds."""
from pathlib import Path
import argparse
import collections
import json
import os
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256
from src.real_response_data import save_json,key
from src.query_response_trust import QueryResponseBound
from src.query_aware_refinement import OrthogonalUpdateNet,query_aware_refine
from scripts.run_ptycho_internal_pilot import load_case,verify_execution


def verify_new():
    verify_execution()
    path=ROOT/'protocols/m4_query_aware_execution_v1.json';d=json.loads(path.read_text())
    for group in ['code_hashes','input_hashes']:
        for file,digest in d[group].items():
            if sha256(ROOT/file)!=digest:raise RuntimeError(f'Changed new execution dependency:{file}')
    return json.loads((ROOT/'protocols/m4_query_aware_pilot_v1.json').read_text()),sha256(path)


def apply_model(case,problem,ids,model,bounded,cfg,differentiable):
    bound=QueryResponseBound(case.operator,ids,batch=cfg['query_batch']) if bounded else None
    return query_aware_refine(case.object[None],problem,case.fraction,model,bound,
        cfg['radius'] if bounded else None,cfg['steps'],differentiable)


@torch.no_grad()
def score(case,columns,batch):
    error=target_energy=drift=initial_energy=0.
    for start in range(0,len(case.label_ids),batch):
        ix=case.label_ids[start:start+batch];ids=torch.as_tensor(ix,device=case.device)
        prediction=case.operator.fields(columns,ids).abs().square().sum(1).double()
        initial=case.operator.fields(case.object[None],ids).abs().square().sum(1).double()
        target=case.targets(ix).double()
        error+=float((prediction-target).square().sum());target_energy+=float(target.square().sum())
        drift+=float((prediction-initial).square().sum());initial_energy+=float(initial.square().sum())
    return dict(relative_intensity_rmse=float(np.sqrt(error/target_energy)),
        relative_prediction_drift=float(np.sqrt(drift/initial_energy)),squared_error=error,
        target_squared_norm=target_energy,label_count=len(case.label_ids))


@torch.no_grad()
def evaluate(model,cases,bounded,cfg):
    if model is not None:model.eval()
    rows=[]
    for case,problem in cases:
        columns,trace=apply_model(case,problem,case.label_ids,model,bounded,cfg,False)
        metrics=score(case,columns,cfg['query_batch'])
        loss=float(problem.loss(columns));expected=trace[-1]['visible_loss_after_polynomial']
        assert abs(loss-expected)<5e-7+2e-5*abs(expected)
        if bounded:assert metrics['relative_prediction_drift']<=(1+cfg['radius'])**cfg['steps']-1+1e-4
        rows.append(dict(scan=case.scan,fraction=case.fraction,**metrics,visible_half_mse=loss,trace=trace))
    if model is not None:model.train()
    return dict(mean_relative_intensity_rmse=float(np.mean([r['relative_intensity_rmse'] for r in rows])),rows=rows)


def update(model,optimizer,case,problem,ids,bounded,cfg):
    optimizer.zero_grad(set_to_none=True)
    columns,trace=apply_model(case,problem,ids,model,bounded,cfg,True)
    prediction=case.operator.fields(columns,torch.as_tensor(ids,device=case.device)).abs().square().sum(1)
    loss=(prediction-case.targets(ids)).square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),trace=trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['train','controls'],required=True)
    parser.add_argument('--kind',choices=['complex_unbounded','complex_bounded'])
    parser.add_argument('--seed',type=int);parser.add_argument('--device',required=True)
    args=parser.parse_args();contract,digest=verify_new();cfg=contract['method'];opt=contract['optimization']
    name='uniform_controls' if args.mode=='controls' else f'{args.kind}_{args.seed}'
    out=ROOT/'runs/m4_query_aware_pilot_v1'/name;out.mkdir(parents=True,exist_ok=False)
    base=dict(pid=os.getpid(),host=os.uname().nodename,device=args.device,started=time.time(),
        freeze_sha256=digest,material_stress_labels_loaded=False,reference_pixels_loaded=False)
    save_json(out/'status.json',dict(base,status='loading'))
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.cuda.set_device(torch.device(args.device))
        torch.manual_seed(args.seed or opt['seeds'][0]);np.random.seed(args.seed or opt['seeds'][0])
        torch.cuda.reset_peak_memory_stats(args.device)
        training={(s,f):load_case(s,f,args.device,'train',cfg['visible_batch'])
            for s in contract['roles']['train'] for f in contract['fractions']}
        validation=[load_case(s,f,args.device,'validation',cfg['visible_batch'])
            for s in contract['roles']['validation'] for f in contract['fractions']]
        if args.mode=='controls':
            results={}
            for bounded in [False,True]:
                tag='bounded' if bounded else 'unbounded'
                results[tag]=dict(training=evaluate(None,list(training.values()),bounded,cfg),
                    validation=evaluate(None,validation,bounded,cfg))
                save_json(out/f'{tag}.json',results[tag]);print(json.dumps(dict(control=tag,
                    validation=results[tag]['validation']['mean_relative_intensity_rmse'])),flush=True)
            record=dict(base,status='completed',conditions=15,variants=2,rows=30,
                files={f'{tag}.json':sha256(out/f'{tag}.json') for tag in results})
            save_json(out/'summary.json',record);save_json(out/'status.json',record);return
        if args.kind not in contract['models'] or args.seed not in opt['seeds']:
            raise ValueError('Undeclared model or seed')
        bounded=args.kind=='complex_bounded'
        model=OrthogonalUpdateNet().to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
        rng=np.random.default_rng(args.seed)
        save_json(out/'config.json',dict(base,kind=args.kind,seed=args.seed,torch=torch.__version__,
            parameters=sum(p.numel() for p in model.parameters()),optimization=opt,method=cfg))
        curves=[];sampling=[];checkpoints={};logs=[];best=float('inf');best_step=None;cumulative=0.
        branches=collections.Counter();bound_active_steps=0
        for step in range(opt['steps']+1):
            if step:
                scan=str(rng.choice(['W','IC1','IC2'],p=[.5,.25,.25]));fraction=float(rng.choice(contract['fractions']))
                case,problem=training[(scan,fraction)]
                ids=rng.choice(case.label_ids,size=cfg['query_batch'],replace=False)
                tick=time.perf_counter();row=update(model,optimizer,case,problem,ids,bounded,cfg)
                torch.cuda.synchronize(args.device);cumulative+=time.perf_counter()-tick
                branches.update(r['certificate']['branch'] for r in row['trace'])
                bound_active_steps+=sum(r['query_bound'] is not None and r['query_bound']['cap']<1
                    and r['step']>=1-1e-6 for r in row['trace'])
                sampling.append(dict(step=step,scan=scan,fraction=fraction,ids=ids.tolist(),**row))
                if step%25==0:
                    logs.append(dict(step=step,mean_loss=float(np.mean([x['loss'] for x in sampling[-25:]])),
                        training_seconds=cumulative,branches=dict(branches),bound_active_steps=bound_active_steps))
                    save_json(out/'training_curve.json',logs)
                    save_json(out/'status.json',dict(base,status='training',step=step,updated=time.time(),
                        best_step=best_step,best_validation=best,training_seconds=cumulative,
                        peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30))
            if step%opt['checkpoints_every']==0:
                metrics=evaluate(model,validation,bounded,cfg);curves.append(dict(step=step,**metrics))
                path=out/f'step{step}.pt'
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,kind=args.kind,
                    seed=args.seed,validation=metrics,numpy_generator=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(args.device),freeze_sha256=digest),path)
                checkpoints[path.name]=sha256(path)
                if metrics['mean_relative_intensity_rmse']<best:best=metrics['mean_relative_intensity_rmse'];best_step=step
                save_json(out/'validation_curve.json',curves);save_json(out/'sampling_and_updates.json',sampling)
                print(json.dumps(dict(kind=args.kind,seed=args.seed,step=step,validation=metrics['mean_relative_intensity_rmse'],
                    best_step=best_step,training_seconds=cumulative,bound_active_steps=bound_active_steps)),flush=True)
        selected=torch.load(out/f'step{best_step}.pt',map_location=args.device,weights_only=False)
        model.load_state_dict(selected['model'])
        save_json(out/'selected_training_endpoints.json',evaluate(model,list(training.values()),bounded,cfg))
        result=dict(base,status='completed',kind=args.kind,seed=args.seed,selected_step=best_step,
            selected_validation_relative_rmse=best,selected_checkpoint=f'step{best_step}.pt',checkpoint_hashes=checkpoints,
            steps=opt['steps'],validation_checkpoints=len(curves),training_seconds=cumulative,
            branches=dict(branches),bound_active_steps=bound_active_steps,
            peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30)
        save_json(out/'summary.json',result);save_json(out/'status.json',result)
    except Exception as error:
        save_json(out/'failure.json',dict(base,error=repr(error),time=time.time()))
        save_json(out/'status.json',dict(base,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
