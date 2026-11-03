"""Fixed longer-iteration, probe, smoothing, and photon-gain controls.

This is a declared follow-up on the already-scored view 1. Predictors read only
the original visible input cache; target counts are unavailable to this module.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.broadband_calibration_solver_v2 import CalibrationSolver, computational_centers
from src.global_ptycho_response import GlobalProbeOperator
from src.internal_crosssource import load_model
from src.ptycho_internal_refinement import VisibleIntensityProblem
from src.query_aware_refinement import query_aware_refine
from src.query_response_trust import QueryResponseBound
from src.fixed_smooth_direction import FixedSmoothDirection


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path, value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def readout(obj, operator, ids, scale=1.):
    arrays=[]
    for start in range(0,len(ids),16):
        ix=torch.as_tensor(ids[start:start+16],device=operator.probe.device)
        arrays.append((operator.fields(obj[None],ix).abs().square().sum(1)/scale).cpu().numpy())
    result=np.concatenate(arrays).astype(np.float32)
    assert np.isfinite(result).all() and (result>=0).all()
    return result


def export(out, name, array):
    path=out/(name+'.npy');assert not path.exists()
    np.save(path,np.asarray(array,dtype=np.float32),allow_pickle=False)
    return dict(method=name,path=str(path.relative_to(ROOT)),sha256=digest(path))


def main():
    p=argparse.ArgumentParser();p.add_argument('--energy',type=int,choices=[8,12],required=True)
    p.add_argument('--count',type=int,choices=[60,150,300],required=True)
    p.add_argument('--probe',choices=['fixed','free'],required=True)
    p.add_argument('--device',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(torch.device(args.device));torch.cuda.reset_peak_memory_stats(args.device)
    contract_path=ROOT/'protocols/m4_broadband_extended_v1.json'
    contract=json.loads(contract_path.read_text())
    for path,h in contract['code_and_input_hashes'].items():assert digest(ROOT/path)==h,path
    original=json.loads((ROOT/'protocols/m4_broadband_transfer_v1.json').read_text())
    key=f'e{args.energy}_n{args.count}_{args.probe}'+('_smoke' if args.smoke else '')
    out=ROOT/'runs/m4_broadband_extended_v1'/key;out.mkdir(parents=True,exist_ok=False)
    base=dict(pid=os.getpid(),host=os.uname().nodename,device=args.device,key=key,
              protocol_sha256=digest(contract_path),query_values_read=False,neural_training=False)
    save(out/'status.json',dict(base,status='initializing'))
    try:
        data=np.load(ROOT/f'data/broadband_transfer_inputs_v1/energy{args.energy}.npz',allow_pickle=False)
        ids=np.asarray(original['visible_ids'][str(args.count)]);query=np.asarray(original['query_ids'])
        lookup={int(v):i for i,v in enumerate(data['visible_union_ids'])}
        measured=data['counts'][[lookup[int(v)] for v in ids]].copy()
        xy=data['positions_xy'];probe=data['probe'];shape=tuple(data['object_shape'])
        amplitude=float(np.sqrt(measured.sum((1,2)).mean()/np.abs(probe.astype(np.complex128)).__pow__(2).sum()))
        initial=np.full(shape,amplitude,dtype=np.complex64)
        solver=CalibrationSolver(measured,xy[ids],initial,probe,'poisson',args.probe=='free',
                                 args.device,seed=contract['seed'])
        assert solver.parity()<3e-5
        checkpoints=[2] if args.smoke else contract['native_checkpoints']
        refine_at=[2] if args.smoke else contract['refinement_checkpoints']
        records=[];predictions=[];refinements=[];native_seconds=0.;started=time.time()
        for epoch in checkpoints:
            torch.cuda.synchronize();tick=time.time()
            # Match the original run boundaries before its 1000-epoch state.
            if epoch==1000:
                solver.run_to(100);solver.run_to(300)
            solver.run_to(epoch);torch.cuda.synchronize()
            native_seconds+=time.time()-tick
            obj=solver.object().detach().clone();pr=solver.probe().detach().clone()
            op=GlobalProbeOperator(pr,computational_centers(xy),shape)
            query_pred=readout(obj,op,query)
            predictions.append(export(out,f'native_{epoch}',query_pred))
            vis=readout(obj,op,ids)
            record=dict(epoch=epoch,native_seconds=native_seconds,wall_seconds=time.time()-started,
                        visible_rmse=float(np.linalg.norm(vis.astype(np.float64)-measured)/np.linalg.norm(measured.astype(np.float64))))
            if epoch==1000:
                oldpath=ROOT/f'runs/m4_broadband_transfer_v1/e{args.energy}_n{args.count}_{args.probe}_poisson/physical_state.npz'
                old=np.load(oldpath,allow_pickle=False)
                record['old_state_relative_difference']=float(np.linalg.norm(obj.cpu().numpy()-old['object'])/np.linalg.norm(old['object']))
                # Different devices/torch builds can have small roundoff differences.
            records.append(record);save(out/f'checkpoint_{epoch}.json',record)
            save(out/'status.json',dict(base,status='physical_fit',epoch=epoch,native_seconds=native_seconds))
            if epoch not in refine_at:continue
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();numpy_rng=np.random.get_state()
            np.savez_compressed(out/f'physical_state_{epoch}.npz',object=obj.cpu().numpy(),probe=pr.cpu().numpy())
            # Two one-scalar controls use visible frames only.
            gains={'poisson_gain':float(measured.sum()/vis.sum(dtype=np.float64)),
                   'l2_gain':float(np.sum(vis.astype(np.float64)*measured)/np.sum(vis.astype(np.float64)**2))}
            for method,gain in gains.items():
                assert gain>0
                predictions.append(export(out,f'{method}_{epoch}',query_pred*gain))
            alpha=float(1/np.sqrt(np.mean(measured.astype(np.float64)**2)))
            norm_obj=obj/amplitude;norm_probe=pr*(amplitude*np.sqrt(alpha))
            nop=GlobalProbeOperator(norm_probe,computational_centers(xy),shape)
            problem=VisibleIntensityProblem(nop,ids,torch.tensor(measured.astype(np.float32)*alpha,device=args.device),batch=16)
            specs=[dict(name='uniform_unbounded3',family='uniform_unbounded3'),
                   dict(name='uniform_bounded3',family='uniform_bounded3')]+original['models']
            specs += [dict(name=f'smooth_{sigma:g}',family='smooth',sigma=sigma) for sigma in contract['smoothing_sigmas']]
            for spec in specs:
                tick=time.time()
                if spec['family']=='smooth':model=FixedSmoothDirection(spec['sigma'])
                else:model=load_model(ROOT,spec,args.device) if 'checkpoint' in spec else None
                bounded=spec['family'] in ['uniform_bounded3','complex_bounded']
                bound=QueryResponseBound(nop,query,batch=16) if bounded else None
                with torch.no_grad():
                    columns,trace=query_aware_refine(norm_obj[None],problem,args.count/600,model=model,
                        bound=bound,radius=.1 if bounded else None,steps=3,differentiable=False)
                    torch.cuda.synchronize();algorithm_seconds=time.time()-tick
                    final_loss=float(problem.loss(columns))
                    assert abs(final_loss-trace[-1]['visible_loss_after_polynomial'])<5e-7+2e-5*abs(final_loss)
                    prediction=readout(columns[0],nop,query,alpha)
                name=f'{spec["name"]}_{epoch}';predictions.append(export(out,name,prediction))
                refinements.append(dict(method=name,algorithm_seconds=algorithm_seconds,trace=trace))
                del model,columns
            del problem,nop
            # Loading frozen networks initializes temporary modules. Their RNG
            # consumption must not change the subsequent native batch sequence.
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng);np.random.set_state(numpy_rng)
        assert solver.parity()<3e-5
        report=dict(base,status='completed',energy=args.energy,visible_count=args.count,probe=args.probe,
                    native_epochs=checkpoints[-1],native_seconds=native_seconds,seconds=time.time()-started,
                    checkpoints=records,predictions=predictions,refinements=refinements,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
                    scope='Retrospective strengthening on view 1; no new query access or fitting.')
        save(out/'report.json',report);save(out/'status.json',dict(base,status='completed',predictions=len(predictions)))
        print(json.dumps(dict(key=key,predictions=len(predictions),native_seconds=native_seconds)))
    except Exception as error:
        save(out/'status.json',dict(base,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
