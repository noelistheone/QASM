"""Replay the paper's two learned readouts from archived observation-only inputs.

The fixture contains one nominal-noise/scattering seed across all 100 test
objects and four budgets, and one broadband case with all 120 query positions.
Reference arrays are opened only after the predictions have been computed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FIXTURE=ROOT/'reproduction/fixtures'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


@torch.no_grad()
def born(device,output_dir):
    from src.born_scattering import BornConfig,build_matrix
    from src.posterior_components import MaskPosteriorComponents
    from src.controlled_query_learning import current_response
    from src.response_learning import angular_weights
    network=MaskPosteriorComponents(8).to(device)
    network.load_state_dict(torch.load(FIXTURE/'born_weights.pt',map_location=device,weights_only=True));network.eval()
    matrices=[]
    for wave in [4,6,8,5,7,10,12]:
        matrices.append(torch.tensor(build_matrix(BornConfig(wave=float(wave),numRec=60)),device=device))
    physical=SimpleNamespace(matrices=matrices,weights=angular_weights(device))
    outputs={}
    for budget in [0,2,5,10]:
        values=np.load(FIXTURE/f'born_inputs_budget{budget}.npz',allow_pickle=False)
        parts=[]
        for start in range(0,len(values['masks']),16):
            feature,mean,mask=[torch.tensor(values[k][start:start+16],device=device) for k in ['feature','mean','masks']]
            factors=network(feature,mean,mask)[2]
            if budget==0 and start==0:
                np.savez_compressed(output_dir/'born_state_example.npz',mean=mean[0,0].cpu().numpy(),
                                    factors=factors[0].reshape(8,128,128).cpu().numpy())
            parts.append(current_response(mean,factors,physical).cpu().numpy())
        outputs[budget]=np.concatenate(parts)
    expected=np.load(FIXTURE/'born_scoring.npz',allow_pickle=False)
    results=[]
    for budget,prediction in outputs.items():
        original=expected[f'prediction_budget{budget}'];target=expected['targets35']
        relative=float(np.linalg.norm(prediction-original)/np.linalg.norm(original))
        assert relative<2e-5,(budget,relative)
        results.append(dict(budget=budget,objects=len(prediction),queries=35,relative_replay_difference=relative,
                            new_query_mre=float(np.abs(prediction[:,15:]/target[:,15:]-1).mean())))
    return dict(case='born',results=results,scope='One declared trained seed and one observation-noise realization; figure tables average the full seed/noise set.')


@torch.no_grad()
def broadband(device,output_dir):
    from src.global_ptycho_response import GlobalProbeOperator
    from src.ptycho_internal_refinement import VisibleIntensityProblem
    from src.query_aware_refinement import OrthogonalUpdateNet,query_aware_refine
    data=np.load(FIXTURE/'broadband_inputs.npz',allow_pickle=False)
    counts=data['visible_counts'];ids=data['visible_ids'];query=data['query_ids'];xy=data['positions_xy']
    centers=2*np.rint(xy)-xy
    amplitude=float(data['initial_amplitude']);alpha=float(1/np.sqrt(np.mean(counts.astype(np.float64)**2)))
    obj=torch.tensor(data['object'],device=device)/amplitude
    probe=torch.tensor(data['probe'],device=device)*(amplitude*np.sqrt(alpha))
    operator=GlobalProbeOperator(probe,centers,obj.shape)
    problem=VisibleIntensityProblem(operator,ids,torch.tensor(counts.astype(np.float32)*alpha,device=device),batch=16)
    model=OrthogonalUpdateNet().to(device)
    model.load_state_dict(torch.load(FIXTURE/'broadband_weights.pt',map_location=device,weights_only=True));model.eval()
    columns,trace=query_aware_refine(obj[None],problem,len(ids)/600,model=model,steps=3,differentiable=False)
    parts=[]
    for start in range(0,len(query),16):
        ix=torch.as_tensor(query[start:start+16],device=device)
        parts.append((operator.fields(columns,ix).abs().square().sum(1)/alpha).cpu().numpy())
    prediction=np.concatenate(parts)
    expected=np.load(FIXTURE/'broadband_scoring.npz',allow_pickle=False)
    original=expected['learned_prediction'];relative=float(np.linalg.norm(prediction-original)/np.linalg.norm(original))
    assert relative<3e-5,relative
    target=expected['query_counts'].astype(np.float64)
    np.savez_compressed(output_dir/'broadband_query_example.npz',query_id=query[0],
                        observed=target[0],native=expected['native_prediction'][0],learned=prediction[0])
    rmse=lambda x:float(np.linalg.norm(x.astype(np.float64)-target)/np.linalg.norm(target))
    return dict(case='broadband',energy=8,visible=300,queries=120,relative_replay_difference=relative,
                native_rmse=rmse(expected['native_prediction']),learned_rmse=rmse(prediction),trace=trace)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['born','broadband','all'],default='all')
    p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((FIXTURE/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert digest(FIXTURE/name)==expected,name
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.device.startswith('cuda'):torch.cuda.set_device(torch.device(a.device))
    result=dict(device=a.device,torch=torch.__version__,started=time.time(),cases=[])
    a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists()
    for name,function in [('born',born),('broadband',broadband)]:
        if a.case in [name,'all']:result['cases'].append(function(a.device,a.output.parent))
    result.update(status='passed',seconds=time.time()-result['started'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
