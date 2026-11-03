"""Replay the nearest image-space moment comparison from compact fixtures."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from src.born_scattering import BornConfig,build_matrix
    from src.response_learning import angular_weights
    from src.posterior_components import MaskPosteriorComponents
    from src.controlled_query_learning import current_response
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    base=ROOT/'reproduction/fixtures';extra=ROOT/'reproduction/m7_fixtures'
    for folder in [base,extra]:
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    started=time.time();rows=[]
    matrices=[torch.tensor(build_matrix(BornConfig(wave=float(w),numRec=60)),device=args.device)
              for w in [4,6,8,5,7,10,12]]
    physical=SimpleNamespace(matrices=matrices,weights=angular_weights(args.device))
    for objective in ['sample','fixed']:
        net=MaskPosteriorComponents(8).to(args.device)
        net.load_state_dict(torch.load(extra/f'nppc_{objective}_weights.pt',map_location=args.device,weights_only=True))
        net.eval();outputs={}
        for budget in [0,2,5,10]:
            values=np.load(base/f'born_inputs_budget{budget}.npz');parts=[]
            for start in range(0,len(values['masks']),16):
                f,m,mask=[torch.tensor(values[k][start:start+16],device=args.device) for k in ['feature','mean','masks']]
                factors=net(f,m,mask)[2]
                parts.append(current_response(m,factors,physical).cpu().numpy())
            outputs[budget]=np.concatenate(parts)
        scoring=np.load(extra/f'nppc_{objective}_scoring.npz')
        targets=np.load(base/'born_scoring.npz')['targets35']
        for budget,prediction in outputs.items():
            expected=scoring[f'prediction_budget{budget}']
            relative=float(np.linalg.norm(prediction-expected)/np.linalg.norm(expected))
            assert relative<2e-5,(objective,budget,relative)
            rows.append(dict(objective=objective,budget=budget,relative_replay_difference=relative,
                new_query_mre=float(np.abs(prediction[:,15:]/targets[:,15:]-1).mean())))
    assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='passed',results=rows,seconds=time.time()-started,
        scope='One seed/noise fixture; manuscript tables average all three seeds and three noise draws.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
