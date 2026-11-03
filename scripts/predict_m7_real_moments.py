"""Transfer the already trained PSD multi-field models to broadband counts."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256
from src.global_ptycho_response import GlobalProbeOperator,GlobalResponseNet,observation_features


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);a=p.parse_args()
    contract=json.loads((ROOT/'protocols/m7_focused_mechanism_v1.json').read_text())
    original=json.loads((ROOT/'protocols/m4_broadband_transfer_v1.json').read_text())
    torch.set_num_threads(4);torch.cuda.set_device(torch.device(a.device))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'runs/m7_real_moments_v1';out.mkdir(parents=True,exist_ok=False)
    models=[]
    for name,info in contract['models'].items():
        if info['group']!='real':continue
        path=ROOT/name;assert sha256(path)==info['sha256']
        checkpoint=torch.load(path,map_location=a.device,weights_only=False)
        net=GlobalResponseNet(info['kind']).to(a.device)
        net.load_state_dict(checkpoint['model']);net.eval()
        models.append((info['kind'],info['seed'],net))
    assert len(models)==9
    records=[];started=time.time()
    for energy in [8,12]:
        data=np.load(ROOT/f'data/broadband_transfer_inputs_v1/energy{energy}.npz',allow_pickle=False)
        for count in [60,150,300]:
            ids=np.asarray(original['visible_ids'][str(count)]);query=np.asarray(original['query_ids'])
            lookup={int(v):i for i,v in enumerate(data['visible_union_ids'])}
            measured=data['counts'][[lookup[int(v)] for v in ids]]
            alpha=float(1/np.sqrt(np.mean(measured.astype(np.float64)**2)))
            amplitude=float(np.sqrt(measured.sum((1,2)).mean()/np.abs(data['probe'].astype(np.complex128)).__pow__(2).sum()))
            centers=2*np.rint(data['positions_xy'])-data['positions_xy']
            for probe in ['fixed','free']:
                phase='m4_broadband_extended_v1' if probe=='fixed' else 'm4_broadband_gauge_v1'
                path=ROOT/'runs'/phase/f'e{energy}_n{count}_{probe}/physical_state_1000.npz'
                assert sha256(path)==contract['states'][str(path.relative_to(ROOT))]
                state=np.load(path,allow_pickle=False)
                obj=torch.tensor(state['object'],device=a.device,dtype=torch.complex128)/amplitude
                pr=torch.tensor(state['probe'],device=a.device,dtype=torch.complex128)*(amplitude*np.sqrt(alpha))
                visible_op=GlobalProbeOperator(pr,centers[ids],obj.shape)
                feature,_,_=observation_features(obj,visible_op,
                    torch.tensor(measured.astype(np.float64)*alpha,device=a.device),count/600,batch=16)
                obj=obj.to(torch.complex64);feature=feature.float()
                op=GlobalProbeOperator(pr.to(torch.complex64),centers,obj.shape)
                for kind,seed,net in models:
                    tick=time.time();columns=net(feature,obj);parts=[];coherent=[]
                    for start in range(0,len(query),16):
                        index=torch.as_tensor(query[start:start+16],device=a.device)
                        field=op.fields(columns,index)
                        parts.append((net.readout(field)/alpha).cpu().numpy())
                        if kind=='moment':coherent.append((field[:,0].abs().square()/alpha).cpu().numpy())
                    variants={kind:np.concatenate(parts)}
                    if kind=='moment':variants['moment_no_factors']=np.concatenate(coherent)
                    for variant,pred in variants.items():
                        assert pred.shape==(120,256,256) and np.isfinite(pred).all() and (pred>=0).all()
                        dest=out/f'e{energy}_n{count}_{probe}_{variant}_{seed}.npz'
                        np.savez_compressed(dest,prediction=pred)
                        records.append(dict(energy=energy,count=count,probe=probe,kind=variant,seed=seed,
                            file=str(dest.relative_to(ROOT)),sha256=sha256(dest),seconds=time.time()-tick))
                print(json.dumps(dict(energy=energy,count=count,probe=probe,seconds=time.time()-started)),flush=True)
    assert len(records)==144
    (out/'report.json').write_text(json.dumps(dict(status='completed',predictions=records,
        seconds=time.time()-started,scoring_targets_loaded=False),indent=2)+'\n')


if __name__=='__main__':main()
