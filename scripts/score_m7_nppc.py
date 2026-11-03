"""Compare the frozen NPPC variants with the identical response-state cases."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256


def main():
    out=ROOT/'analysis/m7_nppc_v1';out.mkdir(parents=True,exist_ok=False)
    contract=json.loads((ROOT/'protocols/m7_focused_mechanism_v1.json').read_text())
    records=[]
    for noise in contract['born']['noise_seeds']:
        report=json.loads((ROOT/f'runs/m7_nppc_v1/noise{noise}/report.json').read_text())
        assert report['status']=='completed' and len(report['predictions'])==120
        for rec in report['predictions']:
            assert sha256(ROOT/rec['file'])==rec['sha256'];records.append(dict(noise=noise,**rec))
    targets=np.load(ROOT/'data/official_scattering_v1/scoring/targets35.npy',allow_pickle=False)
    arrays={}
    for rec in records:
        key=(f'nppc_{rec["mode"]}_{rec["variant"]}',rec['budget'])
        prediction=np.load(ROOT/rec['file'],allow_pickle=False)['prediction']
        arrays.setdefault(key,[]).append(np.abs(prediction/targets-1))
    for budget in [0,2,5,10]:
        values=[]
        for noise in contract['born']['noise_seeds']:
            for seed in [20261126,20261127,20261128]:
                path=ROOT/f'runs/m3_official_assessment_v1/noise{noise}_x1/primary_moment8_seed{seed}/mask0_budget{budget}.npz'
                pred=np.load(path,allow_pickle=False)['prediction'];values.append(np.abs(pred/targets-1))
        arrays['response_state',budget]=values
    rng=np.random.default_rng(2026110207);bootstrap=rng.integers(0,100,(2000,100))
    groups=dict(original=slice(0,15),new=slice(15,35),interpolation=slice(15,25),extrapolation=slice(25,35))
    rows=[];contrasts=[];object_errors={}
    for (family,budget),values in arrays.items():
        mean_errors=np.mean(np.stack(values),axis=0)
        for name,sl in groups.items():
            obj=mean_errors[:,sl].mean(1);ci=np.quantile(obj[bootstrap].mean(1),[.025,.975])
            rows.append(dict(family=family,budget=budget,queries=name,mre=float(obj.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
            object_errors[f'{family}_b{budget}_{name}']=obj
            ref=np.mean(np.stack(arrays['response_state',budget]),axis=0)[:,sl].mean(1)
            delta=ref-obj;interval=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            contrasts.append(dict(reference=family,budget=budget,queries=name,ours_minus_reference=float(delta.mean()),ci_low=float(interval[0]),ci_high=float(interval[1])))
    for name,values in [('groups',rows),('comparisons',contrasts)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    np.savez_compressed(out/'object_errors.npz',**object_errors)
    summary=dict(status='completed',objects=100,training_seeds=3,noise_draws=3,predictions=360,
                 groups=rows,comparisons=contrasts,protocol_sha256=sha256(ROOT/'protocols/m7_focused_mechanism_v1.json'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for family in sorted({r['family'] for r in rows}):
        print(family,[round(r['mre']*100,4) for r in rows if r['family']==family and r['queries']=='new'])


if __name__=='__main__':main()
