"""Score every declared real-moment prediction and factor-removal control."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import sha256


def pool(x,k):
    n,h,w=x.shape
    return x.reshape(n,h//k,k,w//k,k).sum((2,4))


def main():
    folder=ROOT/'runs/m7_real_moments_v1'
    report=json.loads((folder/'report.json').read_text());assert report['status']=='completed' and len(report['predictions'])==144
    for rec in report['predictions']:assert sha256(ROOT/rec['file'])==rec['sha256']
    targets=np.load(ROOT/'data/m7_real_query_counts.npz',allow_pickle=False)
    rows=[]
    for rec in report['predictions']:
        pred=np.load(ROOT/rec['file'],allow_pickle=False)['prediction'].astype(np.float64)
        truth=targets[f'energy{rec["energy"]}'].astype(np.float64)
        for k in [1,4,16]:
            pr=pool(pred,k);tr=pool(truth,k)
            rows.append(dict(**{x:rec[x] for x in ['energy','count','probe','kind','seed']},pool=k,
                             rmse=float(np.linalg.norm(pr-tr)/np.linalg.norm(tr)),fluence_ratio=float(pr.sum()/tr.sum())))
    out=ROOT/'analysis/m7_real_moments_v1';out.mkdir(parents=True,exist_ok=False)
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[]
    for probe in ['fixed','free']:
        for kind in ['coherent','moment','moment_no_factors','direct']:
            for k in [1,4,16]:
                vals=[r['rmse'] for r in rows if r['probe']==probe and r['kind']==kind and r['pool']==k]
                groups.append(dict(probe=probe,kind=kind,pool=k,rmse=float(np.mean(vals)),seed_condition_results=18))
    (out/'summary.json').write_text(json.dumps(dict(status='completed',predictions=144,groups=groups,rows=rows),indent=2)+'\n')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
