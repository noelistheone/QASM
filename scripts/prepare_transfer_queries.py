"""Reference labels for frozen, untrained physical queries; never training inputs."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import BornConfig,BatchedBorn,sha256
from src.response_learning import PhysicalQueries


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--batch',type=int,default=256)
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'data/m2_transfer_queries_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Already prepared')
    images=np.load(ROOT/'data/m2_response_pilot_v1/images.npy',mmap_mode='r')
    split=json.loads((ROOT/'data/m2_mask_features_v1/split.json').read_text())
    matrices=[BatchedBorn(BornConfig(wave=w,numRec=60),cache_dir=ROOT/'data/born_cache',device=a.device).matrix for w in [5,7,10,12]]
    physical=PhysicalQueries(matrices);started=time.perf_counter();parts=[]
    with torch.no_grad():
        for start in range(0,len(images),a.batch):
            x=torch.tensor(np.asarray(images[start:start+a.batch].copy()),device=a.device)
            parts.append(physical(x).cpu().numpy())
    query=np.concatenate(parts);assert query.shape==(10000,20) and np.isfinite(query).all() and (query>0).all()
    np.save(out/'queries.npy',query)
    report=dict(status='completed',waves=[5,7,10,12],objects=len(images),seconds=time.perf_counter()-started,
                query_scale_reporting_only=np.sqrt(np.mean(query[split['mean_train']].astype(np.float64)**2,axis=0)).tolist(),
                sha256=sha256(out/'queries.npy'),protocol_sha256=sha256(ROOT/'protocols/m2_query_transfer_v1.json'),
                source_sha256=sha256(Path(__file__)),used_for_fitting=False,official_test_opened=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
