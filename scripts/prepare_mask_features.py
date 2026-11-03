from pathlib import Path
import argparse,json,sys,time,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import BornConfig,BatchedBorn,sha256
from src.response_learning import ridge_measurement_map

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'data/m2_response_pilot_v1';out=ROOT/'data/m2_mask_features_v1';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Already prepared')
    t0=time.perf_counter();images=np.load(root/'images.npy',mmap_mode='r')
    op=BatchedBorn(BornConfig(numRec=180),cache_dir=ROOT/'data/born_cache',device=args.device)
    mapping,info=ridge_measurement_map(op.matrix);np.save(out/'full_ridge_map.npy',mapping.float().cpu().numpy());del mapping
    signals=np.lib.format.open_memmap(out/'clean_signals.npy',mode='w+',dtype=np.float32,shape=(10000,7200))
    for start in range(0,10000,128):
        x=torch.tensor(np.asarray(images[start:start+128].copy()),device=args.device).flatten(1)
        signals[start:start+128]=(x@op.matrix.T).cpu().numpy()
    signals.flush();split=json.loads((root/'split.json').read_text());mean_ids=split['train'][:4000]
    ids=dict(mean_train=mean_ids,covariance_train=split['train'][4000:],validation=split['validation'],assessment=split['assessment'])
    (out/'split.json').write_text(json.dumps(ids,indent=2))
    q=np.load(root/'queries.npy');scale=np.sqrt(np.mean(q[mean_ids].astype(float)**2,0));variance=float(np.var(np.asarray(images[mean_ids],dtype=float)))
    sources=[Path(__file__),ROOT/'src/response_learning.py',ROOT/'src/born_scattering.py'];snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for s in sources:shutil.copy2(s,snapshot/str(s.relative_to(ROOT)).replace('/','__'))
    report=dict(status='completed',seconds=time.perf_counter()-t0,ridge_info=info,query_scale=scale.tolist(),image_variance=variance,
        hashes={s:sha256(out/s) for s in ['full_ridge_map.npy','clean_signals.npy','split.json']},
        source_hashes={str(s.relative_to(ROOT)):sha256(s) for s in sources},
        parent_manifest_sha256=sha256(root/'manifest.json'),protocol_sha256=sha256(ROOT/'protocols/m2_mask_conditioned_learning_v1.json'))
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
