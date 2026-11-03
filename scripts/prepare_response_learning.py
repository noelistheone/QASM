"""Freeze object splits and prepare truth-free ridge features for the M2 pilot."""
from pathlib import Path
import argparse,json,os,sys,time
import lmdb
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.born_scattering import BornConfig,BatchedBorn,sha256
from src.response_learning import angular_power,angular_weights,ridge_measurement_map


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1')
    p.add_argument('--out',type=Path,default=ROOT/'data/m2_response_pilot_v1')
    p.add_argument('--batch',type=int,default=128);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'manifest.json').exists():raise RuntimeError('Completed dataset already exists; use its manifest')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol_path=ROOT/'protocols/m2_response_learning_pilot_v1.json'
    protocol=json.loads(protocol_path.read_text());t0=time.perf_counter()
    split_seed=protocol['data']['split_seed'];ids=np.random.default_rng(split_seed).permutation(10000)
    split={'train':ids[:8000].tolist(),'validation':ids[8000:9000].tolist(),'assessment':ids[9000:].tolist()}
    (args.out/'split.json').write_text(json.dumps(split,indent=2))
    files={name:np.lib.format.open_memmap(args.out/f'{name}.npy',mode='w+',dtype=np.float32,shape=shape)
           for name,shape in [('images',(10000,1,128,128)),('features',(10000,1,128,128)),
                              ('measurements',(10000,2400)),('queries',(10000,15)),('feature_queries',(10000,15))]}
    env=lmdb.open(str(ROOT/'data/inv-scatter-train'),readonly=True,lock=False,create=False)
    with env.begin() as tx:
        assert tx.stat()['entries']==10000
        for i in range(10000):files['images'][i]=np.frombuffer(tx.get(str(i).encode()),dtype=np.float32).reshape(1,128,128)
    env.close();files['images'].flush()
    op=BatchedBorn(BornConfig(numRec=60),cache_dir=ROOT/'data/born_cache',device=args.device)
    mapping64,ridge_info=ridge_measurement_map(op.matrix)
    check_y=torch.tensor(np.asarray(files['images'][:10].copy()),device=args.device).flatten(1)@op.matrix.T
    check_f=check_y.double()@mapping64
    a=op.matrix.double();rhs=check_y.double()@a
    residual=((check_f@a.T)@a+ridge_info['lambda']*check_f-rhs).norm(dim=1)/rhs.norm(dim=1)
    assert float(residual.max())<1e-8
    mapping=mapping64.float();del mapping64,a,check_f,rhs
    matrices=[]
    for wave in [4,6,8]:
        if wave==6:matrices.append(op.matrix)
        else:matrices.append(BatchedBorn(BornConfig(wave=wave,numRec=60),cache_dir=ROOT/'data/born_cache',device=args.device).matrix)
    weights=angular_weights(args.device)
    rng=torch.Generator(device=args.device).manual_seed(protocol['observation']['noise_seed'])
    # Noise seed is in the root protocol for exact reproducibility across batches.
    for start in range(0,10000,args.batch):
        stop=min(start+args.batch,10000)
        x=torch.tensor(np.asarray(files['images'][start:stop].copy()),device=args.device).flatten(1)
        y=x@op.matrix.T
        y+=torch.randn(y.shape,device=args.device,generator=rng)* (1e-4/np.sqrt(2))
        f=y@mapping
        q=torch.cat([angular_power(x@a.T,weights) for a in matrices],1)
        qf=torch.cat([angular_power(f@a.T,weights) for a in matrices],1)
        files['measurements'][start:stop]=y.cpu().numpy()
        files['features'][start:stop]=f.reshape(-1,1,128,128).cpu().numpy()
        files['queries'][start:stop]=q.cpu().numpy();files['feature_queries'][start:stop]=qf.cpu().numpy()
        if start//args.batch%5==0 or stop==10000:
            status=dict(status='running',objects=stop,total=10000,seconds=time.perf_counter()-t0,pid=os.getpid())
            (args.out/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    for v in files.values():v.flush()
    train=split['train'];q=np.asarray(files['queries'][train],dtype=np.float64)
    query_scale=np.sqrt(np.mean(q*q,axis=0))
    image_variance=float(np.var(np.asarray(files['images'][train],dtype=np.float64)))
    info=dict(status='completed',objects=10000,split_counts={k:len(v) for k,v in split.items()},
              protocol_sha256=sha256(protocol_path),split_sha256=sha256(args.out/'split.json'),
              query_scale=query_scale.tolist(),image_variance=image_variance,ridge_info=ridge_info,
              ridge_double_normal_residual_max=float(residual.max()),batch=args.batch,device=args.device,
              noise_seed=protocol['observation']['noise_seed'],
              hashes={name:sha256(args.out/f'{name}.npy') for name in files},seconds=time.perf_counter()-t0,
              source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/response_learning.py',ROOT/'src/born_scattering.py']})
    (args.out/'manifest.json').write_text(json.dumps(info,indent=2));(args.out/'status.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
