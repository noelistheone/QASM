"""Fixed-checkpoint transfer to two newly generated object morphologies.

No data from this cohort are used for fitting, calibration, or model selection.
The generator and comparator list are frozen in a protocol before execution.
"""
from pathlib import Path
import argparse, csv, json, sys, time
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import sha256
from src.scattering_observation_features import FrozenScatteringFeatures
from src.frozen_scattering_predictor import FrozenScatteringPredictor, CombinedQueryOperators
from src.independent_scattering_assessment import model_specs, masks_for_ids, observation_noise
from src.posterior_components import MaskPosteriorComponents


def generate(config):
    rng = np.random.default_rng(config['generator_seed'])
    axis = (np.arange(128) + .5) / 64 - 1
    xx, yy = np.meshgrid(axis, axis)
    objects, labels = [], []
    for family in config['families']:
        for _ in range(config['objects_per_family']):
            if family == 'inclusions':
                image = np.zeros((128, 128))
                for _ in range(rng.integers(4, 13)):
                    cx, cy = rng.uniform(-.65, .65, 2)
                    a, b = rng.uniform(.07, .28), rng.uniform(.04, .18)
                    angle = rng.uniform(0, np.pi)
                    dx, dy = xx-cx, yy-cy
                    x = np.cos(angle)*dx+np.sin(angle)*dy
                    y = -np.sin(angle)*dx+np.cos(angle)*dy
                    image = np.maximum(image, rng.uniform(.2, 1.)*((x/a)**2+(y/b)**2 < 1))
            elif family == 'perforated':
                a, b = rng.uniform(.60, .85, 2)
                angle = rng.uniform(0, np.pi)
                x, y = np.cos(angle)*xx+np.sin(angle)*yy, -np.sin(angle)*xx+np.cos(angle)*yy
                image = rng.uniform(.3, .9)*(((x/a)**2+(y/b)**2) < 1)
                for _ in range(rng.integers(5, 16)):
                    cx, cy = rng.uniform(-.65, .65, 2)
                    radius = rng.uniform(.04, .18)
                    image *= ((xx-cx)**2+(yy-cy)**2 >= radius**2)
            else:
                raise ValueError(family)
            objects.append(gaussian_filter(image, sigma=config['edge_sigma_pixels']).astype(np.float32))
            labels.append(family)
    return np.stack(objects)[:, None], np.asarray(labels)


@torch.no_grad()
def predict(args, config):
    device = torch.device(args.device)
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT/'runs/m7_morphology_v1'/f'noise{args.noise_seed}'
    out.mkdir(parents=True, exist_ok=False)
    images, labels = generate(config)
    ids = np.arange(len(images)) + config['id_offset']
    context = FrozenScatteringFeatures(ROOT, device)
    physics = CombinedQueryOperators(context)
    clean = torch.tensor(images, device=device).flatten(1) @ context.matrix.T
    observation = clean + torch.tensor(observation_noise(ids, args.noise_seed, 1.), device=device)
    features = {}
    for budget in config['budgets']:
        parts = []
        masks = masks_for_ids(ids, budget, 0)
        for start in range(0, len(ids), 32):
            part = context(observation[start:start+32], masks[start:start+32])
            parts.append({k:v.cpu() for k,v in part.items()})
        features[budget] = {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    records = []; started = time.time()
    specs = [s for s in model_specs() if s['family'] in config['families_of_models']]
    for spec in specs:
        net = FrozenScatteringPredictor(context, spec['kind'], spec['run'], spec['rank'])
        for budget in config['budgets']:
            predictions = []
            for start in range(0, len(ids), 16):
                f = {k:v[start:start+16].to(device) for k,v in features[budget].items()}
                predictions.append(net(f,budget).cpu().numpy())
            path = out/f'{spec["name"]}_b{budget}.npz'
            np.savez_compressed(path, prediction=np.concatenate(predictions))
            records.append(dict(family=spec['family'],seed=spec['seed'],budget=budget,
                file=str(path.relative_to(ROOT)),sha256=sha256(path)))
        print(spec['name'], round(time.time()-started,2), flush=True)
        del net
    for mode in ['sample','fixed']:
        for seed in config['nppc_seeds']:
            path = ROOT/f'runs/m2_nppc_20261101/{mode}_seed{seed}/best.pt'
            net = MaskPosteriorComponents(8).to(device)
            net.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model'])
            net.eval()
            for budget in config['budgets']:
                predictions = []
                for start in range(0,len(ids),16):
                    f = {k:v[start:start+16].to(device) for k,v in features[budget].items()}
                    factors = net(f['feature'],f['mean'],f['masks'])[2]
                    from src.controlled_query_learning import current_response
                    predictions.append(current_response(f['mean'],factors,physics).cpu().numpy())
                dest = out/f'nppc_{mode}_{seed}_b{budget}.npz'
                np.savez_compressed(dest,prediction=np.concatenate(predictions))
                records.append(dict(family=f'nppc_{mode}',seed=seed,budget=budget,
                    file=str(dest.relative_to(ROOT)),sha256=sha256(dest)))
            print(f'nppc_{mode}_{seed}',round(time.time()-started,2),flush=True)
            del net
    # Generate scoring responses only after all frozen predictions are saved.
    targets = []
    for start in range(0,len(ids),32):
        targets.append(physics(torch.tensor(images[start:start+32],device=device)).cpu().numpy())
    np.savez_compressed(out/'scoring.npz',targets=np.concatenate(targets),images=images,labels=labels,ids=ids)
    (out/'report.json').write_text(json.dumps(dict(status='completed',records=records,
        protocol_sha256=sha256(ROOT/'protocols/m7_morphology_v1.json'),
        scoring_sha256=sha256(out/'scoring.npz'),seconds=time.time()-started),indent=2)+'\n')


def score(config):
    out = ROOT/'analysis/m7_morphology_v1';out.mkdir(parents=True,exist_ok=False)
    arrays = {}; scoring = None
    for noise in config['noise_seeds']:
        base = ROOT/'runs/m7_morphology_v1'/f'noise{noise}'
        report = json.loads((base/'report.json').read_text())
        assert report['status']=='completed'
        with np.load(base/'scoring.npz',allow_pickle=False) as z:
            if scoring is not None: assert np.array_equal(scoring['targets'],z['targets'])
            scoring = {k:z[k].copy() for k in z.files}
        for r in report['records']:
            assert sha256(ROOT/r['file'])==r['sha256']
            pred = np.load(ROOT/r['file'],allow_pickle=False)['prediction']
            arrays.setdefault((r['family'],r['budget']),[]).append(np.abs(pred/scoring['targets']-1))
    rows = [];comparisons = []; rng = np.random.default_rng(2026110209)
    for cohort in config['families']+['combined']:
        take = np.ones(len(scoring['ids']),bool) if cohort=='combined' else scoring['labels']==cohort
        n = int(take.sum()); draws = rng.integers(0,n,(2000,n))
        for (family,budget),values in arrays.items():
            for query,sl in dict(original=slice(0,15),new=slice(15,35),interpolation=slice(15,25),extrapolation=slice(25,35)).items():
                obj = np.mean(values,axis=0)[take,sl].mean(1)
                ci = np.quantile(obj[draws].mean(1),[.025,.975])
                rows.append(dict(cohort=cohort,family=family,budget=budget,queries=query,objects=n,
                    mre=float(obj.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
                ref = np.mean(arrays['primary_moment8',budget],axis=0)[take,sl].mean(1)
                delta = ref-obj; interval=np.quantile(delta[draws].mean(1),[.025,.975])
                comparisons.append(dict(cohort=cohort,reference=family,budget=budget,queries=query,
                    ours_minus_reference=float(delta.mean()),ci_low=float(interval[0]),ci_high=float(interval[1])))
    for name,records in [('groups',rows),('comparisons',comparisons)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez_compressed(out/'objects_and_targets.npz',**scoring)
    (out/'summary.json').write_text(json.dumps(dict(status='completed',groups=rows,comparisons=comparisons,
        protocol_sha256=sha256(ROOT/'protocols/m7_morphology_v1.json')),indent=2)+'\n')
    for r in rows:
        if r['queries']=='new': print(r['cohort'],r['family'],r['budget'],round(r['mre']*100,3))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0')
    p.add_argument('--noise-seed',type=int);p.add_argument('--score',action='store_true');args=p.parse_args()
    config=json.loads((ROOT/'protocols/m7_morphology_v1.json').read_text())
    assert sha256(Path(__file__))==config['script_sha256']
    for path,value in config['checkpoint_hashes'].items(): assert sha256(ROOT/path)==value
    if args.score: score(config)
    else:
        assert args.noise_seed in config['noise_seeds']
        predict(args,config)
