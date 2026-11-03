"""Acquire and simulate the100 official objects only after the final freeze.

This is the only acquisition/simulation process. Predictors do not import it.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import stat
import sys
import time
import urllib.request
import zipfile

import lmdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.born_scattering import BatchedBorn, BornConfig, sha256
from src.independent_scattering_assessment import (
    NOISE_SEEDS, MULTIPLIERS, atomic_json, load_verified_freeze, observation_noise)
from src.response_learning import PhysicalQueries


def read_physical_objects(directory, expected=100):
    env = lmdb.open(str(directory), readonly=True, lock=False, create=False)
    try:
        with env.begin() as transaction:
            keys = [key for key, _ in transaction.cursor()]
            if set(keys) != {str(i).encode() for i in range(expected)}:
                raise ValueError('Official object keys differ from the frozen0–99 contract')
            objects = []
            for i in range(expected):
                values = np.frombuffer(transaction.get(str(i).encode()), dtype=np.float32)
                if values.size != 128*128 or not np.isfinite(values).all():
                    raise ValueError(f'Incompatible object{i}; do not drop or repair silently')
                objects.append(values.reshape(1, 128, 128).copy())
    finally:
        env.close()
    return np.stack(objects)


def extract_archive(archive, destination):
    # No traversal, symlinks, or implicit overwrite of a previous extraction.
    destination.mkdir()
    with zipfile.ZipFile(archive) as source:
        if sum(info.file_size for info in source.infolist()) > 100_000_000:
            raise ValueError('Archive exceeds the expected small100-object dataset')
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe archive member')
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as inp, target.open('xb') as out:
                    while data := inp.read(2**20):
                        out.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path, default=ROOT/'protocols/m3_official_assessment_frozen_v1.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    freeze = load_verified_freeze(ROOT, args.freeze)
    source = freeze['protocol']['source']
    out = ROOT/'data/official_scattering_v1'
    if out.exists():
        raise RuntimeError('Inspect existing official preparation; do not overwrite')
    origin = ROOT/'sources/resources/official_assessment_v1'
    origin.mkdir(exist_ok=True)
    archive = origin/source['archive']
    started = time.time()
    # Record the first actual acquisition separately from the immutable freeze.
    receipt = origin/'acquisition.json'
    if receipt.exists():
        raise RuntimeError('Existing acquisition receipt requires inspection before recovery')
    atomic_json(receipt, dict(status='acquiring', started=started, pid=os.getpid(),
        freeze_sha256=sha256(args.freeze), source=source, official_test_opened=False))
    try:
        with urllib.request.urlopen(source['url'], timeout=60) as response, archive.open('xb') as handle:
            while chunk := response.read(2**20):
                handle.write(chunk)
        if archive.stat().st_size != source['expected_bytes']:
            raise ValueError('Official archive byte count differs from frozen metadata')
        if hashlib.md5(archive.read_bytes()).hexdigest() != source['expected_md5']:
            raise ValueError('Official archive checksum differs from frozen metadata')
        extracted = origin/'extracted'
        extract_archive(archive, extracted)
        candidates = list(extracted.rglob('data.mdb'))
        if len(candidates) != 1:
            raise ValueError('Expected exactly one LMDB database')
        objects = read_physical_objects(candidates[0].parent)
        out.mkdir()
        scoring = out/'scoring'
        scoring.mkdir()
        ids = np.arange(100, dtype=np.int64)
        np.save(scoring/'objects.npy', objects)
        np.save(out/'ids.npy', ids)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        physical = torch.tensor(objects, device=args.device)
        operator = BatchedBorn(BornConfig(numRec=180), cache_dir=ROOT/'data/born_cache', device=args.device)
        with torch.no_grad():
            signals = torch.cat([operator.forward(v, unnormalize=False) for v in physical.split(16)]).cpu().numpy()
        del operator
        queries = PhysicalQueries([BatchedBorn(BornConfig(wave=w, numRec=60),
            cache_dir=ROOT/'data/born_cache', device=args.device).matrix for w in [4, 6, 8, 5, 7, 10, 12]])
        with torch.no_grad():
            targets = torch.cat([queries(v) for v in physical.split(16)]).cpu().numpy()
        if not (np.isfinite(targets).all() and (targets > 0).all()):
            raise ValueError('Relative-query metric has an incompatible target; retain all objects and inspect')
        np.save(scoring/'targets35.npy', targets)
        observations = []
        for seed in NOISE_SEEDS:
            for multiplier in MULTIPLIERS:
                path = out/f'noise{seed}_x{multiplier:g}.npy'
                np.save(path, signals+observation_noise(ids, seed, multiplier))
                observations.append(dict(noise_seed=seed, multiplier=multiplier, file=path.name, sha256=sha256(path)))
        atomic_json(out/'manifest.json', dict(status='completed', objects=100,
            freeze_sha256=sha256(args.freeze), source_archive_sha256=sha256(archive),
            source_lmdb_sha256=sha256(candidates[0]), object_range=[float(objects.min()), float(objects.max())],
            normalization='physical f read unchanged; fieldsA f,queries|Aq f|²; no image resize or clipping',
            ids_sha256=sha256(out/'ids.npy'), observations=observations,
            scoring_hashes={p.name:sha256(p) for p in scoring.glob('*.npy')},
            device=args.device, seconds=time.time()-started, official_test_opened=True))
        atomic_json(receipt, dict(status='completed', freeze_sha256=sha256(args.freeze),
            started=started, finished=time.time(), archive_sha256=sha256(archive),
            manifest_sha256=sha256(out/'manifest.json'), official_test_opened=True))
    except Exception as error:
        atomic_json(receipt, dict(status='failed', started=started, error=repr(error),
            freeze_sha256=sha256(args.freeze), partial_files_preserved=True))
        raise
    print(json.dumps(dict(status='completed', objects=100, observations=len(observations))))


if __name__ == '__main__':
    main()
