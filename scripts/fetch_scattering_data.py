"""Fetch and verify the public InverseBench scattering LMDB archives.

Data are written under data/ (ignored by Git). Source: DOI 10.22002/zg89b-mpv16.
The image-specific upstream dataset notice is CC BY-NC-SA 4.0; see README.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://data.caltech.edu/api/records/zg89b-mpv16/files"
ARCHIVES = {
    "train": (83591632, "5717bed0a55efd676a3284630186b46389ccf789ad1b9e883c5d49d0a83b8ebe"),
    "test": (839942, "9c48cc6eaa481e5b08820e2bac9ab5b919786da83d4d81469685c1a940b8b120"),
    "val": (100095, "1d27c818a37f71e563d86cf3904163d90f60933ab3382d526a579c22d05e1f52"),
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=ARCHIVES, default="train")
    parser.add_argument("--archive", type=Path, help="Verify an already downloaded archive instead")
    args = parser.parse_args()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    name = f"inv-scatter-{args.split}"
    destination = data / name
    if destination.exists():
        raise FileExistsError(f"Keep the existing dataset: {destination}")
    expected_size, expected_hash = ARCHIVES[args.split]
    url = f"{SOURCE}/{name}.zip/content"
    with tempfile.TemporaryDirectory(prefix="scattering_", dir=data) as temporary:
        staging = Path(temporary)
        archive = args.archive
        if archive is None:
            archive = staging / f"{name}.zip"
            request = urllib.request.Request(url, headers={"User-Agent": "QASM-data-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=60) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
        if archive.stat().st_size != expected_size or sha256(archive) != expected_hash:
            raise ValueError("Archive differs from the recorded source; nothing was extracted")
        extracted = staging / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                path = (extracted / entry.filename).resolve()
                if not path.is_relative_to(extracted.resolve()):
                    raise ValueError(f"Unsafe archive member: {entry.filename}")
            zipped.extractall(extracted)
        prepared = extracted / name
        if not (prepared / "data.mdb").is_file():
            raise ValueError("The archive does not contain the expected LMDB dataset")
        shutil.move(str(prepared), destination)
    print(json.dumps({"split": args.split, "path": str(destination.relative_to(ROOT)),
                      "url": url, "sha256": expected_hash, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
