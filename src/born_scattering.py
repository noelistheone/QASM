"""Batched form of the pinned InverseBench linear scattering operator.

The benchmark is a real scalar scattering contrast, not a ptychographic
intensity. Its measurement is a complex scattered field. The source convention
conjugates the incident field. We preserve that convention exactly; this is not
a new Born-model derivation. See vendor/InverseBench/LICENSE and the parity audit.

Real measurement coordinates are ordered (transmitter, receiver, real/imag).
The cache maps physical contrast f in [0, 1] to these coordinates. A network
image x is normalized as f=(x+1)/2. No clipping occurs in the forward or gradient.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import fcntl
import tempfile

import numpy as np
from scipy.special import hankel1
import torch


@dataclass(frozen=True)
class BornConfig:
    Lx: float = 0.18
    Ly: float = 0.18
    Nx: int = 128
    Ny: int = 128
    wave: float = 6.0
    numRec: int = 360
    numTrans: int = 20
    sensorRadius: float = 1.6
    sigma_noise: float = 1e-4
    source_commit: str = "1db5932bc2995504491663d1e2c498e7317c5404"

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def greens(config):
    """Independent NumPy construction; no unused domain self-interaction term."""
    c = config
    dx, dy = c.Lx / c.Nx, c.Ly / c.Ny
    x, y = np.meshgrid(np.arange(-c.Nx/2, c.Nx/2)*dx,
                       np.arange(-c.Ny/2, c.Ny/2)*dy)
    kb = 2*np.pi / (dx*c.wave)

    def field(n):
        angles = np.deg2rad(np.linspace(0, 359, n))
        distance = np.hypot(x[..., None]-c.sensorRadius*np.cos(angles),
                            y[..., None]-c.sensorRadius*np.sin(angles))
        return (0.25j*hankel1(0, kb*distance)).reshape(-1, n)

    return field(c.numTrans), kb**2*field(c.numRec), dx*dy


def build_matrix(config, dtype=np.float32):
    t, r, area = greens(config)
    a = np.empty((config.numTrans, config.numRec, 2, config.Nx*config.Ny), dtype=dtype)
    for j in range(config.numTrans):
        z = area*(np.conj(t[:, j, None])*r).T
        a[j, :, 0, :], a[j, :, 1, :] = z.real, z.imag
    return a.reshape(-1, config.Nx*config.Ny)


def cached_matrix(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"born_{config.key}.npy"
    metadata = path.with_suffix('.json')
    # A cache entry consists of two files. Serialize both creation and loading,
    # so a second worker cannot observe an array without its matching manifest.
    # Unique temporaries also prevent the former shared .partial.npy race.
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not path.exists() or not metadata.exists():
            temporary = None
            temporary_json = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=path.stem+'.', suffix='.npy', delete=False) as f:
                    temporary = Path(f.name)
                    np.save(f, build_matrix(config))
                temporary.replace(path)
                manifest = {"config": asdict(config), "dtype": "float32",
                            "sha256": sha256(path), "coordinates": "transmitter,receiver,real-imag,pixel"}
                with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix=path.stem+'.', suffix='.json', delete=False) as f:
                    temporary_json = Path(f.name)
                    json.dump(manifest, f, indent=2)
                temporary_json.replace(metadata)
            finally:
                for candidate in (temporary, temporary_json):
                    if candidate is not None and candidate.exists():candidate.unlink()
        manifest = json.loads(metadata.read_text())
        if manifest['config'] != asdict(config):
            raise ValueError('Cached operator configuration mismatch')
    return np.load(path, mmap_mode='r'), path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class BatchedBorn:
    def __init__(self, config, matrix=None, cache_dir=None, device='cuda', dtype=torch.float32):
        self.config, self.device, self.dtype = config, torch.device(device), dtype
        self.cache_path = None
        if matrix is None:
            if cache_dir is None:
                matrix = build_matrix(config, np.float64 if dtype == torch.float64 else np.float32)
            else:
                matrix, self.cache_path = cached_matrix(config, cache_dir)
        self.matrix = torch.tensor(np.asarray(matrix), device=self.device, dtype=dtype)
        self.calls = {'forward': 0, 'gradient': 0}

    def forward(self, x, unnormalize=True):
        self.calls['forward'] += 1
        f = x.flatten(1).to(self.dtype)
        if unnormalize:
            f = (f+1)*0.5
        return f @ self.matrix.T

    def as_complex(self, y):
        c = self.config
        return torch.view_as_complex(y.reshape(-1, c.numTrans, c.numRec, 2).contiguous())

    def adjoint(self, y, normalized=True):
        scale = 0.5 if normalized else 1.0
        return (scale*(y.to(self.dtype) @ self.matrix)).reshape(
            -1, 1, self.config.Ny, self.config.Nx)

    def loss(self, x, observation):
        return (self.forward(x)-observation).square().sum(1)

    def gradient(self, x, observation, return_loss=False):
        self.calls['gradient'] += 1
        residual = self.forward(x)-observation
        g = self.adjoint(2*residual)
        return (g, residual.square().sum(1)) if return_loss else g

    def noisy(self, x, generator):
        y = self.forward(x)
        # torch.randn_like(complex) has variance 1/2 in each real component.
        noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
        return y + (self.config.sigma_noise / np.sqrt(2))*noise

    def spectral_norm_squared(self, steps=40):
        generator = torch.Generator(device=self.device).manual_seed(991)
        x = torch.randn(1, 1, self.config.Ny, self.config.Nx, device=self.device,
                        dtype=self.dtype, generator=generator)
        x /= x.norm()
        for _ in range(steps):
            y = self.adjoint(self.forward(x, unnormalize=False), normalized=False)
            x = y/y.norm()
        return float((x*self.adjoint(self.forward(x, unnormalize=False), normalized=False)).sum())
