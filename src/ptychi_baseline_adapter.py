"""Visible-measurement adapter for the pinned public Pty-Chi LSQML solver.

Each task must run in its own process: Pty-Chi configures PyTorch global device,
dtype, random state and numerical settings. No reference image or query target
is accepted. Input coordinates are already calibrated physical patch centers
in pixel units; archive-specific parity corrections belong to the data adapter.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'sources/resources/real_ptycho_expansion_20261101/ptychi'
PINNED_COMMIT = '153ea64217b14fcc3f18e16445ba9f1c5bff4044'


@dataclass(frozen=True)
class LSQMLConfig:
    epochs: int = 100
    batch_size: int = 32
    seed: int = 20261229
    device: str = 'cpu'
    dtype: str = 'float64'
    noise_model: str = 'amplitude_gaussian'


def checked_geometry(centers_xy, object_shape, detector_shape):
    centers = np.asarray(centers_xy, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('Expected nonempty finite (frames,2) x/y centers')
    if len(object_shape) != 2 or any(int(v) != v or v < 1 for v in object_shape):
        raise ValueError('Expected positive integer object height and width')
    if len(detector_shape) != 2 or detector_shape[0] != detector_shape[1] or detector_shape[0] % 2:
        raise ValueError('This adapter uses an even square detector/probe grid')
    extent = detector_shape[0]
    rounded = np.rint(centers).astype(np.int64)
    lower = rounded-extent//2
    upper = rounded+extent//2
    if (lower < 0).any() or (upper > np.asarray(object_shape)[::-1]).any():
        raise ValueError('A probe patch lies outside the declared object grid')
    return centers


def fourier_probe_intensity(obj, probe, centers_xy, batch_size=64):
    """Native-compatible centered intensity, evaluated without fitting data.

    Integer object crops and fractional periodic Fourier shifts of the probe
    match the pinned source's single-slice, equal-grid far-field convention.
    The FFT here is unitary, whereas the source probe is divided by sqrt(Npix).
    """
    if obj.ndim != 2 or probe.ndim != 2 or not obj.is_complex() or not probe.is_complex():
        raise ValueError('Expected a complex object and a complex probe')
    centers_np = checked_geometry(centers_xy, obj.shape, probe.shape)
    if batch_size < 1:
        raise ValueError('Prediction batch size must be positive')
    centers = torch.as_tensor(centers_np, device=obj.device, dtype=obj.real.dtype)
    size = probe.shape[-1]
    frequency = torch.fft.fftfreq(size, device=obj.device, dtype=obj.real.dtype)
    fy, fx = torch.meshgrid(frequency, frequency, indexing='ij')
    probe_spectrum = torch.fft.fft2(probe)
    outputs = []
    for start in range(0, len(centers), batch_size):
        selected = centers[start:start+batch_size]
        rounded = selected.round().to(torch.int64)
        fractional = selected-rounded
        patches = torch.stack([obj[y-size//2:y+size//2, x-size//2:x+size//2]
                               for x, y in rounded.tolist()])
        phase = torch.exp(-2j*torch.pi*(fy[None]*fractional[:, 1, None, None]
                                     + fx[None]*fractional[:, 0, None, None]))
        shifted = torch.fft.ifft2(probe_spectrum[None]*phase)
        field = torch.fft.fftshift(torch.fft.fft2(patches*shifted, norm='ortho'), dim=(-2, -1))
        outputs.append(field.abs().square())
    return torch.cat(outputs)


class KnownProbeLSQML:
    """Standard object-only LSQML with explicit visible data and fixed epochs.

    The initial constant amplitude and numerical scaling use visible fluence
    alone. The source performs amplitude-Gaussian least squares by default;
    this does not assert that archived detector values are raw photon counts.
    """
    def __init__(self, visible_intensity, visible_centers_xy, probe,
                 object_shape, config=LSQMLConfig(), initial_object=None):
        intensity = np.asarray(visible_intensity, dtype=np.float64)
        probe_np = np.asarray(probe)
        if intensity.ndim != 3 or not len(intensity) or not np.isfinite(intensity).all() or (intensity < 0).any():
            raise ValueError('Expected finite nonnegative visible intensities')
        if probe_np.shape != intensity.shape[-2:] or not np.iscomplexobj(probe_np) or not np.isfinite(probe_np).all():
            raise ValueError('Expected a finite complex probe on the detector grid')
        centers = checked_geometry(visible_centers_xy, object_shape, probe_np.shape)
        if len(centers) != len(intensity):
            raise ValueError('One coordinate is required per visible frame')
        if config.epochs < 1 or config.batch_size < 1 or config.dtype not in ('float32', 'float64'):
            raise ValueError('Invalid solver duration, batch size or precision')
        if config.noise_model not in ('amplitude_gaussian', 'poisson'):
            raise ValueError('Unknown noise objective')
        device = torch.device(config.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Only explicit CPU/CUDA tasks are supported')
        if device.type == 'cuda':
            if device.index is None:
                raise ValueError('Specify a single CUDA device index')
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA device is unavailable')
            torch.cuda.set_device(device)
        mean_fluence = float(intensity.sum((-1, -2)).mean())
        probe_power = float(np.square(np.abs(probe_np)).sum())
        if mean_fluence <= 0 or probe_power <= 0:
            raise ValueError('A dark scan or zero probe cannot determine scale')
        self.scale = 1/mean_fluence
        self.initial_amplitude = float(np.sqrt(mean_fluence/probe_power))
        shape = tuple(map(int, object_shape))
        if initial_object is None:
            initial_object = np.full(shape, self.initial_amplitude, dtype=np.complex128)
        else:
            initial_object = np.asarray(initial_object)
            if initial_object.shape != shape or not np.iscomplexobj(initial_object) or not np.isfinite(initial_object).all():
                raise ValueError('Initial object must be finite, complex and have the declared grid')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
        if commit != PINNED_COMMIT:
            raise RuntimeError('Pty-Chi version differs from the checked source')
        sys.path.insert(0, str(SOURCE/'src'))
        import ptychi.api as api
        from ptychi.api.task import PtychographyTask
        options = api.LSQMLOptions()
        rec = options.reconstructor_options
        rec.default_device = api.Devices.CPU if device.type == 'cpu' else api.Devices.GPU
        rec.default_dtype = api.Dtypes.FLOAT64 if config.dtype == 'float64' else api.Dtypes.FLOAT32
        rec.batch_size = min(config.batch_size, len(intensity))
        rec.num_epochs = config.epochs
        rec.random_seed = config.seed
        rec.noise_model = api.NoiseModels.GAUSSIAN if config.noise_model == 'amplitude_gaussian' else api.NoiseModels.POISSON
        rec.allow_nondeterministic_algorithms = False
        rec.rescale_probe_intensity_in_first_epoch = False
        rec.forward_model_options.pad_for_shift = 0
        options.data_options.fft_shift = True
        options.object_options.pixel_size_m = 1.
        options.probe_options.pixel_size_m = 1.
        options.object_options.optimizable = True
        options.object_options.step_size = 1.
        options.object_options.remove_object_probe_ambiguity.enabled = False
        options.object_options.determine_position_origin_coords_by = api.ObjectPosOriginCoordsMethods.SPECIFIED
        options.object_options.position_origin_coords = [-.5, -.5]
        options.probe_options.optimizable = False
        options.probe_position_options.optimizable = False
        options.opr_mode_weight_options.optimizable = False
        dtype = torch.float64 if config.dtype == 'float64' else torch.float32
        complex_dtype = torch.complex128 if config.dtype == 'float64' else torch.complex64
        self.probe = torch.tensor(probe_np.copy(), dtype=complex_dtype, device=device)
        self.task = PtychographyTask(options,
            diffraction_data=torch.tensor(intensity*self.scale, dtype=dtype, device=device),
            object_data=torch.tensor(initial_object[None].copy(), dtype=complex_dtype, device=device),
            probe_data=self.probe[None, None]*np.sqrt(self.scale/intensity.shape[-1]**2),
            probe_position_x_px=torch.tensor(centers[:, 0], dtype=dtype, device=device),
            probe_position_y_px=torch.tensor(centers[:, 1], dtype=dtype, device=device))
        self.config = config
        self.centers = centers.copy()
        self.metadata = dict(config=asdict(config), source_commit=commit,
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            visible_frames=len(intensity), numerical_intensity_scale=self.scale,
            initial_constant_amplitude=self.initial_amplitude,
            initialization='visible-fluence constant' if initial_object is not None and np.all(initial_object == self.initial_amplitude) else 'caller-supplied object',
            probe_optimized=False, positions_optimized=False,
            object_grid=list(shape), coordinate_unit='computational pixel',
            physical_unit_note='Equal grid size set to 1 only fixes a ratio for single-slice far field; no metre-scale claim.',
            query_targets_used=False, reference_image_used=False)

    def run(self):
        self.task.run(n_epochs=self.config.epochs)
        return self.object()

    def object(self):
        return self.task.get_data('object')[0]

    @torch.no_grad()
    def predict(self, centers_xy, batch_size=64):
        return fourier_probe_intensity(self.object(), self.probe, centers_xy, batch_size)

    @torch.no_grad()
    def native_visible_prediction(self):
        count = len(self.centers)
        native = self.task.reconstructor.forward_model(torch.arange(count, device=self.probe.device))
        return torch.fft.fftshift(native, dim=(-2, -1))/self.scale

    def training_trace(self):
        return self.task.reconstructor.loss_tracker.table.to_dict('records')
