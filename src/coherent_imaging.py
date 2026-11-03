"""Declared single-patch coherent imaging operator for CPU verification.

F(x, p) = fftshift(fft2(p*x, norm='ortho')); I = gain * |F|^2.
This module does not infer experimental scaling, scan coordinates or photon dose.
It is a controlled building block, not a reproduced full ptychography solver.

Complex gradients use the real inner product Re(vdot(g, dx)).
"""
import numpy as np


def _pair(patch, probe):
    patch = np.asarray(patch, dtype=np.complex128)
    probe = np.asarray(probe, dtype=np.complex128)
    if patch.ndim != 2 or patch.shape != probe.shape or patch.size == 0:
        raise ValueError('patch and probe must have the same nonempty 2D shape')
    if not (np.isfinite(patch).all() and np.isfinite(probe).all()):
        raise ValueError('patch and probe must be finite')
    return patch, probe


def _gain(gain):
    gain = float(gain)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('gain must be finite and strictly positive')
    return gain


def field(patch, probe):
    patch, probe = _pair(patch, probe)
    return np.fft.fftshift(np.fft.fft2(probe * patch, norm='ortho'))


def field_adjoint(detector_field, probe):
    detector_field, probe = _pair(detector_field, probe)
    return np.conj(probe) * np.fft.ifft2(
        np.fft.ifftshift(detector_field), norm='ortho')


def intensity(patch, probe, gain=1.0):
    return _gain(gain) * np.abs(field(patch, probe)) ** 2


def intensity_jvp(patch, probe, direction, gain=1.0):
    patch, direction = _pair(patch, direction)
    z = field(patch, probe)
    dz = field(direction, probe)
    return 2 * _gain(gain) * np.real(np.conj(z) * dz)


def intensity_vjp(patch, probe, residual, gain=1.0):
    z = field(patch, probe)
    residual = np.asarray(residual)
    if residual.shape != z.shape or np.iscomplexobj(residual):
        raise ValueError('intensity cotangent must be real and match the detector')
    if not np.isfinite(residual).all():
        raise ValueError('intensity cotangent must be finite')
    return 2 * _gain(gain) * field_adjoint(residual * z, probe)


def squared_loss_and_gradient(patch, probe, observed, gain=1.0):
    predicted = intensity(patch, probe, gain)
    observed = np.asarray(observed)
    if observed.shape != predicted.shape or np.iscomplexobj(observed):
        raise ValueError('observed intensity must be real with a matching shape')
    if not np.isfinite(observed).all():
        raise ValueError('observations must be finite')
    residual = predicted - observed
    return (0.5 * float(np.sum(residual ** 2)),
            intensity_vjp(patch, probe, residual, gain))
