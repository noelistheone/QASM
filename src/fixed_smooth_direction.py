"""Untrained positive Fourier smoothing of the whitened visible gradient."""
import math
import torch


class FixedSmoothDirection:
    def __init__(self, sigma):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('A positive, fixed smoothing width is required')
        self.sigma = float(sigma)

    def __call__(self, columns, problem, evaluated, fraction):
        root = problem.denominator.sqrt()
        v = evaluated['gradient'] / root
        fy = torch.fft.fftfreq(v.shape[-2], device=v.device, dtype=v.real.dtype)
        fx = torch.fft.fftfreq(v.shape[-1], device=v.device, dtype=v.real.dtype)
        weight = torch.exp(-2*math.pi**2*self.sigma**2*(fy[:,None]**2+fx[None,:]**2))
        smooth = torch.fft.ifft2(torch.fft.fft2(v)*weight)
        smooth = smooth * (problem.coverage > 0)
        if not (v.norm() > 0 and smooth.norm() > 0):
            raise ValueError('Nonzero visible gradient required')
        # For v supported on the illuminated set, Re<v, M S M v> > 0:
        # S is a strictly positive Fourier multiplier and M is an orthoprojector.
        direction = -smooth * (v.norm()/smooth.norm()) / root
        if (evaluated['gradient'].conj()*direction).sum().real >= 0:
            raise FloatingPointError('Positive smoothing lost strict descent numerically')
        return direction
