"""Global complex response fields with native-compatible ptychographic readout.

The columns share one object canvas. Their Gram matrix is a positive response
representation, not an assertion of calibrated uncertainty or physical modes.
This module is independent of the frozen M1/M3 and standard-baseline sources.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.ptychi_baseline_adapter import checked_geometry


class GlobalProbeOperator:
    """Linear complex field operator and its exact discrete adjoint.

    Integer object crops, fractional Fourier probe shifts, centered unitary
    detector FFT; coordinates are calibrated x/y pixel centers.
    """
    def __init__(self, probe, centers, shape):
        if probe.ndim != 2 or not probe.is_complex():
            raise ValueError('Expected a complex two-dimensional probe')
        centers = checked_geometry(centers, shape, probe.shape)
        self.shape = tuple(shape)
        self.size = probe.shape[-1]
        self.probe = probe
        # Choose the discrete patch in original float64 coordinates. Casting a
        # center near k+1/2 to float32 first can change the chosen integer pixel.
        rounded_numpy = np.rint(centers).astype(np.int64)
        rounded = torch.as_tensor(rounded_numpy, device=probe.device)
        self.fractional = torch.as_tensor(centers-rounded_numpy,
                                         dtype=probe.real.dtype, device=probe.device)
        axis = torch.arange(self.size, device=probe.device)-self.size//2
        self.rounded = rounded
        self.axis = axis
        freq = torch.fft.fftfreq(self.size, device=probe.device, dtype=probe.real.dtype)
        self.fy, self.fx = torch.meshgrid(freq, freq, indexing='ij')
        self.spectrum = torch.fft.fft2(probe)

    def shifted_probe(self, ids):
        d = self.fractional[ids]
        phase = torch.exp(-2j*torch.pi*(d[:, 0, None, None]*self.fx
                                      + d[:, 1, None, None]*self.fy))
        return torch.fft.ifft2(self.spectrum[None]*phase)

    def pixels(self, ids):
        rounded = self.rounded[ids]
        return ((rounded[:, 1, None, None]+self.axis[None, :, None])*self.shape[1]
                + rounded[:, 0, None, None]+self.axis[None, None, :]).long()

    def fields(self, columns, ids):
        if columns.ndim != 3 or tuple(columns.shape[-2:]) != self.shape or not columns.is_complex():
            raise ValueError('Columns must be complex (rank,height,width)')
        pixels = self.pixels(ids)
        # Output: measurement, column, detector_y, detector_x.
        patch = columns.reshape(len(columns), -1)[:, pixels].permute(1, 0, 2, 3)
        exit_wave = patch*self.shifted_probe(ids)[:, None]
        return torch.fft.fftshift(torch.fft.fft2(exit_wave, norm='ortho'), dim=(-2, -1))

    def adjoint(self, fields, ids):
        if fields.ndim != 4 or fields.shape[-2:] != (self.size, self.size):
            raise ValueError('Expected (measurement,column,detector_y,detector_x)')
        patches = torch.fft.ifft2(torch.fft.ifftshift(fields, dim=(-2, -1)), norm='ortho')
        patches = patches*self.shifted_probe(ids).conj()[:, None]
        rank = fields.shape[1]
        out = torch.zeros(rank, int(np.prod(self.shape)), dtype=fields.dtype, device=fields.device)
        ix = self.pixels(ids).reshape(1, -1).expand(rank, -1)
        out.scatter_add_(1, ix, patches.permute(1, 0, 2, 3).reshape(rank, -1))
        return out.reshape(rank, *self.shape)

    def coverage(self, ids):
        out = torch.zeros(int(np.prod(self.shape)), dtype=self.probe.real.dtype, device=self.probe.device)
        out.scatter_add_(0, self.pixels(ids).reshape(-1), self.shifted_probe(ids).abs().square().reshape(-1))
        return out.reshape(self.shape)


def invariant_features(obj, coverage, gradient, fraction):
    """Nine channels invariant to a common phase of object and loss gradient."""
    amplitude = obj.abs()
    phase = obj/(amplitude+1e-6)
    x = phase[:, 1:]*phase[:, :-1].conj()
    y = phase[1:]*phase[:-1].conj()
    x = F.pad(x, (0, 1)); y = F.pad(y, (0, 0, 0, 1))
    positive = coverage[coverage > 0]
    if not len(positive):
        raise ValueError('No illuminated pixels')
    mean_coverage = positive.mean()
    preconditioned = gradient/(coverage+1e-6*mean_coverage)
    relative = obj.conj()*preconditioned/(amplitude.square()+1e-4)
    features = torch.stack([torch.log1p(amplitude), x.real, x.imag, y.real, y.imag,
                            torch.log1p(coverage/mean_coverage),
                            torch.asinh(relative.real), torch.asinh(relative.imag),
                            torch.full_like(amplitude, float(fraction))])
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite invariant input features')
    return features


@torch.no_grad()
def observation_features(obj, operator, measured, fraction, batch=128):
    """Visible data only; measured and operator use the same intensity scale."""
    gradient = torch.zeros_like(obj)
    coverage = torch.zeros_like(obj.real)
    for start in range(0, len(measured), batch):
        ids = torch.arange(start, min(start+batch, len(measured)), device=obj.device)
        field = operator.fields(obj[None], ids)
        residual = field.abs().square()[:, 0]-measured[ids]
        gradient += operator.adjoint(field*residual[:, None], ids)[0]
        coverage += operator.coverage(ids)
    return invariant_features(obj, coverage, gradient, fraction), coverage, gradient


class ConvBlock(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inputs, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class GlobalResponseNet(nn.Module):
    """Shared whole-canvas CNN; a query-independent global field is decoded.

    Coherent: one corrected field. Moment: one corrected + four residual
    fields. Direct: the same five fields followed by a flexible positive
    detector-space head, removing the shared linear PSD measurement rule.
    """
    def __init__(self, kind='moment', width=24, residual_rank=4):
        super().__init__()
        if kind not in ('moment', 'coherent', 'direct'):
            raise ValueError('Unknown response model')
        self.kind = kind
        self.rank = 0 if kind == 'coherent' else residual_rank
        self.e0 = ConvBlock(9, width)
        self.e1 = ConvBlock(width, 2*width)
        self.middle = ConvBlock(2*width, 4*width)
        self.d1 = ConvBlock(6*width, 2*width)
        self.d0 = ConvBlock(3*width, width)
        self.field_head = nn.Conv2d(width, 2*(1+self.rank), 1)
        nn.init.normal_(self.field_head.weight, std=.001)
        nn.init.zeros_(self.field_head.bias)
        if kind == 'direct':
            # Powers + complex cross terms relative to column0 + detector xy.
            self.query_head = nn.Sequential(nn.Conv2d(3*self.rank+3, 32, 1), nn.SiLU(),
                                            nn.Conv2d(32, 1, 1))
            nn.init.zeros_(self.query_head[-1].weight)
            nn.init.zeros_(self.query_head[-1].bias)

    def forward(self, features, base_object):
        h0 = self.e0(features[None])
        h1 = self.e1(F.avg_pool2d(h0, 2))
        h2 = self.middle(F.avg_pool2d(h1, 2))
        h1 = self.d1(torch.cat([h1, F.interpolate(h2, size=h1.shape[-2:], mode='bilinear', align_corners=False)], 1))
        h0 = self.d0(torch.cat([h0, F.interpolate(h1, size=h0.shape[-2:], mode='bilinear', align_corners=False)], 1))
        values = self.field_head(h0)[0].reshape(1+self.rank, 2, *base_object.shape)
        correction = .1*torch.complex(values[:, 0], values[:, 1])
        # One common phase gauge is inherited from the global iterative input.
        columns = correction*base_object[None]
        columns = torch.cat([columns[:1]+base_object[None], columns[1:]], 0)
        return columns

    def readout(self, fields):
        powers = fields.abs().square()
        if self.kind != 'direct':
            return powers.sum(1)
        cross = fields[:, 1:]*fields[:, :1].conj()
        size = fields.shape[-1]
        frequency = torch.fft.fftshift(torch.fft.fftfreq(size, device=fields.device, dtype=fields.real.dtype))
        fy, fx = torch.meshgrid(frequency, frequency, indexing='ij')
        xy = torch.stack([fx, fy])[None].expand(len(fields), -1, -1, -1)
        inputs = torch.cat([torch.log1p(powers), torch.asinh(cross.real), torch.asinh(cross.imag), xy], 1)
        delta = self.query_head(inputs)[:, 0]
        reference = powers[:, 0]+1e-8
        # Stable inverse-softplus; zero head reproduces the coherent field.
        inverse = reference+torch.log(-torch.expm1(-reference))
        return F.softplus(inverse+delta)
