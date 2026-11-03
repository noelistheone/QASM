"""Common complex patch assembly and explicitly tracked spatial support.

Centers are calibrated (x,y) pixel coordinates. The unnormalized splat is the
adjoint of zero-padded bilinear extraction at center+arange(size)-size/2.
Normalizing the splat is an overlap average, not an inverse of the sampler.
"""
import numpy as np
import torch


def assemble_patches(patches, centers_xy, canvas_shape, crop_size=None):
    if patches.ndim != 3 or not patches.is_complex() or patches.shape[-1] != patches.shape[-2]:
        raise ValueError('Expected a complex batch of square patches')
    count, full, _ = patches.shape
    crop = full if crop_size is None else crop_size
    if not isinstance(crop, int) or crop < 2 or crop > full or crop % 2 or full % 2:
        raise ValueError('Full and cropped patch sizes must be positive even integers')
    if len(canvas_shape) != 2 or any(int(v) != v or v < 2 for v in canvas_shape):
        raise ValueError('Expected integer canvas dimensions greater than one')
    centers = torch.as_tensor(centers_xy, device=patches.device, dtype=patches.real.dtype)
    if centers.shape != (count, 2) or not bool(torch.isfinite(centers).all() and torch.isfinite(patches).all()):
        raise ValueError('Expected one finite center per finite patch')
    h, w = map(int, canvas_shape)
    offset = full//2-crop//2
    value = patches[:, offset:offset+crop, offset:offset+crop]
    start = centers-crop/2
    lower = start.floor().to(torch.int64)
    fraction = start-lower
    yy, xx = torch.meshgrid(torch.arange(crop, device=patches.device),
                            torch.arange(crop, device=patches.device), indexing='ij')
    x = lower[:, 0, None, None]+xx
    y = lower[:, 1, None, None]+yy
    total = torch.zeros(h*w, device=patches.device, dtype=patches.dtype)
    weights = torch.zeros(h*w, device=patches.device, dtype=patches.real.dtype)
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        wx = fraction[:, 0] if dx else 1-fraction[:, 0]
        wy = fraction[:, 1] if dy else 1-fraction[:, 1]
        weight = (wx*wy)[:, None, None].expand(-1, crop, crop)
        xi, yi = x+dx, y+dy
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        index = (yi*w+xi)[valid]
        total.scatter_add_(0, index, (value*weight)[valid])
        weights.scatter_add_(0, index, weight[valid])
    total = total.reshape(h, w)
    weights = weights.reshape(h, w)
    support = weights > 0
    averaged = total/torch.where(support, weights, torch.ones_like(weights))
    return dict(object=averaged, accumulated=total, weight=weights, support=support)


def geometric_coverage(centers_xy, canvas_shape, crop_size):
    """Exact support union of positive bilinear splat weights, without images."""
    xy = np.asarray(centers_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Expected finite nonempty coordinates')
    if not isinstance(crop_size, int) or crop_size < 2 or crop_size % 2:
        raise ValueError('Expected an even crop size')
    h, w = map(int, canvas_shape)
    if min(h, w) < 2:
        raise ValueError('Canvas dimensions must exceed one')
    lower = np.floor(xy-crop_size/2).astype(np.int64)
    upper = np.ceil(xy+crop_size/2-1).astype(np.int64)
    x0, y0 = np.maximum(lower[:, 0], 0), np.maximum(lower[:, 1], 0)
    x1, y1 = np.minimum(upper[:, 0], w-1), np.minimum(upper[:, 1], h-1)
    valid = (x0 <= x1) & (y0 <= y1)
    x0, y0, x1, y1 = [v[valid] for v in [x0, y0, x1, y1]]
    difference = np.zeros((h+1, w+1), dtype=np.int64)
    np.add.at(difference, (y0, x0), 1)
    np.add.at(difference, (y1+1, x0), -1)
    np.add.at(difference, (y0, x1+1), -1)
    np.add.at(difference, (y1+1, x1+1), 1)
    return difference.cumsum(0).cumsum(1)[:h, :w] > 0


def query_probe_coverage(support, query_centers_xy, probe, batch_size=64):
    """Fraction of probe power falling on supported pixels at each query.

    This is geometry/calibration information, not an intensity-error bound: a
    diffraction pixel involves coherent interference and unknown object values.
    The probe shift matches the native integer-crop/Fourier-shift convention.
    """
    mask = np.asarray(support, dtype=bool)
    probe = np.asarray(probe, dtype=np.complex128)
    centers = np.asarray(query_centers_xy, dtype=np.float64)
    size = probe.shape[-1]
    if probe.shape != (size, size) or size % 2 or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('Invalid probe or query geometry')
    if not np.isfinite(probe).all() or not np.isfinite(centers).all() or np.linalg.norm(probe) == 0:
        raise ValueError('Nonfinite coordinates or invalid probe')
    h, w = mask.shape
    rounded = np.rint(centers).astype(np.int64)
    if ((rounded-size//2) < 0).any() or ((rounded+size//2) > np.array([w, h])).any():
        raise ValueError('Query probe window leaves declared canvas')
    frequency = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(frequency, frequency, indexing='ij')
    spectrum = np.fft.fft2(probe)
    values = []
    for start in range(0, len(centers), batch_size):
        integer = rounded[start:start+batch_size]
        fractional = centers[start:start+batch_size]-integer
        phase = np.exp(-2j*np.pi*(fy[None]*fractional[:, 1, None, None]
                                  +fx[None]*fractional[:, 0, None, None]))
        power = np.abs(np.fft.ifft2(spectrum[None]*phase))**2
        windows = np.stack([mask[y-size//2:y+size//2, x-size//2:x+size//2] for x, y in integer])
        values.append(np.sum(windows*power, axis=(-1, -2))/np.sum(power, axis=(-1, -2)))
    return np.concatenate(values)
