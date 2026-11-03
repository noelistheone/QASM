"""Finite detector-convention comparisons with a profiled positive scale."""
import torch
from torch.nn import functional as F


def orient(array, code):
    if code not in range(8):raise ValueError('Eight square-detector symmetries')
    if code & 4:array=array.transpose(-2,-1)
    if code & 2:array=array.flip(-2)
    if code & 1:array=array.flip(-1)
    return array


def profile_crops(predictions, raw, orientation):
    """Compare M field predictions against every contained detector crop.

    predictions: (M,frames,N,N), raw: (frames,R,R). Each candidate uses one
    scale across every selected frame and detector pixel, never per-frame gains.
    """
    if predictions.ndim!=4 or raw.ndim!=3 or predictions.shape[1]!=len(raw):
        raise ValueError('Expected M predictions of the same raw frames')
    n=predictions.shape[-1];r=raw.shape[-1]
    if predictions.shape[-2]!=n or raw.shape[-2]!=r or n>r:
        raise ValueError('Contained square crops required')
    if not torch.isfinite(predictions).all() or not torch.isfinite(raw).all() or (predictions<0).any() or (raw<0).any():
        raise ValueError('Finite nonnegative intensity required')
    patches=F.unfold(orient(raw,orientation)[:,None],kernel_size=n)
    m=predictions.shape[0];count=(r-n+1)**2
    measured=patches.reshape(len(raw)*n*n,count)
    predicted=predictions.reshape(m,-1)
    pnorm=predicted.square().sum(1);ynorm=measured.square().sum(0)
    if (pnorm<=0).any() or (ynorm<=0).any():raise ValueError('Nonzero candidate signals required')
    dot=predicted@measured
    scale=dot/pnorm[:,None]
    error=(1-dot.square()/(pnorm[:,None]*ynorm[None])).clamp_min(0).sqrt()
    unscaled=((pnorm[:,None]+ynorm[None]-2*dot)/ynorm[None]).clamp_min(0).sqrt()
    p_amp=predicted.sqrt();y_amp=measured.sqrt()
    dot_amp=p_amp@y_amp;pnorm_amp=predicted.sum(1);ynorm_amp=measured.sum(0)
    scale_amp=(dot_amp/pnorm_amp[:,None]).square()
    error_amp=(1-dot_amp.square()/(pnorm_amp[:,None]*ynorm_amp[None])).clamp_min(0).sqrt()
    return dict(intensity_relative_rmse=error,intensity_gain=scale,
                amplitude_relative_rmse=error_amp,amplitude_intensity_gain=scale_amp,
                unscaled_intensity_relative_rmse=unscaled)
