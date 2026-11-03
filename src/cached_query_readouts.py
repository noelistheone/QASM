"""Evaluate frozen scalar and Ridge readouts on shared observation-only features.

This is an evaluation adapter, not a new predictor. Its outputs are checked
against the original full scalar-CNN forward methods before a robustness run.
"""
import numpy as np
import torch
from torch.nn import functional as F


def direct_from_cache(net, feature, mean, masks, features66, mean_queries35):
    raw = net.spatial.raw_forward(feature, mean, masks)
    spatial = net.norm(F.adaptive_avg_pool2d(raw, 4).flatten(1))
    summary = (features66-net.feature_mean)/net.feature_scale
    if bool((mean_queries35 <= 0).any()):
        raise FloatingPointError('Frozen mean queries must be positive for the log feature')
    shared = torch.cat([spatial, summary], 1)
    n, q = mean_queries35.shape
    x = torch.cat([shared[:, None].expand(n,q,-1),
                   net.descriptor[None,:q].expand(n,-1,-1),
                   ((mean_queries35.log()-net.log_mean)/net.log_scale)[:,:,None]], -1)
    correction = net.head(x).squeeze(-1)
    if hasattr(net, 'readout_scale'):
        return mean_queries35 + net.readout_scale[None,:q]*correction
    return mean_queries35*correction.exp()


def ridge_from_cache(coefficients, features66, mean_queries35):
    width = len(coefficients['feature_mean'])
    if width not in [35,66]:
        raise ValueError('Expected the declared budget-specific or pooled Ridge35')
    x = np.asarray(features66, dtype=np.float64)[:,:width]
    mean = np.asarray(mean_queries35, dtype=np.float64)
    delta = (((x-coefficients['feature_mean'])/coefficients['feature_scale'])
             @coefficients['coefficient'].T + coefficients['intercept'])
    delta *= coefficients['target_scale']
    original = mean[:,:15] + delta
    old, new = np.array([4.,6.,8.]), np.array([5.,7.,10.,12.])
    weights = np.zeros((4,3))
    for i,wave in enumerate(new):
        j = int(np.clip(np.searchsorted(old,wave)-1,0,1))
        t = (wave-old[j])/(old[j+1]-old[j])
        weights[i,j], weights[i,j+1] = 1-t, t
    transfer = mean[:,15:] + np.einsum('ji,bik->bjk',weights,delta.reshape(-1,3,5)).reshape(-1,20)
    return np.concatenate([original, transfer], 1)
