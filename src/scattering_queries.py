"""Declared scientific queries for the controlled linear-scattering task.

These are normalized scattered-field responses of the benchmark contrast,
not biological material labels or calibrated optical power in watts.
"""
import torch


def sector_power(field_real, transmitters, receivers, sectors=3):
    """Mean squared complex field, averaged over transmitters and a sector."""
    if receivers % sectors:
        raise ValueError('Equal sectors require divisible receiver count')
    field=field_real.reshape(-1,transmitters,sectors,receivers//sectors,2)
    return field.square().sum(-1).mean((1,3))


def quadratic_ensemble_identity(field_real, transmitters, receivers, sectors=3):
    """E||B f||² = ||B E f||² + E||B(f-E f)||², with empirical weights.

    First dimension enumerates draws for ONE object. The variance uses 1/S,
    not the unbiased 1/(S-1) factor, because this is a discrete-law identity.
    """
    mean=field_real.mean(0,keepdim=True)
    query_mean=sector_power(field_real,transmitters,receivers,sectors).mean(0)
    plug_in=sector_power(mean,transmitters,receivers,sectors)[0]
    variance_correction=sector_power(field_real-mean,transmitters,receivers,sectors).mean(0)
    return query_mean,plug_in,variance_correction
