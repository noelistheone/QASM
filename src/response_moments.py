"""Exact residual-energy identities for a deterministic reference reconstruction.

The identities are standard quadratic algebra. Learning their conditional terms
is an experimental design choice, not an established method advantage.
"""
import torch


def residual_terms(reference_field, true_field, weights):
    """Fields (batch,pairs,2), weights (pairs,queries), sum(weights[:,j])=1."""
    delta=true_field-reference_field
    base=reference_field.square().sum(-1)@weights
    cross=2*(reference_field*delta).sum(-1)@weights
    energy=delta.square().sum(-1)@weights
    return base,cross,energy


def constrained_response(base,energy,alignment):
    """Cauchy-consistent readout. energy>=0 and |alignment|<=1 required.

    Cross=2*sqrt(base*energy)*alignment. This makes the total nonnegative.
    It does not ensure calibration or that all queries share one image law.
    """
    cross=2*torch.sqrt(base*energy)*alignment
    return base+cross+energy
