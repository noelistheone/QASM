"""Standard linear-algebra controls for quadratic scientific response queries.

The exact observability criterion is ker(A) subset ker(B) for q=||Bf||^2.
These utilities are mathematical baselines, not a new learning method.
"""
import numpy as np


def decompose_query(a, b, relative_cutoff=1e-12):
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    keep=s>relative_cutoff*s[0]
    pinv=(vt[keep].T/s[keep])@u[:,keep].T
    c=b@pinv
    visible=c@a
    unresolved=b-visible
    return dict(readout=c,visible_operator=visible,unresolved_operator=unresolved,
                numerical_rank=int(keep.sum()),
                unresolved_fraction=np.linalg.norm(unresolved)/max(np.linalg.norm(b),1e-300))


def observed_power_moments(mean_y, noise_covariance, readout):
    """Unbiased q estimate and its Gaussian-noise variance for B=C A.

    mean_y is noiseless Af; it is used only for oracle variance verification.
    The executable estimator needs y, C, R, and subtracts tr(C R C^T).
    """
    m=readout@mean_y
    covariance=readout@noise_covariance@readout.T
    return dict(response=float(m@m),bias_correction=float(np.trace(covariance)),
                estimator_variance=float(4*m@covariance@m+2*np.trace(covariance@covariance)))


def unbiased_observed_power(y, noise_covariance, readout):
    field=np.asarray(y)@readout.T
    return np.sum(field*field,axis=-1)-np.trace(readout@noise_covariance@readout.T)
