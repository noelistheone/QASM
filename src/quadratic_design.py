"""Exact Gaussian/quadratic value-of-information reference, not a new method.

Unknown nuisance variables can be included in the joint latent covariance.
Queries depend on its leading structure coordinates only. This reference is
needed to avoid attributing ordinary curvature corrections to learned policies.
"""
import numpy as np


def quadratic_risk_reduction(mean, covariance, observation_matrix, noise_covariance,
                             query_matrices, structure_dimension):
    """Expected squared-error Bayes-risk reduction for q_j(x)=x.T S_j x.

    A joint Gaussian state z=(x,eta) is observed through H z + Gaussian noise.
    Returns the covariance of E[q(x)|observation], including quadratic terms.
    This is exact only for this Gaussian, linear-observation, quadratic-query
    reference. It is not a posterior-calibration guarantee for diffusion draws.
    """
    mu=np.asarray(mean,dtype=float)
    cov=np.asarray(covariance,dtype=float)
    h=np.asarray(observation_matrix,dtype=float)
    noise=np.asarray(noise_covariance,dtype=float)
    s=np.asarray(query_matrices,dtype=float)
    d=structure_dimension
    if cov.shape != (len(mu),len(mu)) or h.shape[1] != len(mu):
        raise ValueError('Joint mean, covariance and observation dimensions disagree')
    if noise.shape != (len(h),len(h)) or s.shape[1:] != (d,d):
        raise ValueError('Noise or query dimensions disagree')
    if not np.allclose(cov,cov.T) or not np.allclose(noise,noise.T):
        raise ValueError('Covariances must be symmetric')
    if np.linalg.eigvalsh(cov).min() < -1e-10 or np.linalg.eigvalsh(noise).min()<=0:
        raise ValueError('Joint covariance must be PSD and measurement noise positive definite')
    if not np.allclose(s,s.transpose(0,2,1)):
        raise ValueError('Query matrices must be symmetric')
    cy=h@cov@h.T+noise
    cross=cov[:d,:]@h.T
    gain=np.linalg.solve(cy,cross.T).T
    information=cross@gain.T
    information=(information+information.T)/2
    mu_x=mu[:d]
    qn=len(s)
    first=np.zeros((qn,qn));second=np.zeros((qn,qn))
    for j in range(qn):
        for k in range(qn):
            first[j,k]=4*mu_x@s[j]@information@s[k]@mu_x
            second[j,k]=2*np.trace(s[j]@information@s[k]@information)
    return {'linear_term':first,'curvature_term':second,'total':first+second,
            'structure_covariance_reduction':information,
            'structure_posterior_covariance':cov[:d,:d]-information,
            'predictive_observation_covariance':cy,'gain':gain}
