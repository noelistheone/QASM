"""Low-rank Gaussian physical updates used as a reference and trainable layer.

These are standard Gaussian-conditioning and quadratic-moment identities.
They do not certify that a learned factor is the true posterior covariance.
Factors have shape (batch, rank, image_dimension), i.e. columns of L are rows here.
"""
import math
import torch


def condition_latent(measurement_factors, innovation, noise_variance):
    """Condition z = H mu + (HL) v + noise, v~N(0,I), with positive diagonal noise.

    Inputs HL:(batch,channels,rank), innovation:(batch,channels).
    The small rank-space solve is performed in float64, including gradients.
    """
    if measurement_factors.shape[:-1] != innovation.shape:
        raise ValueError('Measurement coordinates do not agree')
    dtype = measurement_factors.dtype
    response = measurement_factors.to(torch.float64)
    residual = innovation.to(torch.float64)
    variance = torch.as_tensor(noise_variance, device=response.device, dtype=response.dtype)
    variance = torch.broadcast_to(variance, residual.shape)
    if not bool((variance > 0).all()):
        raise ValueError('Noise variance must be strictly positive')
    whitened = response / variance.sqrt()[..., None]
    innovation_white = residual / variance.sqrt()
    k = response.shape[-1]
    precision = torch.eye(k, device=response.device, dtype=response.dtype) + whitened.transpose(-1,-2) @ whitened
    cholesky = torch.linalg.cholesky(precision)
    rhs = (whitened.transpose(-1,-2) @ innovation_white[...,None])
    mean = torch.cholesky_solve(rhs, cholesky)[...,0]
    covariance = torch.cholesky_inverse(cholesky)
    # Positive sum form avoids subtracting two large Mahalanobis terms.
    remaining = innovation_white - (whitened @ mean[...,None])[...,0]
    mahalanobis = remaining.square().sum(-1) + mean.square().sum(-1)
    log_determinant = variance.log().sum(-1) + 2 * cholesky.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    nll = .5 * (mahalanobis + log_determinant + residual.shape[-1] * math.log(2*math.pi))
    identity = torch.eye(k, device=response.device, dtype=response.dtype)
    return dict(mean=mean, covariance=covariance, reduction=identity-covariance,
                predictive_nll=nll, predictive_mahalanobis=mahalanobis, original_dtype=dtype)


def project_queries(mean, factors, physical):
    """q(mu+Lv)=constant+2 linear^T v+v^T quadratic v for all declared queries."""
    batch, rank, dimension = factors.shape
    if mean.flatten(1).shape != (batch, dimension):
        raise ValueError('Mean/factor shape mismatch')
    constants, linears, quadratics = [], [], []
    weights = physical.weights.repeat_interleave(2, dim=0).to(torch.float64)
    for matrix in physical.matrices:
        mean_field = (mean.flatten(1) @ matrix.T).to(torch.float64)
        field = (factors.reshape(batch*rank, dimension) @ matrix.T).reshape(batch, rank, -1).to(torch.float64)
        constants.append(mean_field.square() @ weights)
        linears.append(torch.einsum('bkm,mq,bm->bqk', field, weights, mean_field))
        quadratics.append(torch.einsum('bkm,mq,blm->bqkl', field, weights, field))
    return dict(constant=torch.cat(constants,1), linear=torch.cat(linears,1),
                quadratic=torch.cat(quadratics,1))


def quadratic_readout(projected, latent_mean=None, latent_covariance=None):
    quadratic = projected['quadratic']
    if latent_mean is None:
        return projected['constant'] + quadratic.diagonal(dim1=-2,dim2=-1).sum(-1)
    if latent_covariance is None:
        raise ValueError('Updated readout needs the conditional covariance')
    return (projected['constant'] + 2*torch.einsum('bqk,bk->bq',projected['linear'],latent_mean)
            + torch.einsum('bk,bqkl,bl->bq',latent_mean,quadratic,latent_mean)
            + torch.einsum('bqkl,blk->bq',quadratic,latent_covariance))


def quadratic_utility(projected, reduction, scale):
    """Expected squared-error Bayes-risk reduction under this Gaussian model.

    reduction = I-C_a, which depends on a candidate action but not its outcome.
    Calling this on a learned surrogate is a model-based score, not empirical gain.
    """
    linear, quadratic = projected['linear'], projected['quadratic']
    first = 4*torch.einsum('bqk,bkl,bql->bq',linear,reduction,linear)
    product = quadratic @ reduction[:,None,:,:]
    second = 2*torch.einsum('bqkl,bqlk->bq',product,product)
    return dict(first_order=(first/scale.square()).mean(1),
                second_order=((first+second)/scale.square()).mean(1), per_query=first+second)


def field_action_view(mean, factors, matrix, action_rows, actions):
    """Project to complete physical channels, then gather only requested actions.

    Caller supplies an unacquired action per object. No measured target appears
    in the learned factors or the action score.
    """
    batch, rank, dimension = factors.shape
    rows = action_rows[actions]
    means = mean.flatten(1) @ matrix.T
    fields = (factors.reshape(batch*rank,dimension) @ matrix.T).reshape(batch,rank,-1)
    selected_mean = means.gather(1,rows)
    selected_factors = fields.gather(2,rows[:,None,:].expand(-1,rank,-1)).transpose(1,2)
    return selected_mean, selected_factors
