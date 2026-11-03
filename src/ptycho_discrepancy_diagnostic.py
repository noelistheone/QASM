"""Standard data discrepancies for an explicitly retrospective diagnostic.

These losses do not assert a calibrated noise likelihood for processed data.
No model training, line search, data loading, or query selection occurs here.
"""
import math
import torch

KINDS = ('intensity', 'amplitude', 'poisson_surrogate')


def pixel_loss_and_derivative(intensity, target, epsilon, kind):
    if kind not in KINDS:
        raise ValueError('Unknown discrepancy')
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError('Positive, finite smoothing floor required')
    if kind == 'intensity':
        residual = intensity-target
        return .5*residual.square(), residual
    prediction = intensity+epsilon
    observation = target+epsilon
    if kind == 'amplitude':
        residual = prediction.sqrt()-observation.sqrt()
        return .5*residual.square(), .5*(1-(observation/prediction).sqrt())
    # Generalized KL with the target-only constant retained. Noninteger target
    # arrays remain allowed; the name deliberately makes no photon-count claim.
    return prediction-observation+observation*torch.log(observation/prediction), 1-observation/prediction


def evaluate_discrepancy(operator, columns, ids, target, kind, epsilon, batch=256):
    ids = torch.as_tensor(ids, device=columns.device, dtype=torch.long)
    if tuple(target.shape) != (len(ids), operator.size, operator.size):
        raise ValueError('Measurement shape mismatch')
    if not torch.isfinite(target).all() or torch.any(target < 0):
        raise ValueError('Finite nonnegative targets required')
    loss = target.new_zeros(())
    gradient = torch.zeros_like(columns)
    for start in range(0, len(ids), batch):
        selected = ids[start:start+batch]
        fields = operator.fields(columns, selected)
        intensity = fields.abs().square().sum(1)
        values, derivative = pixel_loss_and_derivative(intensity, target[start:start+batch], epsilon, kind)
        loss = loss+values.sum()/target.numel()
        gradient = gradient+2*operator.adjoint(derivative[:, None]*fields, selected)/target.numel()
    return dict(loss=loss, gradient=gradient)


def real_inner(a, b):
    return (a.conj()*b).sum().real


def cosine(a, b):
    denominator=a.norm()*b.norm()
    if denominator == 0:
        return None
    return float(real_inner(a, b)/denominator)
