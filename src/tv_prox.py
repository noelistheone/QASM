"""Isotropic TV with replicated-edge differences and a nonnegative box.

Independent PyTorch implementation of the convex dual FGP iteration described
by Beck & Teboulle and used in the Sun et al. (2019) author implementation.
The definition is sum_pixels sqrt((D_y f)^2+(D_x f)^2), without pixel averaging.
The last forward difference on each axis is zero. D* is its exact adjoint.
"""
import torch


def difference(x):
    dy = torch.zeros_like(x); dx = torch.zeros_like(x)
    dy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    dx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return torch.stack((dy, dx), dim=-3)


def difference_adjoint(p):
    py, px = p.unbind(-3)
    out = torch.zeros_like(py)
    out[..., :-1, :] -= py[..., :-1, :]
    out[..., 1:, :] += py[..., :-1, :]
    out[..., :, :-1] -= px[..., :, :-1]
    out[..., :, 1:] += px[..., :, :-1]
    return out


def total_variation(x):
    return difference(x).square().sum(-3).sqrt().flatten(1).sum(1)


def tv_prox(y, weight, dual=None, iterations=50, lower=0., upper=float('inf')):
    """min_x .5||x-y||^2 + weight*TV(x), lower<=x<=upper.

    y has shape [batch,height,width]. weight is positive scalar or [batch].
    Returned feasible dual variables can warm-start the next proximal call.
    """
    weight = torch.as_tensor(weight, dtype=y.dtype, device=y.device)
    if weight.ndim == 0:
        weight = weight.expand(y.shape[0])
    weight = weight[:, None, None]
    if bool((weight <= 0).any()):
        raise ValueError('Positive proximal weights required')
    p = torch.zeros_like(difference(y)) if dual is None else dual.clone()
    extrapolated = p.clone(); momentum = 1.
    for _ in range(iterations):
        x = (y-weight*difference_adjoint(extrapolated)).clamp(lower, upper)
        proposed = extrapolated + difference(x)/(8*weight[:, None, :, :])
        proposed = proposed / proposed.square().sum(-3, keepdim=True).sqrt().clamp_min(1)
        next_momentum = (1+(1+4*momentum**2)**0.5)/2
        extrapolated = proposed + ((momentum-1)/next_momentum)*(proposed-p)
        p = proposed; momentum = next_momentum
    x = (y-weight*difference_adjoint(p)).clamp(lower, upper)
    return x, p


def proximal_certificate(y, x, weight, dual, lower=0., upper=float('inf')):
    """Primal minus feasible-dual value is a global optimality certificate."""
    weight = torch.as_tensor(weight, dtype=y.dtype, device=y.device)
    w = weight[:, None, None] if weight.ndim == 1 else weight
    primal = .5*(x-y).square().flatten(1).sum(1) + weight*total_variation(x)
    z = y-w*difference_adjoint(dual)
    projected = z.clamp(lower, upper)
    dual_value = .5*((z-projected).square()+y.square()-z.square()).flatten(1).sum(1)
    violation = dual.square().sum(-3).sqrt().flatten(1).max(1).values-1
    return dict(primal=primal, dual=dual_value, gap=primal-dual_value,
                dual_violation=violation.clamp_min(0))
