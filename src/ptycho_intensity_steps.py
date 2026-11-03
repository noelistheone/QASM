"""Bounded physical intensity-loss steps for future learned-update experiments.

The forward solve is a one-dimensional quartic minimization. Its derivative is
admitted only on an isolated, nondegenerate interior branch or a strict active
endpoint. This module does not imply improved unseen-measurement prediction.
"""
import math
import numpy as np
import torch


class UncertifiedStepDerivative(RuntimeError):
    pass


def intensity_coefficients(fields, directions, target, weights=None):
    """Coefficients c[0:5] of 1/2 mean_w (sum_j |z_j+t h_j|^2-y)^2.

    The leading axis indexes globally shared fields. The other axes index any
    collection of measurement coordinates; no spatial approximation is made.
    """
    if not fields.is_complex() or fields.shape != directions.shape or fields.ndim < 2:
        raise ValueError('Matching complex fields with a leading mode axis are required')
    if target.shape != fields.shape[1:] or target.is_complex():
        raise ValueError('Target shape must omit the mode axis')
    if not torch.isfinite(fields).all() or not torch.isfinite(directions).all():
        raise ValueError('Nonfinite propagated field')
    if not torch.isfinite(target).all() or torch.any(target < 0):
        raise ValueError('Finite nonnegative intensity required')
    if weights is None:
        weights = torch.ones_like(target)
    weights = torch.broadcast_to(weights, target.shape)
    if not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Positive total, nonnegative measurement weights required')
    residual = fields.abs().square().sum(0)-target
    first = 2*(fields.conj()*directions).sum(0).real
    second = directions.abs().square().sum(0)
    normalizer = weights.sum()
    terms = [.5*residual.square(), residual*first,
             .5*first.square()+residual*second, first*second, .5*second.square()]
    return torch.stack([(weights*t).sum()/normalizer for t in terms])


def choose_quartic_step(coefficients, upper=1.):
    """Minimize the represented polynomial on [0, upper], including both ends.

    Stationary points are roots of its cubic derivative. Constant terms are
    removed during selection to avoid cancellation. Branch certificates use
    the rescaled interval u=t/upper and nonconstant coefficient scale.
    """
    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape != (5,) or not np.isfinite(c).all() or not math.isfinite(upper) or upper <= 0:
        raise ValueError('Five finite coefficients and a finite positive interval required')
    scaled = c*np.power(float(upper), np.arange(5))
    scale = float(np.max(np.abs(scaled[1:])))
    if not math.isfinite(scale):
        raise ValueError('Interval scaling overflows')
    if scale == 0:
        return dict(step=0., branch='flat_uncertified', coefficient_jacobian=None,
                    normalized_change=0., candidate_gap=0., candidates=[0., float(upper)])
    polynomial = scaled/scale
    polynomial[0] = 0.
    derivative = np.arange(1,5)*polynomial[1:]
    descending = np.trim_zeros(derivative[::-1], 'f')
    roots = np.roots(descending) if len(descending)>1 else np.array([])
    candidates = [0., 1.]
    for root in roots:
        if abs(root.imag) <= 1e-8*max(1., abs(root.real)):
            value = float(root.real)
            if 0. < value < 1. and all(abs(value-other)>1e-10 for other in candidates):
                candidates.append(value)
    candidates.sort()
    values = np.polynomial.polynomial.polyval(candidates, polynomial)
    # Earliest coordinate breaks exact value ties; no derivative is claimed there.
    best = min(range(len(candidates)), key=lambda i: (values[i], candidates[i]))
    u = candidates[best]
    ordered = np.sort(values)
    gap = float(ordered[1]-ordered[0])
    slope = float(np.polynomial.polynomial.polyval(u, derivative))
    curvature = float(np.polynomial.polynomial.polyval(u, np.arange(1,4)*derivative[1:]))
    jacobian = None
    branch = 'uncertified'
    if gap > 1e-10:
        if u == 0. and slope > 1e-9:
            jacobian = np.zeros(5)
            branch = 'strict_lower_endpoint'
        elif u == 1. and slope < -1e-9:
            jacobian = np.zeros(5)
            branch = 'strict_upper_endpoint'
        elif 0. < u < 1. and curvature > 1e-8 and abs(slope) < 1e-7:
            t = float(upper)*u
            original_curvature = scale*curvature/(float(upper)**2)
            jacobian = np.array([0., -1., -2*t, -3*t*t, -4*t**3])/original_curvature
            branch = 'isolated_interior_ift'
    if values[best] > 1e-13:
        raise FloatingPointError('Selected step increases the represented visible loss')
    return dict(step=float(upper)*u, branch=branch,
                coefficient_jacobian=None if jacobian is None else jacobian.tolist(),
                normalized_change=float(values[best]), candidate_gap=gap,
                normalized_stationarity_residual=slope, normalized_curvature=curvature,
                candidates=[float(upper)*x for x in candidates])


class _QuarticStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, upper):
        result = choose_quartic_step(coefficients.detach().cpu().double().numpy(), float(upper))
        ctx.certificate = result
        ctx.save_for_backward(coefficients)
        return coefficients.new_tensor(result['step'])

    @staticmethod
    def backward(ctx, derivative):
        coefficients, = ctx.saved_tensors
        jacobian = ctx.certificate['coefficient_jacobian']
        if jacobian is None:
            raise UncertifiedStepDerivative(
                f"No smooth selected-step derivative: {ctx.certificate['branch']}")
        gradient = derivative*coefficients.new_tensor(jacobian)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Unrepresentable step derivative')
        return gradient, None


def differentiable_quartic_step(coefficients, upper=1.):
    if (coefficients.ndim != 1 or coefficients.shape[0] != 5
            or not coefficients.is_floating_point()):
        raise ValueError('Real five-coefficient tensor required')
    return _QuarticStep.apply(coefficients, float(upper))


def positive_diagonal_direction(real_gradient, log_weights, denominator, floor=1e-3):
    """A descent proposal using the same positive pixel weights for every mode.

    real_gradient follows PyTorch's complex-gradient convention dL=Re<G,dB>.
    Common spatial weights preserve a unitary change of field coordinates if
    the features producing those weights are themselves unitary invariant.
    """
    if not real_gradient.is_complex() or not math.isfinite(floor) or floor <= 0:
        raise ValueError('Complex real-gradient and positive floor required')
    if torch.any(denominator <= 0) or not torch.isfinite(denominator).all():
        raise ValueError('Positive finite preconditioning denominator required')
    if not torch.isfinite(log_weights).all() or not torch.isfinite(real_gradient).all():
        raise ValueError('Finite proposal inputs required')
    factor = (torch.nn.functional.softplus(log_weights)+floor)/denominator
    return -factor*real_gradient
