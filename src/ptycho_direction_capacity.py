"""Truth-dependent local direction capacities for diagnosis, not inference."""
import math
import torch


def inner(x, y):
    return (x.conj()*y).sum().real


def direction_oracles(visible, query, margin=1/math.sqrt(5)):
    """Minimize a query linearization in three visible-gradient descent cones.

    Inputs are gradients in whitened complex coordinates with a real inner
    product. All proposals have norm at most ||visible||. A zero proposal is
    returned when that cone cannot decrease the query linearization.
    """
    if (visible.shape != query.shape or not visible.is_complex()
            or not query.is_complex() or not 0 <= margin < 1
            or not torch.isfinite(visible).all() or not torch.isfinite(query).all()):
        raise ValueError('Matching finite complex gradients and margin in [0,1) required')
    radius = visible.norm()
    if radius <= 0:
        raise ValueError('A nonzero visible gradient is required for this diagnostic')
    norm_query = query.norm()
    z = visible.abs().square()
    projection = torch.where(z > 0, visible*(visible.conj()*query).real.clamp_min(0)/z.clamp_min(torch.finfo(z.dtype).tiny),
                             torch.zeros_like(visible))
    projected_norm = projection.norm()
    diagonal = -radius*projection/projected_norm if projected_norm > 0 else torch.zeros_like(visible)
    vhat = visible/radius

    def angle_cone(gamma):
        if norm_query == 0:
            return torch.zeros_like(visible)
        alignment = inner(vhat, query)/norm_query
        if alignment >= gamma:
            return -radius*query/norm_query
        perpendicular = query-vhat*inner(vhat, query)
        norm_perpendicular = perpendicular.norm()
        if norm_perpendicular <= 1e-14*norm_query:
            return torch.zeros_like(visible)
        candidate = gamma*vhat+math.sqrt(1-gamma**2)*perpendicular/norm_perpendicular
        return -radius*candidate if inner(query, candidate) > 0 else torch.zeros_like(visible)

    return dict(uniform=-visible, positive_diagonal_oracle=diagonal,
                nonascending_halfspace_oracle=angle_cone(0.),
                fixed_angle_oracle=angle_cone(margin))
