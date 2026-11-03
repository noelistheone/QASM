"""A sufficient intensity-change bound using query operators, never labels.

For z=A_Q B and h=A_Q D, delta I(t)=t v+t^2 a, with
v=2 Re sum(conj(z)h) and a=sum |h|^2. The triangle inequality
gives ||delta I(t)||_2 <= t||v||_2+t^2||a||_2 for t>=0.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint


def field_statistics(fields, directions):
    """Measurement-first complex fields, with mode on axis one."""
    if fields.shape != directions.shape or fields.ndim != 4 or not fields.is_complex():
        raise ValueError('Matching complex (measurement,mode,y,x) arrays required')
    reference = fields.abs().square().sum(1).double()
    linear = (2*(fields.conj()*directions).sum(1).real).double()
    quadratic = directions.abs().square().sum(1).double()
    return torch.stack([reference.square().sum(), linear.square().sum(), quadratic.square().sum()])


def sufficient_cap(statistics, radius, relative_smoothing=1e-6):
    """Smooth conservative norms; clip the proposed direction at t<=1.

    The positive reference energy and positive radius are required. The
    smoothing increases both upper-bound coefficients, retaining the bound.
    Branch derivatives are ordinary derivatives away from the clipping tie.
    """
    if (statistics.shape != (3,) or not torch.isfinite(statistics).all()
            or torch.any(statistics < 0) or statistics[0] <= 0
            or not math.isfinite(relative_smoothing) or relative_smoothing <= 0):
        raise ValueError('Nonnegative statistics and positive reference energy/smoothing required')
    radius = torch.as_tensor(radius, device=statistics.device, dtype=statistics.dtype)
    if radius.ndim != 0 or not torch.isfinite(radius) or radius <= 0:
        raise ValueError('A finite positive scalar relative intensity-change radius is required')
    reference = statistics[0].sqrt()
    epsilon_squared = relative_smoothing**2*statistics[0]
    linear = (statistics[1]+epsilon_squared).sqrt()
    quadratic = (statistics[2]+epsilon_squared).sqrt()
    budget = radius*reference
    # Rationalized positive root avoids subtracting nearly equal terms.
    root = 2*budget/(linear+(linear.square()+4*quadratic*budget).sqrt())
    cap = torch.minimum(torch.ones_like(root), root)
    return cap, dict(unclipped_root=root, reference_norm=reference,
        conservative_linear_norm=linear, conservative_quadratic_norm=quadratic,
        radius=radius, strict_branch=bool(abs(float(root.detach())-1.)>1e-7))


class QueryResponseBound:
    """Only query coordinates and the known linear field operator are inputs."""
    def __init__(self, operator, ids, batch=256, checkpoint_batches=True):
        self.operator=operator
        self.ids=torch.as_tensor(ids,dtype=torch.long,device=operator.probe.device)
        if (self.ids.ndim!=1 or not len(self.ids) or len(self.ids.unique())!=len(self.ids)
                or torch.any(self.ids<0) or torch.any(self.ids>=len(operator.rounded))
                or not isinstance(batch,int) or batch<1):
            raise ValueError('Unique valid query positions and a positive integer batch required')
        self.batch=batch;self.checkpoint_batches=checkpoint_batches

    def _batch_statistics(self,columns,direction,ids):
        return field_statistics(self.operator.fields(columns,ids),self.operator.fields(direction,ids))

    def statistics(self,columns,direction):
        sums=torch.zeros(3,dtype=torch.float64,device=columns.device)
        for start in range(0,len(self.ids),self.batch):
            ids=self.ids[start:start+self.batch]
            if self.checkpoint_batches and torch.is_grad_enabled() and (columns.requires_grad or direction.requires_grad):
                value=checkpoint(self._batch_statistics,columns,direction,ids,
                    use_reentrant=False,preserve_rng_state=False)
            else:
                value=self._batch_statistics(columns,direction,ids)
            sums=sums+value
        return sums

    def constrain(self,columns,direction,radius):
        statistics=self.statistics(columns,direction)
        cap,detail=sufficient_cap(statistics,radius)
        return cap.to(direction.real.dtype)*direction,dict(statistics=statistics,cap=cap,**detail)
