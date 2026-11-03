"""Equivariant complex proposals with visible descent and query-change control."""
import math
import torch
from torch import nn

from src.global_ptycho_response import invariant_features
from src.ptycho_intensity_steps import choose_quartic_step, differentiable_quartic_step


def orthogonal_direction(columns, gradient, problem, latent, maximum_ratio=2.):
    """A bounded orthogonal component in the coverage metric.

    The real complex inner product is used for projection. Before whitening
    back to object coordinates, the proposal has the same norm as the plain
    gradient and cosine at least 1/sqrt(1+maximum_ratio^2) with its negative.
    """
    if (len(columns)!=1 or latent.shape!=(2,*problem.operator.shape)
            or not math.isfinite(maximum_ratio) or maximum_ratio<=0):
        raise ValueError('A single field, two real latent maps and a positive ratio required')
    if not torch.isfinite(latent).all():
        raise FloatingPointError('Nonfinite learned proposal')
    root_d=problem.denominator.sqrt()
    v=gradient/root_d
    squared_norm=v.abs().square().sum()
    if squared_norm<=0:
        raise ValueError('Zero visible gradient has no certified strict-descent proposal')
    norm=squared_norm.sqrt()
    support=problem.coverage>0
    phase=columns[0]/(columns[0].abs()+1e-6)
    unprojected=phase*torch.complex(latent[0],latent[1])*support*norm/support.sum().sqrt()
    unprojected=unprojected[None]
    coefficient=(v.conj()*unprojected).sum().real/squared_norm
    perpendicular=unprojected-coefficient*v
    perpendicular_norm_squared=perpendicular.abs().square().sum()
    perpendicular=perpendicular*(maximum_ratio*norm)/(
        maximum_ratio**2*squared_norm+perpendicular_norm_squared).sqrt()
    proposal=-v+perpendicular
    proposal=proposal*norm/proposal.norm()
    return proposal/root_d


class OrthogonalUpdateNet(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(9,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,2,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,columns,problem,evaluated,fraction):
        summed=evaluated['gradient'][0]*problem.normalizer/2
        features=invariant_features(columns[0],problem.coverage,summed,fraction)
        latent=self.net(features[None])[0]
        return orthogonal_direction(columns,evaluated['gradient'],problem,latent)


def query_aware_refine(initial,problem,fraction,model=None,bound=None,radius=None,steps=3,
                       differentiable=True):
    if (radius is None)!=(bound is None):
        raise ValueError('Supply both a query bound and radius, or neither')
    columns=initial;records=[];expected=None
    for iteration in range(steps):
        ev=problem.evaluate(columns)
        before=float(ev['loss'].detach())
        if expected is not None and abs(before-expected)>5e-7+2e-5*abs(expected):
            raise FloatingPointError('Visible polynomial disagrees with the next physical state')
        if model is None:
            direction=problem.direction(ev['gradient'],torch.zeros_like(columns[0].real))
        else:
            direction=model(columns,problem,ev,fraction)
        cap_details=None
        if bound is not None:
            direction,details=bound.constrain(columns,direction,radius)
            if differentiable and not details['strict_branch']:
                raise RuntimeError('Query bound is at an uncertified clipping tie')
            cap_details={k:float(v.detach()) if isinstance(v,torch.Tensor) and v.ndim==0 else v
                         for k,v in details.items() if k!='statistics'}
        c=problem.coefficients(ev,direction)
        certificate=choose_quartic_step(c.detach().double().cpu().numpy())
        step=differentiable_quartic_step(c) if differentiable else c.new_tensor(certificate['step'])
        columns=columns+step*direction
        expected=float(sum(c[j]*step**j for j in range(5)).detach())
        if expected>before+5e-7+2e-5*abs(before):
            raise FloatingPointError('Visible loss increased')
        records.append(dict(iteration=iteration,visible_loss_before=before,
            visible_loss_after_polynomial=expected,step=float(step.detach()),
            certificate=certificate,query_bound=cap_details,
            algorithm_forward_frames=2*len(problem.ids)+(2*len(bound.ids) if bound is not None else 0),
            algorithm_adjoint_frames=len(problem.ids)))
    return columns,records
