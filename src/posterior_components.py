"""Independent implementation of NPPC's published mathematical objectives.

Reference: Nehme, Yair and Michaeli, NeurIPS2023, main Eqs3–5 and SIAppendixA.
Two losses are intentionally separate: sample-error normalization from the
official implementation and a fixed-scale residual-second-moment objective.
This module is a control for future physical-update learning, not a new method.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.response_learning import Block
from src.mask_conditioned_learning import MaskUNet


def ordered_components(raw,epsilon=1e-12,detach_previous=True):
    """Modified Gram–Schmidt; previous directions are detached in projections."""
    flat=raw.flatten(2);directions=[];factors=[];variances=[]
    for i in range(flat.shape[1]):
        v=flat[:,i]
        for old in directions:
            basis=old.detach() if detach_previous else old
            v=v-(v*basis).sum(1,keepdim=True)*basis
        squared=v.square().sum(1)
        unit=v/torch.sqrt(squared.clamp_min(epsilon))[:,None]
        directions.append(unit);factors.append(v);variances.append(squared)
    return torch.stack(directions,1),torch.stack(variances,1),torch.stack(factors,1)


def component_loss(directions,variances,error,normalization,fixed_energy_scale,variance_weight):
    """Error-dependent scaling changes the estimand; see the normalization audit.

    Fixed energy is a positive training statistic, independent of each target.
    Stop-gradient on variance targets is part of the published NPPC mechanism.
    """
    err=error.flatten(1).detach()
    projection=torch.einsum('bkd,bd->bk',directions,err)
    if normalization=='sample':denominator=err.square().sum(1).clamp_min(1e-12)
    elif normalization=='fixed':
        if fixed_energy_scale<=0:raise ValueError('Positive fixed training scale required')
        denominator=torch.full_like(variances[:,0],fixed_energy_scale)
    else:raise ValueError(normalization)
    captured=projection.square()
    direction_loss=1-(captured/denominator[:,None]).sum(1).mean()
    variance_loss=((variances-captured.detach())/denominator[:,None]).square().mean()
    return direction_loss+variance_weight*variance_loss,dict(direction_loss=direction_loss,
        variance_loss=variance_loss,captured_energy=captured.detach().sum(1),error_energy=err.square().sum(1))


class MaskPosteriorComponents(nn.Module):
    def __init__(self,rank=8):
        super().__init__();self.rank=rank
        self.trunk=MaskUNet();self.trunk.e0=Block(2,32);self.trunk.head=nn.Conv2d(32,rank,1)
        nn.init.normal_(self.trunk.head.weight,std=.01);nn.init.zeros_(self.trunk.head.bias)

    def raw_forward(self,feature,mean,mask):
        t=self.trunk;condition=torch.cat([mask,mask.mean(1,keepdim=True)],1)
        def film(x,index):
            a,b=t.film[index](condition).chunk(2,1)
            return x*(1+a[:,:,None,None])+b[:,:,None,None]
        x=torch.cat([feature,mean.detach()],1)
        a=film(t.e0(x),0);b=film(t.e1(F.avg_pool2d(a,2)),1);c=film(t.e2(F.avg_pool2d(b,2)),2)
        d=film(t.e3(F.avg_pool2d(c,2)),3)
        u=t.d2(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        u=t.d1(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        u=t.d0(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return t.head(u)

    def forward(self,feature,mean,mask):
        return ordered_components(self.raw_forward(feature,mean,mask))

    def differentiable_factors(self,feature,mean,mask):
        """Full derivative for future-update objectives; NPPC keeps its published stop-gradient."""
        return ordered_components(self.raw_forward(feature,mean,mask),detach_previous=False)[2]
