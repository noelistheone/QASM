"""Controlled learning baselines for fixed physical queries; not a new-method claim."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def angular_weights(device='cpu', dtype=torch.float32):
    tx=np.linspace(0,359,20)[:,None];rx=np.linspace(0,359,60)[None,:]
    angles=np.abs((tx-rx+180)%360-180)
    bands=[(0,10),(10,30),(30,60),(60,90),(90,180)]
    masks=np.stack([(angles>=a)&((angles<b) if b<180 else (angles<=b)) for a,b in bands])
    weights=masks.reshape(5,-1).astype(np.float64)
    weights/=weights.sum(1,keepdims=True)
    return torch.tensor(weights.T,device=device,dtype=dtype)


def angular_power(fields, weights):
    """One wavelength; input (batch,2400), output (batch,5)."""
    return fields.reshape(-1,1200,2).square().sum(-1)@weights


class PhysicalQueries(nn.Module):
    def __init__(self, matrices):
        super().__init__()
        self.register_buffer('matrices',torch.stack(matrices))
        self.register_buffer('weights',angular_weights(self.matrices.device,self.matrices.dtype))

    def forward(self, image):
        x=image.flatten(1)
        return torch.cat([angular_power(x@a.T,self.weights) for a in self.matrices],dim=1)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(8,cout),nn.SiLU(),
                               nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(8,cout),nn.SiLU())
    def forward(self,x):return self.net(x)


class ResponseUNet(nn.Module):
    """Same spatial trunk across all three losses; a small direct-query head."""
    def __init__(self,width=32):
        super().__init__()
        self.e0=Block(1,width);self.e1=Block(width,width*2)
        self.e2=Block(width*2,width*4);self.e3=Block(width*4,width*8)
        self.d2=Block(width*12,width*4);self.d1=Block(width*6,width*2)
        self.d0=Block(width*3,width)
        self.image_head=nn.Conv2d(width,1,1)
        nn.init.zeros_(self.image_head.weight);nn.init.zeros_(self.image_head.bias)
        self.query_head=nn.Sequential(nn.Linear(width*16,256),nn.SiLU(),nn.Linear(256,15))
        nn.init.zeros_(self.query_head[-1].weight)
        nn.init.constant_(self.query_head[-1].bias,math.log(math.expm1(1.0)))

    def forward(self, image):
        a=self.e0(image);b=self.e1(F.avg_pool2d(a,2));c=self.e2(F.avg_pool2d(b,2))
        d=self.e3(F.avg_pool2d(c,2))
        u=self.d2(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        u=self.d1(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        u=self.d0(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        prediction=image+self.image_head(u)
        direct=F.softplus(self.query_head(F.adaptive_avg_pool2d(u,(4,4)).flatten(1)))
        return prediction,direct


def ridge_measurement_map(matrix, relative_alpha=1e-5):
    """Return T=(AA^T+lambda I)^-1 A, computed in float64.

    y@T is the physical-space ridge solution. Uses a power norm estimate with a
    fixed start; validation records the independently recomputed normal residual.
    """
    a=matrix.to(torch.float64)
    g=torch.Generator(device=a.device).manual_seed(991)
    x=torch.randn(a.shape[1],device=a.device,dtype=a.dtype,generator=g);x/=x.norm()
    for _ in range(60):
        v=a.T@(a@x);x=v/v.norm()
    norm2=float((a@x).square().sum())
    lam=relative_alpha*norm2
    gram=a@a.T
    gram.diagonal().add_(lam)
    factor=torch.linalg.cholesky(gram)
    mapping=torch.cholesky_solve(a,factor)
    return mapping,{'relative_alpha':relative_alpha,'lambda':lam,'norm_squared':norm2}
