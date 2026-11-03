"""Mask-aware mean reconstruction for a fixed instrument and varying actions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.response_learning import Block

INITIAL=(0,6,12,18,24)
REMAINING=np.array([i for i in range(30) if i not in INITIAL])

def masks_from_orders(orders,extra):
    masks=np.zeros((len(orders),30),dtype=np.float32);masks[:,list(INITIAL)]=1
    for i,order in enumerate(orders):masks[i,order[:int(extra[i])]]=1
    return masks

def evaluation_masks(ids,budget):
    orders=[np.random.default_rng(20261116+int(obj)).permutation(REMAINING) for obj in ids]
    return masks_from_orders(orders,np.full(len(ids),budget))

def masked_feature(observation,masks,mapping):
    # Coordinates remain transmitter, receiver, real/imag.
    channels=masks.repeat_interleave(6,dim=1)[:,None,:,None].expand(-1,20,-1,2).reshape(-1,7200)
    density=30/masks.sum(1,keepdim=True)
    return ((observation*channels)@mapping*density).reshape(-1,1,128,128)

def masked_cg_feature(observation,masks,matrix,regularization,iterations=30):
    """Fixed-budget CG for the actual subset normal equation, no density scaling.

    Returns a numerical feature, not a claim of an exact converged minimizer.
    The original full-rig-preconditioned feature is retained above as a failed
    initialization diagnostic; sparse masking exposed its weak-mode amplification.
    """
    channels=masks.repeat_interleave(6,dim=1)[:,None,:,None].expand(-1,20,-1,2).reshape(-1,7200)
    rhs=(observation*channels)@matrix
    x=torch.zeros_like(rhs);r=rhs.clone();direction=r.clone();rr=r.square().sum(1,keepdim=True)
    for _ in range(iterations):
        nd=((direction@matrix.T)*channels)@matrix+regularization*direction
        alpha=rr/(direction*nd).sum(1,keepdim=True).clamp_min(1e-30)
        x=x+alpha*direction;r=r-alpha*nd;rr_new=r.square().sum(1,keepdim=True)
        direction=r+rr_new/rr.clamp_min(1e-30)*direction;rr=rr_new
    residual=((x@matrix.T)*channels)@matrix+regularization*x-rhs
    relative=residual.norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-30)
    return x.reshape(-1,1,128,128),relative

class MaskUNet(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.e0=Block(1,width);self.e1=Block(width,width*2);self.e2=Block(width*2,width*4);self.e3=Block(width*4,width*8)
        self.d2=Block(width*12,width*4);self.d1=Block(width*6,width*2);self.d0=Block(width*3,width)
        self.film=nn.ModuleList([nn.Linear(31,2*c) for c in [width,width*2,width*4,width*8]])
        for layer in self.film:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.head=nn.Conv2d(width,1,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,image,mask):
        condition=torch.cat([mask,mask.mean(1,keepdim=True)],1)
        def film(x,index):
            a,b=self.film[index](condition).chunk(2,dim=1)
            return x*(1+a[:,:,None,None])+b[:,:,None,None]
        a=film(self.e0(image),0);b=film(self.e1(F.avg_pool2d(a,2)),1);c=film(self.e2(F.avg_pool2d(b,2)),2)
        d=film(self.e3(F.avg_pool2d(c,2)),3)
        u=self.d2(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        u=self.d1(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        u=self.d0(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return image+self.head(u)
