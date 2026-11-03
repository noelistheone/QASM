"""Query-conditioned residual heads around a deterministic physical reference."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.response_learning import Block


def query_metadata(dtype=torch.float32):
    bands=[(0,10),(10,30),(30,60),(60,90),(90,180)]
    return torch.tensor([[w/6,a/180,b/180,(a+b)/360,(b-a)/180]
                         for w in [4,6,8] for a,b in bands],dtype=dtype)


class ResidualResponseNet(nn.Module):
    """All physical response inputs retain amplitude outside normalization layers.

    The bounded output has a valid per-query Cauchy bound, not a guaranteed
    globally consistent image distribution or an uncertainty interval.
    """
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.register_buffer('metadata',query_metadata())
        self.encoder=None
        extra=0
        if mode!='feature_mlp':
            self.encoder=nn.Sequential(Block(1,32),nn.AvgPool2d(2),Block(32,64),
                nn.AvgPool2d(2),Block(64,128),nn.AvgPool2d(2),Block(128,256),
                nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(4096,256),nn.SiLU())
            extra=256
        self.head=nn.Sequential(nn.Linear(extra+15+5,256),nn.SiLU(),
                                nn.Linear(256,128),nn.SiLU(),nn.Linear(128,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        if mode in ('bounded','moments','positive'):
            with torch.no_grad():self.head[-1].bias[0]=math.log(math.expm1(.1))

    def forward(self,image,base):
        # base and all returned terms use the same per-query RMS normalization.
        features=base if self.encoder is None else torch.cat([self.encoder(image),base],1)
        inputs=torch.cat([features[:,None,:].expand(-1,15,-1),
                          self.metadata[None,:,:].expand(len(base),-1,-1)],2)
        values=self.head(inputs)
        if self.mode in ('response','feature_mlp'):
            return base+values[...,0],values[...,0],torch.zeros_like(base)
        energy=F.softplus(values[...,0])
        cross=torch.zeros_like(base) if self.mode=='positive' else 2*torch.sqrt(
            base.clamp_min(0)*energy+1e-30)*torch.tanh(values[...,1])
        return base+cross+energy,cross,energy
