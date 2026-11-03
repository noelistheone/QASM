"""A shared scalar-query readout on the same spatial backbone as moment learning."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.posterior_components import MaskPosteriorComponents


class DirectQueryCNN(nn.Module):
    def __init__(self,feature_mean,feature_scale,log_mean,log_scale):
        super().__init__();self.spatial=MaskPosteriorComponents(8);self.norm=nn.LayerNorm(128)
        self.head=nn.Sequential(nn.Linear(201,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('feature_mean',torch.tensor(feature_mean,dtype=torch.float32))
        self.register_buffer('feature_scale',torch.tensor(feature_scale,dtype=torch.float32))
        self.register_buffer('log_mean',torch.tensor(log_mean,dtype=torch.float32));self.register_buffer('log_scale',torch.tensor(log_scale,dtype=torch.float32))
        waves=np.asarray([4.,6.,8.,5.,7.,10.,12.]);desc=np.concatenate([np.repeat(np.log(waves/6),5)[:,None],np.tile(np.eye(5),(7,1))],axis=1)
        self.register_buffer('descriptor',torch.tensor(desc,dtype=torch.float32))
        angles=np.abs((np.linspace(0,359,20)[:,None]-np.linspace(0,359,180)[None,:]+180)%360-180)
        bands=[(0,10),(10,30),(30,60),(60,90),(90,180)]
        weights=np.stack([(angles>=lo)&((angles<hi) if hi<180 else (angles<=hi)) for lo,hi in bands])
        self.register_buffer('band_weights',torch.tensor(weights,dtype=torch.float32))

    def forward(self,batch,physical,new_physical=None):
        raw=self.spatial.raw_forward(batch['feature'],batch['mean'],batch['mask'])
        spatial=self.norm(F.adaptive_avg_pool2d(raw,4).flatten(1))
        # These physical features use observations and frozen images, never target labels.
        mean_q=physical(batch['mean']);feature_q=physical(batch['feature'])
        acquired=batch['mask'].repeat_interleave(6,1)[:,None,:];band=self.band_weights[None]*acquired[:,None]
        power=batch['observation'].reshape(-1,20,180,2).square().sum(-1)-1e-8
        measured=(power[:,None]*band).sum((-1,-2))/band.sum((-1,-2))
        summary=torch.cat([mean_q,feature_q,measured,batch['mask'],batch['mask'].mean(1,keepdim=True)],1)
        summary=(summary-self.feature_mean)/self.feature_scale
        if new_physical is not None:mean_q=torch.cat([mean_q,new_physical(batch['mean'])],1)
        if bool((mean_q<=0).any()):raise FloatingPointError('Positive physical mean required for logqueryfeature')
        n,q=mean_q.shape;shared=torch.cat([spatial,summary],1)
        x=torch.cat([shared[:,None,:].expand(n,q,-1),self.descriptor[None,:q].expand(n,-1,-1),
                     ((mean_q.log()-self.log_mean)/self.log_scale)[:,:,None]],-1)
        return mean_q*torch.exp(self.head(x).squeeze(-1))
