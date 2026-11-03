"""Gaussian latent conditioning and exact affine-quadratic utility reference."""
import numpy as np
import torch

def action_rows():
    index=np.arange(20*180*2).reshape(20,180,2)
    return np.stack([index[:,a*6:(a+1)*6,:].ravel() for a in range(30)])

def condition(precision,rhs):
    chol=torch.linalg.cholesky(precision)
    covariance=torch.cholesky_inverse(chol)
    mean=torch.cholesky_solve(rhs[:,None],chol)[:,0]
    return mean,covariance

def candidate_covariances(precision,action_information):
    chol=torch.linalg.cholesky(precision[None,:,:]+action_information)
    return torch.cholesky_inverse(chol)

def query_means(mean,covariance,Q,linear,constant):
    plugin=torch.einsum('i,qij,j->q',mean,Q,mean)+2*(linear@mean)+constant
    return plugin,plugin+torch.einsum('qij,ji->q',Q,covariance)

def action_utilities(mean,covariance,next_covariances,Q,linear,scale,eigenvalues):
    delta=covariance[None,:,:]-next_covariances
    h=torch.einsum('qij,j->qi',Q,mean)+linear
    first=4*torch.einsum('qi,aij,qj->aq',h,delta,h)
    products=torch.einsum('qij,ajk->aqik',Q,delta)
    second=2*torch.einsum('aqij,aqji->aq',products,products)
    image=torch.einsum('ai,i->a',delta.diagonal(dim1=-2,dim2=-1),eigenvalues)
    return dict(image_variance=image,first_order=(first/scale.square()).mean(1),
                second_order=((first+second)/scale.square()).mean(1))
