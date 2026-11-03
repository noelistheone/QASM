"""Outcome-free action scoring from an inferred low-rank spatial factor."""
import numpy as np
import torch
from src.low_rank_physical_update import project_queries,condition_latent,quadratic_utility


def score_all_actions(mean,factors,masks,matrix,physical,rows,scale):
    """Inputs contain neither true object nor unacquired observations."""
    batch,rank,dimension=factors.shape
    projected=project_queries(mean,factors,physical)
    field=(factors.reshape(batch*rank,dimension)@matrix.T).reshape(batch,rank,-1)
    per_action=field[:,:,rows].permute(0,2,3,1).reshape(batch*30,rows.shape[1],rank)
    update=condition_latent(per_action,torch.zeros(per_action.shape[:2],device=factors.device,dtype=factors.dtype),5e-9)
    repeated={key:value.repeat_interleave(30,dim=0) for key,value in projected.items()}
    utility=quadratic_utility(repeated,update['reduction'],scale)
    gram=(factors.to(torch.float64)@factors.to(torch.float64).transpose(1,2)).repeat_interleave(30,dim=0)
    image=torch.einsum('bij,bji->b',gram,update['reduction'])
    scores={'image_variance':image.reshape(batch,30),
            'first_order':utility['first_order'].reshape(batch,30),
            'second_order':utility['second_order'].reshape(batch,30)}
    for key in scores:scores[key]=scores[key].masked_fill(masks.bool(),-torch.inf)
    return scores,projected


def choose_simple(masks,ids,step,policy):
    mask_array=masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    choices=[]
    for mask,obj in zip(mask_array,ids):
        remaining=np.flatnonzero(mask==0);acquired=np.flatnonzero(mask!=0)
        if not len(remaining):raise ValueError('No unobserved action remains')
        if policy=='deterministic_random':
            action=np.random.default_rng(20261129+int(obj)+100000*step).choice(remaining)
        elif policy=='uniform_farthest_block':
            distance=np.abs(remaining[:,None]-acquired[None,:]);distance=np.minimum(distance,30-distance)
            action=remaining[np.argmax(distance.min(1))]
        else:raise ValueError(policy)
        choices.append(int(action))
    return np.asarray(choices)
