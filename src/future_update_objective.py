"""Separate current-query fitting from learning through a future physical update."""
import torch
from src.posterior_components import component_loss
from src.low_rank_physical_update import project_queries,field_action_view,condition_latent,quadratic_readout


def update_objective(network,batch,context,mode,actions,action_rows,energy_scale,diagnostics=True):
    if mode=='continue_fixed':
        directions,variances,factors=network(batch['feature'],batch['mean'],batch['mask'])
        loss,component_metrics=component_loss(directions,variances,batch['truth']-batch['mean'],
                                              'fixed',energy_scale,.1)
        if not diagnostics:
            return dict(loss=loss,component=component_metrics)
    elif mode in ['current','future']:
        factors=network.differentiable_factors(batch['feature'],batch['mean'],batch['mask'])
        component_metrics=None
    else:raise ValueError(mode)
    # Previously acquired and newly acquired channels must be disjoint.
    if bool((batch['mask'].gather(1,actions[:,None])!=0).any()):
        raise ValueError('Future action is already acquired')
    projected=project_queries(batch['mean'],factors,context.physical)
    current=quadratic_readout(projected)
    predicted_field,field_factors=field_action_view(batch['mean'],factors,context.matrix,action_rows,actions)
    new_observation=batch['observation'].gather(1,action_rows[actions])
    update=condition_latent(field_factors,new_observation-predicted_field,5e-9)
    future=quadratic_readout(projected,update['mean'],update['covariance'])
    current_loss=((current-batch['target'])/context.query_scale).square().mean()
    future_loss=((future-batch['target'])/context.query_scale).square().mean()
    if mode=='current':loss=current_loss
    elif mode=='future':loss=.5*(current_loss+future_loss)
    return dict(loss=loss,current_loss=current_loss,future_loss=future_loss,
                current=current,future=future,factors=factors,update=update,component=component_metrics)
