"""Exact discrete-law check of sample-error normalization versus covariance."""
from pathlib import Path
import json
import numpy as np

def main():
    # Same conditioning observation y; zero conditional mean. No zero-error atom.
    e=np.array([[1.,0.],[-1.,0.],[0.,10.],[0.,-10.]])
    p=np.array([.45,.45,.05,.05]);energy=(e*e).sum(1)
    covariance=np.einsum('n,ni,nj->ij',p,e,e)
    normalized=np.einsum('n,ni,nj->ij',p/energy,e,e)
    assert np.allclose(p@e,0) and np.argmax(np.diag(covariance))==1 and np.argmax(np.diag(normalized))==0
    # Official norm loss is E[(v-(w^T e)^2)^2 / ||e||^4 | y].
    weighted_variances=p[:,None]*e**2
    weighted_variances=(weighted_variances/energy[:,None]**2).sum(0)/(p/energy**2).sum()
    derivative=2*np.sum(p[:,None]*(weighted_variances[None,:]-e**2)/energy[:,None]**2,axis=0)
    assert np.max(np.abs(derivative))<1e-12
    report=dict(passed=True,conditional_mean=(p@e).tolist(),true_covariance=covariance.tolist(),
        sample_normalized_direction_matrix=normalized.tolist(),true_top_direction=2,normalized_top_direction=1,
        sample_normalized_variance_loss_optima_for_coordinate_directions=weighted_variances.tolist(),
        variance_loss_derivative_at_optimum=derivative.tolist(),
        interpretation='A positive normalization depending only on y preserves the conditional covariance estimand; a realized-error-dependent weight generally changes it. This does not invalidate relative-error uncertainty visualization.')
    out=Path(__file__).resolve().parents[1]/'audit/m2_nppc_20261101';out.mkdir(exist_ok=True,parents=True)
    (out/'normalization_estimand.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
