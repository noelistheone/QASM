from pathlib import Path
import json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.response_moments import residual_terms,constrained_response

def main():
    torch.manual_seed(113);checks=[]
    def check(name,value,tolerance):
        error=float(value);row=dict(name=name,error=error,tolerance=tolerance,passed=bool(error<=tolerance))
        checks.append(row);assert row['passed'],row
    a=torch.randn(41,32,2,dtype=torch.float64);b=torch.randn_like(a)
    w=torch.rand(32,5,dtype=a.dtype);w/=w.sum(0)
    base,cross,energy=residual_terms(a,b,w)
    truth=b.square().sum(-1)@w
    check('sample_quadratic_identity',(base+cross+energy-truth).abs().max(),1e-14)
    check('sample_Cauchy_bound',(cross.square()-4*base*energy).clamp_min(0).max(),1e-13)
    # Same deterministic reference for a conditional sample ensemble.
    fixed=a[:1].expand_as(a);base,cross,energy=residual_terms(fixed,b,w)
    qb=base[0];qc=cross.mean(0);qe=energy.mean(0)
    check('conditional_quadratic_identity',(qb+qc+qe-truth.mean(0)).abs().max(),1e-14)
    check('conditional_Cauchy_bound',(qc.square()-4*qb*qe).clamp_min(0).max(),1e-13)
    rho=qc/(2*torch.sqrt(qb*qe))
    check('constrained_parameterization',(constrained_response(qb,qe,rho)-truth.mean(0)).abs().max(),1e-14)
    rho_extreme=torch.tensor([-1.,1.],dtype=a.dtype)[:,None]
    check('nonnegative_extreme_alignments',(-constrained_response(qb,qe,rho_extreme)).clamp_min(0).max(),1e-14)
    check('zero_base_query',abs(float(constrained_response(torch.tensor(0.),torch.tensor(3.),torch.tensor(-1.)))-3),0)
    # Exact two-point distribution mean1, variance.25, arbitrary reference2.
    x=torch.tensor([.5,1.5],dtype=torch.float64).reshape(2,1,1)
    base,cross,energy=residual_terms(torch.full_like(x,2.),x,torch.ones(1,1,dtype=x.dtype))
    check('biased_reference_needs_signed_cross',abs(float((base+cross+energy).mean())-1.25),0)
    check('positive_only_correction_counterexample',abs(float(base.mean()-x.square().mean())-2.75),0)
    # q=f^2, E[f]=0, E[f^2]=v. Joint risk derivative has a threshold.
    v=.25
    for lam in [1.,4.,20.]:
        optimum=np.sqrt(max(0.,v-1/(2*lam)))
        grid=np.linspace(-1,1,20001)
        # Drop the constant lambda E[f^4]; it cannot affect the minimizer.
        risk=grid**2+lam*(grid**4-2*v*grid**2)
        check(f'joint_loss_reference_distortion_lambda{lam}',abs(abs(grid[np.argmin(risk)])-optimum),6e-5)
    out=ROOT/'audit/m2_learning_20261101';out.mkdir(exist_ok=True,parents=True)
    (out/'moment_checks.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2))
    print(json.dumps(dict(passed=len(checks))))

if __name__=='__main__':main()
