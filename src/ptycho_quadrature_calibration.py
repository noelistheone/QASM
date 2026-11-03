"""Two real-component scales fitted from visible intensity sufficient statistics.

This is an attribution control, not a reproduction of a trained PtychoPINN-CI
network. It retains one coherent complex object and its interference terms.
"""
import numpy as np
from numpy.polynomial import polynomial as poly


def loss_from_statistics(scales,h,d,yy):
    a,b=np.asarray(scales,dtype=np.float64)
    c=np.array([a*a,a*b,b*b])
    return float(c@h@c-2*d@c+yy)


def fit_scales(h,d,yy):
    """Profile amplitude and enumerate the real stationary direction roots.

    Basis order is |Aa|², 2Re[(Aa)* A(ib)], |A(ib)|². The coefficient vector
    (s1²,s1*s2,s2²) stays on the rank-one cone; s1,s2 are nonnegative.
    """
    h=np.asarray(h,dtype=np.float64);d=np.asarray(d,dtype=np.float64);yy=float(yy)
    if h.shape!=(3,3) or d.shape!=(3,) or not np.isfinite(h).all() or not np.isfinite(d).all() or not np.isfinite(yy):
        raise ValueError('Invalid sufficient statistics')
    h=(h+h.T)/2
    if yy<0 or np.min(np.linalg.eigvalsh(h)) < -1e-10*max(np.linalg.norm(h),1e-300):
        raise ValueError('Statistics are not positive semidefinite')
    one=np.ones(3);den=float(one@h@one);num=float(d@one)
    gain=max(0.,num/den) if den>0 else 1.
    scalar=np.repeat(np.sqrt(gain),2)
    scalar_loss=loss_from_statistics(scalar,h,d,yy)
    # Rescale the two fields, rather than independently rescaling three
    # coefficients; this preserves the rank-one constraint exactly.
    n1=max(0.,h[0,0])**.25 or 1.
    n2=max(0.,h[2,2])**.25 or 1.
    diagonal=np.array([1/n1**2,1/(n1*n2),1/n2**2])
    hs=diagonal[:,None]*h*diagonal[None,:];ds=diagonal*d
    numerator=ds.copy()
    denominator=np.array([hs[0,0],2*hs[0,1],hs[1,1]+2*hs[0,2],2*hs[1,2],hs[2,2]])
    # N(t)^2/D(t); a nonzero optimum obeys 2N'D - ND' = 0.
    stationary=poly.polysub(2*poly.polymul(poly.polyder(numerator),denominator),
                           poly.polymul(numerator,poly.polyder(denominator)))
    scale=float(np.max(np.abs(stationary))) if len(stationary) else 0.
    if scale>0:
        stationary=poly.polytrim(stationary/scale,tol=5e-14)
        roots=poly.polyroots(stationary) if len(stationary)>1 else np.array([])
    else:roots=np.array([])
    angles=[0.,np.pi/2,np.arctan2(n2,n1)]
    accepted=[]
    for root in roots:
        if abs(root.imag)<=1e-7*(1+abs(root.real)) and root.real>=0:
            angles.append(float(np.arctan(root.real)));accepted.append(float(root.real))
    candidates=[dict(scales=scalar.tolist(),loss=scalar_loss,kind='visible_scalar')]
    candidates.append(dict(scales=[1.,1.],loss=loss_from_statistics([1,1],h,d,yy),kind='identity'))
    for angle in angles:
        direction=np.array([np.cos(angle)/n1,np.sin(angle)/n2])
        coefficients=np.array([direction[0]**2,np.prod(direction),direction[1]**2])
        den=float(coefficients@h@coefficients);num=float(d@coefficients)
        if den<=0:continue
        amplitude_squared=max(0.,num/den)
        solution=np.sqrt(amplitude_squared)*direction
        candidates.append(dict(scales=solution.tolist(),loss=loss_from_statistics(solution,h,d,yy),
                               kind='stationary_or_boundary',angle=float(angle)))
    selected=min(candidates,key=lambda item:item['loss'])
    if not np.isfinite(selected['scales']).all():raise FloatingPointError('Nonfinite scale solution')
    tolerance=1e-10*max(1.,yy)
    if selected['loss']>scalar_loss+tolerance:raise AssertionError('Profiled solution lost the included scalar candidate')
    eigen=np.linalg.eigvalsh(hs)
    return dict(scalar_gain=gain,scalar_scales=scalar.tolist(),scalar_loss=scalar_loss,
        quadrature_scales=selected['scales'],quadrature_loss=selected['loss'],
        unchanged_loss=loss_from_statistics([1,1],h,d,yy),
        candidate_count=len(candidates),candidates=candidates,positive_stationary_roots=accepted,
        rescaled_gram_eigenvalues=eigen.tolist(),
        rescaled_gram_condition=float(eigen[-1]/eigen[0]) if eigen[0]>0 else None,
        objective='Visible raw-intensity MSE, multiplied by input-only alpha²',
        solver='Profile common amplitude; nonnegative two-field direction stationary polynomial plus boundaries and scalar candidate',
        interpretation='Two coherent quadrature scales in the fixed input-object gauge; no unique phase/material parameter interpretation.')
