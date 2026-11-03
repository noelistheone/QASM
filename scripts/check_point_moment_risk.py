"""An exact two-object example for the planned coherent-point control.

This is a standard conditional-expectation/rank calculation, not empirical
evidence for a neural model. It also distinguishes the training squared risk
from the relative absolute errors reported for predictive evaluation.
"""
from pathlib import Path
import datetime
import hashlib
import json

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    x = np.eye(2)  # Equal prior; the observed total power is one for either object.
    vectors = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
    vectors[2:] /= np.sqrt(2.)
    operators = np.einsum('qi,qj->qij', vectors, vectors)

    def q(z):
        return np.einsum('...i,qij,...j->...q', z, operators, z)

    mu = x.mean(0)
    moment = np.einsum('bi,bj->ij', x, x) / 2
    covariance = moment - np.outer(mu, mu)
    correction = np.array([.5, -.5])
    target = q(x)
    bayes = target.mean(0)
    assert np.allclose(bayes, .5, atol=1e-15)
    assert np.allclose(q(mu) + q(correction), bayes, atol=1e-15)
    assert np.allclose(covariance, np.outer(correction, correction), atol=1e-15)
    assert np.linalg.matrix_rank(moment) == 2 and np.linalg.matrix_rank(covariance) == 1
    # For a point z=(a,b), t=||z||^2 and the squared discrepancy of its four
    # queries from the Bayes means is (3/2)(t-2/3)^2+1/3, independent of angle.
    angles = np.linspace(0., 2*np.pi, 129)
    t = np.array([0., .2, 2/3, 1., 3.])
    z = np.sqrt(t[:, None, None])*np.stack([np.cos(angles), np.sin(angles)], -1)[None]
    excess = ((q(z)-bayes)**2).sum(-1)
    identity = 1.5*(t[:, None]-2/3)**2 + 1/3
    maximum_identity_error = float(np.max(np.abs(excess-identity)))
    assert maximum_identity_error < 2e-14
    bayes_total_risk = float(np.mean(np.sum((target-bayes)**2, axis=1)))
    best_point = np.sqrt(2/3)*np.array([1., 0.])
    point_total_risk = float(np.mean(np.sum((target-q(best_point))**2, axis=1)))
    assert abs(bayes_total_risk-.5) < 1e-14
    assert abs(point_total_risk-5/6) < 1e-14
    # Positive scalar labels: relative risk targets a different functional.
    scalar = np.array([1., 3.])
    squared_optimum = float(scalar.mean())
    relative_squared_optimum = float(np.mean(1/scalar)/np.mean(1/scalar**2))
    assert abs(squared_optimum-2.) < 1e-14
    assert abs(relative_squared_optimum-1.2) < 1e-14
    relative_mae_at_mean = float(np.mean(np.abs(2/scalar-1)))
    relative_mae_at_weighted_median = float(np.mean(np.abs(1/scalar-1)))
    assert relative_mae_at_weighted_median < relative_mae_at_mean
    report = dict(status='passed', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        objects=x.tolist(), observation='norm(x)^2=1, identical for the two objects',
        query_operators=operators.tolist(), conditional_mean=mu.tolist(),
        conditional_second_moment=moment.tolist(), conditional_covariance=covariance.tolist(),
        exact_conditional_query_means=bayes.tolist(), point_minimum_query_excess_total=1/3,
        bayes_total_squared_risk=bayes_total_risk, best_point_total_squared_risk=point_total_risk,
        checked_norms=t.tolist(), checked_angles=len(angles),
        point_risk_identity_max_abs_error=maximum_identity_error,
        metric_counterexample=dict(positive_targets=scalar.tolist(), squared_loss_optimum=squared_optimum,
            relative_squared_loss_optimum=relative_squared_optimum,
            relative_absolute_loss_optimum=1., relative_mae_at_mean=relative_mae_at_mean,
            relative_mae_at_weighted_median=relative_mae_at_weighted_median),
        interpretation='Standard rank/conditional-risk example. It motivates the matched point control, '
            'but neither proves a learned factor is a posterior covariance nor guarantees a gain '
            'on the existing scattering data. Relative MAE has a different Bayes target.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination = root/'audit/m3_preparation_20261101/point_moment_risk.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status='passed', max_identity_error=maximum_identity_error,
                         point_excess=1/3, relative_risk_targets=[2., 1.2, 1.])))


if __name__ == '__main__':
    main()
