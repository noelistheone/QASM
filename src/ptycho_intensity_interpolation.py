"""Coordinate-only weights for direct interpolation of visible intensities.

This is a non-learning comparison for dense scans, not an object reconstruction.
No probe, reference object or queried detector value enters the predictions.
"""
import numpy as np
from scipy.spatial import cKDTree


def interpolation_weights(visible_xy, query_xy, neighbors=4, power=2.):
    visible = np.asarray(visible_xy, dtype=np.float64)
    query = np.asarray(query_xy, dtype=np.float64)
    if any(x.ndim != 2 or x.shape[1] != 2 or len(x) == 0 or not np.isfinite(x).all()
           for x in [visible, query]):
        raise ValueError('Expected nonempty finite coordinate arrays')
    if not isinstance(neighbors, int) or neighbors < 1 or not np.isfinite(power) or power <= 0:
        raise ValueError('Positive integer neighborhood and positive distance power are required')
    k = min(neighbors, len(visible))
    tree = cKDTree(visible)
    distances, indices = tree.query(query, k=k)
    distances = np.asarray(distances).reshape(len(query), k)
    indices = np.asarray(indices).reshape(len(query), k)
    rows, weights = [], []
    for point, boundary in zip(query, distances[:, -1]):
        # Include boundary ties, then sort by coordinate and row identity so
        # cKDTree implementation details never choose a different distance tie.
        candidates = np.asarray(tree.query_ball_point(point, np.nextafter(boundary, np.inf)), dtype=np.int64)
        d = np.linalg.norm(visible[candidates]-point, axis=1)
        order = np.lexsort((candidates, visible[candidates, 1], visible[candidates, 0], d))[:k]
        chosen, d = candidates[order], d[order]
        if (d == 0).any():
            w = (d == 0).astype(np.float64)
        else:
            # Ratios avoid large intermediate inverse powers at nearby points.
            w = (d.min()/d)**power
        w /= w.sum()
        rows.append(chosen)
        weights.append(w)
    return np.stack(rows), np.stack(weights)


def predict_intensity(visible_intensity, visible_xy, query_xy,
                      neighbors=4, power=2., batch_size=64):
    values = np.asarray(visible_intensity)
    if values.ndim != 3 or len(values) != len(visible_xy) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected one finite nonnegative detector frame per visible coordinate')
    if batch_size < 1:
        raise ValueError('Prediction batch size must be positive')
    indices, weights = interpolation_weights(visible_xy, query_xy, neighbors, power)
    output = np.empty((len(query_xy), *values.shape[1:]), dtype=np.float64)
    for start in range(0, len(query_xy), batch_size):
        stop = min(start+batch_size, len(query_xy))
        output[start:stop] = np.einsum('bk,bkij->bij', weights[start:stop], values[indices[start:stop]])
    return output, dict(indices=indices, weights=weights)
