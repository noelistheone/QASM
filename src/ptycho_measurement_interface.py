"""Identity-preserving input geometry for a future common ptychography task.

This module does not load reference objects, fit models, or choose experiments.
Functions accept explicit visible IDs; query measurements cannot enter a group
or a normalization call implicitly. It is separate from frozen M1 operators.
"""
import hashlib

import numpy as np
from scipy.spatial import cKDTree


def validated_ids(ids, count):
    values = np.asarray(ids)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Frame IDs must be a one-dimensional integer array')
    if len(values) == 0 or len(np.unique(values)) != len(values):
        raise ValueError('Frame IDs must be nonempty and distinct')
    if values.min() < 0 or values.max() >= count:
        raise ValueError('Frame IDs are outside the input array')
    return values.astype(np.int64, copy=False)


def coordinate_partition(coordinates, scan_name, seed=20261224,
                         query_fraction=.2, visible_fractions=(.1, .25, .5)):
    """Nested coordinate-group partitions, independent of detector values.

    Exactly coincident positions remain in the same partition. Fractions refer
    to coordinate groups, with realized frame budgets returned separately.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError('Expected finite (frames,2) pixel coordinates')
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    fractions = np.asarray(visible_fractions, dtype=float)
    if not (0 < query_fraction < 1) or not np.all((fractions > 0) & (fractions <= 1-query_fraction)):
        raise ValueError('Visible and query fractions must fit into the scan')
    if len(fractions) == 0 or not np.all(np.diff(fractions) > 0):
        raise ValueError('Visible fractions must be strictly increasing')
    name_seed = int.from_bytes(hashlib.sha256(scan_name.encode('utf8')).digest()[:8], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed, name_seed]))
    order = rng.permutation(len(unique))
    nquery = max(1, int(np.floor(query_fraction*len(unique))))
    query_groups, available = order[:nquery], order[nquery:]
    if len(available) < 4:
        raise ValueError('Not enough distinct available coordinates')
    query = np.flatnonzero(np.isin(inverse, query_groups)).astype(np.int64)
    visible = {}
    for fraction in fractions:
        nvisible = max(4, int(np.floor(float(fraction)*len(unique))))
        if nvisible > len(available):
            raise ValueError('Requested budget leaves insufficient query coordinates')
        visible[f'{fraction:g}'] = np.flatnonzero(np.isin(inverse, available[:nvisible])).astype(np.int64)
    return dict(query_ids=query, visible_ids=visible, unique_coordinate_count=len(unique),
                coordinate_group_order=order, coordinate_group_index=inverse)


def visible_quadruplets(coordinates, visible_ids, center_ids=None):
    """Build four distinct source-style channels using permitted frames only.

    Prefer one closest point per disjoint quadrant; missing quadrants are filled
    from remaining closest points. This deterministic adapter does not reproduce
    the source's random grouping. Original IDs and every fallback are retained.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError('Expected finite (frames,2) pixel coordinates')
    ids = validated_ids(visible_ids, len(coordinates))
    if len(ids) < 4:
        raise ValueError('Four distinct measured frames are required')
    centers = ids if center_ids is None else validated_ids(center_ids, len(coordinates))
    if not np.isin(centers, ids).all():
        raise ValueError('Group centers must also be visible measurements')
    points = coordinates[ids]
    tree = cKDTree(points)
    # At most 128 initial neighbors. Expand to all points only when quadrants
    # remain absent; this changes efficiency, never the selected nearest IDs.
    candidate_count = min(128, len(ids))
    distances, locations = tree.query(coordinates[centers], k=candidate_count)
    groups, fallback_counts = [], []
    for index, center in enumerate(centers):
        candidates = ids[locations[index]]
        delta = coordinates[candidates]-coordinates[center]
        quadrant = (delta[:, 0] >= 0).astype(np.int64)+2*(delta[:, 1] < 0)
        if len(np.unique(quadrant)) < 4 and candidate_count < len(ids):
            candidates = ids
            delta = coordinates[candidates]-coordinates[center]
            quadrant = (delta[:, 0] >= 0).astype(np.int64)+2*(delta[:, 1] < 0)
        # Include every exact-distance tie at the search boundary, so original
        # ID tie-breaking does not depend on cKDTree's internal ordering.
        elif candidate_count < len(ids):
            radius = np.nextafter(float(distances[index, -1]), np.inf)
            candidates = ids[np.asarray(tree.query_ball_point(coordinates[center], radius), dtype=np.int64)]
            delta = coordinates[candidates]-coordinates[center]
            quadrant = (delta[:, 0] >= 0).astype(np.int64)+2*(delta[:, 1] < 0)
        distance2 = np.sum(delta*delta, axis=1)
        ordering = np.lexsort((candidates, distance2))
        chosen = np.full(4, -1, dtype=np.int64)
        for q in range(4):
            ordered = ordering[quadrant[ordering] == q]
            if len(ordered):
                chosen[q] = candidates[ordered[0]]
        missing = np.flatnonzero(chosen < 0)
        remaining = [int(candidates[j]) for j in ordering if candidates[j] not in chosen]
        chosen[missing] = remaining[:len(missing)]
        assert len(np.unique(chosen)) == 4 and np.isin(chosen, ids).all()
        groups.append(chosen)
        fallback_counts.append(len(missing))
    groups = np.asarray(groups, dtype=np.int64)
    radius = np.linalg.norm(coordinates[groups]-coordinates[centers, None, :], axis=-1)
    return dict(center_ids=centers.copy(), frame_ids=groups,
                relative_coordinates=coordinates[groups]-coordinates[centers, None, :],
                radii=radius, fallback_count=np.asarray(fallback_counts, dtype=np.int64))


def intensity_scales(visible_intensity, source_rounding=False):
    """Return input scales from an explicit visible-only array, in float64.

    Rounding is a source-baseline option, never a mutation of measured targets.
    """
    value = np.asarray(visible_intensity, dtype=np.float64)
    if value.ndim != 3 or min(value.shape) == 0 or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Expected nonnegative finite visible detector frames')
    if source_rounding:
        value = np.rint(value)
    mean_sum = np.mean(np.sum(value, axis=(-2, -1)))
    mean_square = np.mean(value*value)
    if mean_sum <= 0 or mean_square <= 0:
        raise ValueError('All-dark input cannot determine a multiplicative scale')
    return dict(rms_scale=float(1/np.sqrt(mean_square)),
                inverse_mean_fluence=float(1/mean_sum),
                frames=len(value), source_rounding=bool(source_rounding))
