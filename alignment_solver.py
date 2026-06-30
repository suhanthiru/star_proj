"""
alignment_solver.py  --  fit the word onto the stars WITHOUT distorting it.

WHY THIS SHAPE
--------------
The old solver ran an 80-step spring relaxation (pull-to-nearest-star +
edge-length + Laplacian smoothing) followed by a Hungarian assignment that
teleported every vertex onto a distinct star. Stars are not arranged in letter
shapes, so that guaranteed a scrambled word: the Laplacian term acted as a
curve-shortening flow that collapsed letters into knots, and the teleport broke
the baseline and proportions.

Here the word is only ever moved by a single GLOBAL SIMILARITY transform
(uniform scale + rotation + translation) applied to every node at once. Because
all nodes share one transform, the letters mathematically cannot distort:
legibility is preserved by construction. The transform is chosen by
similarity-ICP -- repeatedly match each node to its nearest star and solve the
closed-form best similarity transform (Umeyama 1991) -- so the word sits as
nicely as possible among the local stars while staying rigid.

Star "snapping" survives only as an OPTIONAL, distance-capped cosmetic nudge: a
node moves onto a star only if a star is already within a fraction of the local
stroke spacing, so it can never move far enough to bend a letter.

PUBLIC API
----------
    get_best_alignment(word_nodes, word_edges, sky_nodes, center_point)
        -> (smooth_nodes, snapped_nodes, mse)

    smooth_nodes  -- the undistorted, fitted word; what the renderer draws.
    snapped_nodes -- smooth_nodes with the capped cosmetic snap (for star lookup).
    mse           -- mean residual to nearest stars (fit quality; smaller=better).
"""

import numpy as np
from scipy.spatial import KDTree


def _umeyama_similarity(src, dst):
    """Closed-form best similarity transform (scale s, rotation R, translation t)
    mapping src onto dst, minimizing sum ||s*R*src + t - dst||^2 (Umeyama 1991).
    The reflection guard keeps R a proper rotation, so letters are never mirrored."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    Sc = src - mu_s
    Dc = dst - mu_d
    cov = (Dc.T @ Sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def similarity_icp(source, target, iterations=20):
    """Iterative Closest Point restricted to SIMILARITY transforms.
    Returns the transformed source. Shape is preserved exactly every step."""
    src = source.copy()
    tree = KDTree(target)
    for _ in range(iterations):
        _, idx = tree.query(src)
        s, R, t = _umeyama_similarity(src, target[idx])
        src = (s * (R @ src.T).T) + t
    return src


def get_best_alignment(word_nodes, word_edges, sky_nodes, center_point,
                       patch_radius=120.0, snap_fraction=0.6):
    template = word_nodes[:, :2].astype(float).copy()
    xy = sky_nodes[:, :2]
    tree = KDTree(xy)

    # local star patch around the anchor; expand to nearest-k if it's too sparse
    local_idx = tree.query_ball_point(center_point, r=patch_radius)
    if len(local_idx) < max(8, len(template)):
        k = min(max(200, len(template) * 4), len(xy))
        _, local_idx = tree.query(center_point, k=k)
        local_idx = np.atleast_1d(local_idx)
    local = xy[local_idx]

    # scale the word to a sensible fraction of the patch, keep aspect ratio
    span_word = max(np.ptp(template[:, 0]), 1e-9)
    span_sky = np.ptp(local[:, 0]) if np.ptp(local[:, 0]) > 0 else 1.0
    template *= (span_sky * 0.6) / span_word
    template += center_point - template.mean(0)

    # the ONLY transform allowed: global similarity -> no per-letter distortion
    smooth = similarity_icp(template, local, iterations=20)

    # typical spacing between adjacent outline nodes (used to cap snapping)
    if len(word_edges):
        seg = np.linalg.norm(smooth[word_edges[:, 0]] - smooth[word_edges[:, 1]], axis=1)
        typical = np.median(seg)
    else:
        typical = np.ptp(smooth, axis=0).mean() / 20.0

    # residual to nearest star (fit quality) + optional capped cosmetic snap
    local_tree = KDTree(local)
    dist, nn = local_tree.query(smooth)
    snapped = smooth.copy()
    close = dist < snap_fraction * typical
    snapped[close] = local[nn[close]]
    mse = float(np.mean(dist))

    # return 3-col arrays (renderer uses [:, :2]) to match the pipeline signature
    smooth_out = np.column_stack([smooth, np.zeros(len(smooth))])
    snapped_out = np.column_stack([snapped, np.zeros(len(snapped))])
    return smooth_out, snapped_out, mse
