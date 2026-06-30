"""
gnn_engine.py  --  pick the star region to place the word in.

WHY THE OLD GNN IS GONE
-----------------------
The previous version built a ConstellationGNN whose weights were random and
never trained, then took argmax over word/sky embedding similarity. With random
weights that score is noise, so the "anchor" was effectively a random star --
usually out in the sparse galactic outskirts, where there are almost no stars
near the word. That is exactly why the rendered word floated in empty black
space (and why it was different on every run).

This replaces it with a real, deterministic criterion that directly optimizes
the thing we actually want -- "the word should land on lots of real stars":

  1. score every star by local DENSITY (how many neighbours sit within a small
     radius), and keep only candidates dense enough to host the whole word;
  2. for a handful of the densest candidates, run the actual shape-preserving
     fit and measure its residual (MSE);
  3. return the candidate with the lowest MSE -- the region the word fits best.

PUBLIC API
----------
    find_best_anchor(word_nodes, word_edges, sky_nodes)
        -> (best_index, anchor_xy, best_mse)
"""

import numpy as np
from scipy.spatial import KDTree

from alignment_solver import get_best_alignment


def find_best_anchor(word_nodes, word_edges, sky_nodes,
                     n_candidates=12, density_radius=40.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = sky_nodes[:, :2]
    tree = KDTree(xy)

    # 1. local density of every star
    density = tree.query_ball_point(xy, r=density_radius, return_length=True)

    # 2. candidate pool = stars dense enough to comfortably host the word
    need = max(50, len(word_nodes))
    pool = np.where(density >= need)[0]
    if len(pool) == 0:                       # fallback: just take the densest stars
        pool = np.argsort(density)[::-1][:200]

    pool = pool[np.argsort(density[pool])[::-1]]      # densest first
    pool = pool[: max(n_candidates * 8, 100)]         # restrict to the dense head
    cand = rng.choice(pool, size=min(n_candidates, len(pool)), replace=False)

    # 3. keep the candidate whose actual fit residual is smallest
    best = None
    for idx in cand:
        center = xy[idx]
        _, _, mse = get_best_alignment(word_nodes, word_edges, sky_nodes, center)
        if best is None or mse < best[0]:
            best = (mse, int(idx), center)

    best_mse, best_idx, anchor_xy = best
    return best_idx, anchor_xy, best_mse
