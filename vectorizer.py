"""
vectorizer.py  --  text -> graph of pen-strokes (outline form).

WHY THIS SHAPE
--------------
A font glyph is a *filled outline*: each letter is one or more closed boundary
contours (the silhouette, plus a separate contour for any counter/hole such as
the bowl of 'a' or 'e'). We turn each contour into a closed loop of nodes/edges.

Two things the old version got wrong are fixed here:

  1. It walked TextPath.vertices directly and added the quadratic/cubic Bezier
     *control* points (matplotlib codes CURVE3/CURVE4) as if they were real
     on-curve nodes. Those off-curve points do not lie on the letter, so they
     injected spikes. We use Path.to_polygons(), which *flattens* the Beziers
     into short straight segments that follow the true outline -- no spikes.

  2. The closed outline only stayed legible while perfectly undistorted; the old
     spring solver bent the loops into knots. The current alignment_solver only
     ever applies a global similarity transform (uniform scale + rotation +
     translation), so the outline is moved rigidly and the letterforms -- real
     'a' bowls included -- are preserved exactly.

Dependency-light: numpy + matplotlib only (already used by the project).

PUBLIC API
----------
    text_to_graph(text) -> (nodes: (N,2) float array, edges: (E,2) int array)
"""

import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


def text_to_graph(text):
    fp = FontProperties(family="sans-serif", weight="bold")
    path = TextPath((0, 0), text, size=10, prop=fp)

    # to_polygons() returns one vertex ring per contour with all Bezier curves
    # already flattened to straight segments -- exactly the on-letter outline.
    polys = path.to_polygons(closed_only=True)

    nodes = []
    edges = []
    for poly in polys:
        poly = np.asarray(poly, dtype=float)
        # the ring repeats its first point at the end to close it -- drop the dupe
        if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        if len(poly) < 2:
            continue
        base = len(nodes)
        n = len(poly)
        for k in range(n):
            nodes.append(poly[k])
            edges.append([base + k, base + (k + 1) % n])  # close the loop

    nodes = np.asarray(nodes, dtype=float)
    nodes -= nodes.mean(axis=0)                 # center on origin
    return nodes, np.asarray(edges, dtype=int)


if __name__ == "__main__":
    n, e = text_to_graph("Suhanraj")
    print(f"outline graph: {len(n)} nodes, {len(e)} edges")
