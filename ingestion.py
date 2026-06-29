import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import time

def build_sky_graph(csv_path="hygdata_v42.csv.gz", mag_limit=7.5, k_neighbors=6):
    df = pd.read_csv(csv_path)
    visible_stars = df[df['mag'] <= mag_limit].dropna(subset=['x', 'y', 'z']).copy()
    
    positions = visible_stars[['x', 'y', 'z']].to_numpy()
    star_ids = visible_stars['id'].to_numpy()
    
    sky_tree = KDTree(positions)
    distances, indices = sky_tree.query(positions, k=k_neighbors + 1)
    
    edges = []
    for i, neighbors in enumerate(indices):
        for neighbor_idx in neighbors[1:]:
            edges.append([int(i), int(neighbor_idx)])
            
    return positions, np.array(edges, dtype=int), star_ids