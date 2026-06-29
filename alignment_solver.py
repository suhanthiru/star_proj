import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment

def rigid_icp(source, target, iterations=15):
    src = source.copy()
    for _ in range(iterations):
        tree = KDTree(target)
        _, indices = tree.query(src)
        matched_target = target[indices]
        
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(matched_target, axis=0)
        
        src_centered = src - centroid_src
        tgt_centered = matched_target - centroid_tgt
        
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
            
        src = (R @ src_centered.T).T + centroid_tgt
    return src

def get_best_alignment(word_nodes, word_edges, sky_nodes, center_point): 
    tree = KDTree(sky_nodes[:, :2])
    local_indices = tree.query_ball_point(center_point, r=250) 
    if len(local_indices) < len(word_nodes):
        local_indices = tree.query_ball_point(center_point, r=500)
    local_sky = sky_nodes[local_indices, :2]

    template = word_nodes[:, :2].copy()
    current_width = np.ptp(template[:, 0]) 
    sky_width = np.ptp(local_sky[:, 0])
    
    if current_width > 0:
        template = template * ((sky_width * 0.6) / current_width) 
        
    template = template - np.mean(template, axis=0) + center_point
    aligned_nodes = rigid_icp(template, local_sky, iterations=15)

    orig_lengths = []
    for e in word_edges:
        p1_idx, p2_idx = int(e[0]), int(e[1])
        orig_lengths.append(np.linalg.norm(aligned_nodes[p1_idx] - aligned_nodes[p2_idx]))
        
    points = aligned_nodes.copy()
    sky_tree = KDTree(local_sky)
    
    # The Titanium Springs
    lambda_star = 0.05 
    lambda_edge = 4.0  
    lambda_lap  = 3.0  
    lr = 0.1           

    for _ in range(80): 
        forces = np.zeros_like(points)
        _, nn_idx = sky_tree.query(points)
        nearest_stars = local_sky[nn_idx]
        forces += lambda_star * (nearest_stars - points)
        
        for idx, e in enumerate(word_edges):
            p1_idx, p2_idx = int(e[0]), int(e[1])
            p1, p2 = points[p1_idx], points[p2_idx]
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            if dist > 0.0001:
                dir_vec = vec / dist
                force_mag = (dist - orig_lengths[idx]) * lambda_edge
                forces[p1_idx] += force_mag * dir_vec
                forces[p2_idx] -= force_mag * dir_vec
                
        for i in range(len(points)):
            neighbors = []
            for e in word_edges:
                if int(e[0]) == i: neighbors.append(points[int(e[1])])
                elif int(e[1]) == i: neighbors.append(points[int(e[0])])
            if neighbors:
                centroid = np.mean(neighbors, axis=0)
                forces[i] += lambda_lap * (centroid - points[i])
                
        points += forces * lr

    cost_matrix = np.linalg.norm(points[:, np.newaxis, :] - local_sky[np.newaxis, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    final_nodes = np.copy(word_nodes)
    final_nodes[:, :2] = points 
    safe_snap_distance = np.mean(orig_lengths) * 2.5 
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < safe_snap_distance:
            final_nodes[r, :2] = local_sky[c]

    mse = np.mean(np.linalg.norm(final_nodes[:, :2] - template, axis=1))
    return points, final_nodes, mse