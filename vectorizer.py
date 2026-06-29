import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

def text_to_graph(text, step_size=0.15):
    fp = FontProperties(family='sans-serif', weight='bold')
    path = TextPath((0, 0), text, size=10, prop=fp)
    
    vertices = path.vertices
    codes = path.codes
    
    nodes = []
    edges = []
    
    current_node_idx = 0
    last_point = None
    
    def dist(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
    for i, (vertex, code) in enumerate(zip(vertices, codes)):
        if code == 1: 
            nodes.append(vertex)
            last_point = vertex
            current_node_idx = len(nodes) - 1
        elif code in [2, 3, 4]: 
            if last_point is None or dist(last_point, vertex) > step_size:
                nodes.append(vertex)
                new_idx = len(nodes) - 1
                edges.append([int(current_node_idx), int(new_idx)])
                current_node_idx = new_idx
                last_point = vertex

    nodes = np.array(nodes)
    centroid = np.mean(nodes, axis=0)
    nodes -= centroid
    
    return nodes, np.array(edges, dtype=int)