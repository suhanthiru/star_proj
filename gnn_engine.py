import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

class ConstellationGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConstellationGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def create_pyg_data(nodes, edges):
    if nodes.shape[1] == 2:
        nodes = np.pad(nodes, ((0,0), (0,1)), mode='constant')
    x = torch.tensor(nodes, dtype=torch.float)
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    return Data(x=x, edge_index=edge_index)