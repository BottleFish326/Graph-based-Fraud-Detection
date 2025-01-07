import torch.nn as nn
from torch_geometric.nn import GATConv
import torch

class edgeGAT(nn.Module):
    """
    Edge-aware Graph Attention Network (GAT) for graph-based fraud detection.

    Parameters:
    in_channels (int): Number of input features per node.
    edge_channels (int): Number of input features per edge.
    out_channels (int): Number of output features per node.
    device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
    """
    def __init__(self, in_channels, edge_channels, out_channels, device):
        super(edgeGAT, self).__init__()
        self.device = device
        self.gat1 = GATConv(in_channels, 8, heads=8, dropout=0.3)
        self.gat2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.3)
        self.lin = nn.Linear(out_channels, out_channels) 
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_channels * 2 + edge_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, data):
        """
        Forward pass for the edgeGAT model.

        Parameters:
        data (Data): Input graph data containing node features, edge indices, and edge features.

        Returns:
        Tensor: Log-softmax output for edge classification.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.gat1(x.to(self.device), edge_index.to(self.device))
        x = nn.functional.elu(x)
        x = self.gat2(x, edge_index.to(self.device))
        x = nn.functional.elu(x)
        x = self.lin(x)
        src, dst = edge_index.to(self.device)
        edge_features = torch.cat([x[src], x[dst], edge_attr.to(self.device)], dim=1)
        edge_out = self.edge_mlp(edge_features)
        return nn.functional.log_softmax(edge_out, dim=1)