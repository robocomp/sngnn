import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv

class GGN(torch.nn.Module):
    def __init__(self, num_features, num_layers, aggr='mean', bias=True):
        super(GGN, self).__init__()
        self.gnn = GatedGraphConv(num_features, num_layers, aggr=aggr, bias=bias) #aggr values ['add', 'mean', 'max'] default : add


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gnn(x, edge_index)
        x = x.mean(1).unsqueeze(dim=1)
        return torch.tanh(x)
