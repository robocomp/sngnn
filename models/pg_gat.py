import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class PGAT(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_heads, dropout, num_hidden, num_hidden_layers, activation, concat=True, neg_slope=0.2, bias=True):
        super(PGAT, self).__init__()
        # Activation
        self.activation = activation
        # input layer
        self.gat_input = GATConv(num_features, num_hidden, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            if concat:
                self.layers.append(GATConv(num_hidden*num_heads, num_hidden, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias))
            else:
                self.layers.append(GATConv(num_hidden, num_hidden, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias))
        # output layer
        if concat:
            self.gat_output = GATConv(num_hidden*num_heads, n_classes, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)
        else:
            self.gat_output = GATConv(num_hidden, n_classes, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)



    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat_input(x, edge_index)
        x = self.activation(x)
        for layer in  self.layers:
            x = layer(x, edge_index)
            x = self.activation(x)
        x = self.gat_output(x, edge_index)
        x = x.mean(1).unsqueeze(dim=1)
        return torch.tanh(x)
