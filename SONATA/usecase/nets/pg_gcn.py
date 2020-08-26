import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import dgl

class PGCN(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, gnn_layers, dropout, activation,improved=True, bias=True):
        super(PGCN, self).__init__()
        # activation
        self.dropout = dropout
        self.activation = activation

        self.gnn_input = GCNConv(num_features, num_hidden[0], improved=improved, bias=bias)
        # Hidden layers
        self.layers = nn.ModuleList()
        for l in range(1,gnn_layers-1):
            self.layers.append(GCNConv(num_hidden[l-1], num_hidden[l], improved=improved, bias=bias))

        # output layer
        self.gnn_output = GCNConv(num_hidden[-2], n_classes, improved=improved, bias=bias)

    def forward(self, data, g):

        self.x, edge_index = data.x, data.edge_index
        self.x = self.dropout(self.x)
        self.x = self.gnn_input(self.x, edge_index)
        self.x = self.activation(self.x)

        for layer in  self.layers:
            self.x = layer(self.x, edge_index)
            self.x = self.activation(self.x)
            self.x = self.dropout(self.x)

        self.x = self.gnn_output(self.x, edge_index)
        self.x = torch.tanh(self.x)
        return self.x
