import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PGCN(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, num_hidden_layers, dropout, activation, improved=True, bias=True):
        super(PGCN, self).__init__()
        # dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        #activation
        self.activation = activation
        # input layer
        self.conv_input = GCNConv(num_features, num_hidden, improved=improved, bias=bias)
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(GCNConv(num_hidden, num_hidden, improved=improved, bias=bias))
        # output layer
        self.conv_output = GCNConv(num_hidden, n_classes, improved=improved, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(x)
        x = self.conv_input(x, edge_index)
        x = self.activation(x)
        for layer in self.layers:
             x = layer(x, edge_index)
             x = self.activation(x)
             x = self.dropout(x)
        x = self.conv_output(x, edge_index)
        return torch.tanh(x)
