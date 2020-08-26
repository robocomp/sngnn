import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import dgl

class PRGCN(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_rels, num_bases, num_hidden, gnn_layers, dropout, activation,  bias=True):
        super(PRGCN, self).__init__()
        # dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        # activation
        self.activation = activation
        # input layer
        self.rgcn_input = RGCNConv(num_features, num_hidden[0], num_rels, num_bases, bias=bias) #aggr values ['add', 'mean', 'max'] default : add
        # Hidden layers
        self.layers = nn.ModuleList()
        for l in range(1,gnn_layers-1):
            self.layers.append(RGCNConv(num_hidden[l-1], num_hidden[l], num_rels, num_bases, bias=bias))
        # output layer
        self.rgcn_output = RGCNConv(num_hidden[-2], n_classes, num_rels, num_bases, bias=bias)

    def forward(self, data, g):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.rgcn_input(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.rgcn_output(x, edge_index, edge_type)
        x = torch.tanh(x)
        return x 
