import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import dgl

class PGAT(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_heads, num_hidden, GNN_layers, dropout, dropout1, activation, concat=True, neg_slope=0.2, bias=True):
        super(PGAT, self).__init__()

        # activation
        
        self.dropout = dropout
        self.dropout1 = dropout1
        self.activation = activation


        self.gnn_input = GATConv(num_features, num_hidden[0], heads=num_heads[0], concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)
        # Hidden layers
        self.layers = nn.ModuleList()
        for l in range(1,GNN_layers-1):
            if concat:
                self.layers.append(GATConv(num_hidden[l-1]*num_heads[l-1], num_hidden[l], heads=num_heads[l], concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias))
            else:
                self.layers.append(GATConv(num_hidden[l-1], num_hidden[l], heads=num_heads[l], concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias))
        # output layer for latent space
        if concat:
            self.gnn_output = GATConv(num_hidden[-2]*num_heads[-2], n_classes, heads=num_heads[-1], concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)
        else:
            self.gnn_output = GATConv(num_hidden[-2], n_classes, heads=num_heads[-1], concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)

    def forward(self, data, g):

        self.x, edge_index = data.x, data.edge_index
        self.x = self.dropout1(self.x)
        self.x = self.gnn_input(self.x, edge_index)
        self.x = self.activation(self.x)

        for layer in  self.layers:
            self.x = layer(self.x, edge_index)
            self.x = self.activation(self.x)
            self.x = self.dropout1(self.x)
        self.x = torch.mean(self.gnn_output(self.x, edge_index), 1, keepdim=True, out=None)
        self.x = torch.tanh(self.x)
        return self.x
