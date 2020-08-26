import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import dgl

class PCHEB(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, gnn_layers, K,dropout, activation, bias=True):
        super(PCHEB, self).__init__()

        # activation
        self.dropout = dropout
        self.activation = activation


        self.num_hidden = num_hidden
        self.n_classes = n_classes
        self.num_features = num_features
        self.K = K #Chebyshev filter size K.
        
        # input layer
        self.gnn_input = ChebConv(self.num_features, self.num_hidden[0],  self.K, normalization='sym',bias=bias)

        # Hidden layers
        self.layers = nn.ModuleList()
        for l in range(1,gnn_layers-1):
            self.layers.append(ChebConv(self.num_hidden[l-1], self.num_hidden[l],  self.K, normalization='sym',bias=bias))

        # output layer
        self.gnn_output = ChebConv(self.num_hidden[-2], self.n_classes,  self.K, normalization='sym',bias=bias)
        
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
