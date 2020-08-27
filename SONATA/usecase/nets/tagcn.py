import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
import dgl


class PTAG(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, gnn_layers,dropout,activation,bias=True):
        super(PTAG, self).__init__()        

        # activation
        self.dropout = dropout
        self.activation = activation
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.bias = bias
        self.n_classes = n_classes
        self.gnn_layers = gnn_layers

        # input layer
        self.gnn_input = TAGConv(self.num_features, self.num_hidden[0] , bias=self.bias)

        # Hidden layers
        self.layers = nn.ModuleList()
        for l in range(1,self.gnn_layers-1):
            self.layers.append(TAGConv(self.num_hidden[l-1] , self.num_hidden[l] , bias=self.bias))

        # output layer
        self. gnn_output = TAGConv(self.num_hidden[-2] , self.n_classes, bias=self.bias)
        

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
