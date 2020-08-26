import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import dgl

class PSAGE(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, gnn_layers, dropout, activation,bias=True):
        super(PSAGE, self).__init__()

        # activation
        self.dropout = dropout
        self.activation = activation


        self.num_hidden = num_hidden
        self.n_classes = n_classes
        self.num_features = num_features
        
        # input layer
        self.gnn_input = SAGEConv(self.num_features, self.num_hidden[0], normalize=False, bias=bias)

        # Hidden layers
        self.layers = nn.ModuleList()
        for l in range(1,gnn_layers-1):
            self.layers.append(SAGEConv(self.num_hidden[l-1], self.num_hidden[l], normalize=False, bias=bias))

        # output layer
        self.gnn_output = SAGEConv(self.num_hidden[-2], self.n_classes, normalize=False, bias=bias)
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
