import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GATConv


class PRGAT(torch.nn.Module):
    def __init__(self, num_features, n_classes,num_heads ,num_rels, num_bases, num_hidden, num_hidden_layers_rgcn,num_hidden_layers_gat, dropout, activation, alpha, bias):
        super(PRGAT, self).__init__()
        self.concat = True
        self.neg_slope = alpha
        self.num_hidden_layers_rgcn = num_hidden_layers_rgcn
        self.num_hidden_layers_gat = num_hidden_layers_gat
        # dropout
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        # activation
        self.activation = activation
        # RGCN input layer
        self.rgcn_input = RGCNConv(num_features, num_hidden, num_rels, num_bases, bias=bias) #aggr values ['add', 'mean', 'max'] default : add
        # RGCN Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers_rgcn):
            self.layers.append(RGCNConv(num_hidden, num_hidden, num_rels, num_bases, bias=bias))
        # GAT input layer
        self.layers.append(GATConv(num_hidden, num_hidden, heads=num_heads, concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias))

        # GAT Hidden layers
        for _ in range(num_hidden_layers_gat):
            if self.concat:
                self.layers.append(GATConv(num_hidden*num_heads, num_hidden, heads=num_heads, concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias))
            else:
                self.layers.append(GATConv(num_hidden, num_hidden, heads=num_heads, concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias))
        # GAT output layer
        if self.concat:
            self.gat_output = GATConv(num_hidden*num_heads, n_classes, heads=num_heads, concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias)
        else:
            self.gat_output = GATConv(num_hidden, n_classes, heads=num_heads, concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias)


    def forward(self, data):
        x , edge_index, edge_type = data.x , data.edge_index, data.edge_type
        x = self.rgcn_input(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.dropout(x)
        #print(x.shape)
        for i in range(self.num_hidden_layers_rgcn):
            x = self.layers[i](x, edge_index, edge_type)
            x = self.activation(x)
            x = self.dropout(x)
        for i in range(self.num_hidden_layers_gat+1):
            x = self.layers[i+self.num_hidden_layers_rgcn](x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.gat_output(x, edge_index)
        x = x.mean(1).unsqueeze(dim=1)
        return torch.tanh(x)
# model = PRGAT(20,20,6,12,12,142,3,3,0.0001,F.relu,0.12,True)
# model.save("model1.pt")
