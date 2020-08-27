import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GATConv
import dgl

class PRGAT(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_heads, num_rels, num_bases, num_hidden, num_hidden_layers_rgcn,num_hidden_layers_gat, dropout, activation, neg_slope, bias=True):
        super(PRGAT, self).__init__()

        self.concat = True
        self.neg_slope = neg_slope
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
        self.rgcn_input = RGCNConv(num_features, num_hidden[0], num_rels, num_bases, bias=bias) #aggr values ['add', 'mean', 'max'] default : add
        # RGCN Hidden layers
        self.layers = nn.ModuleList()
        for l in range(1, num_hidden_layers_rgcn-1):
            self.layers.append(RGCNConv(num_hidden[l-1], num_hidden[l], num_rels, num_bases, bias=bias))

        
        # GAT input layer
        self.layers.append(GATConv(num_hidden[l], num_hidden[l+1], heads=num_heads[l+1], concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias))

        # GAT Hidden layers
        for ll in range(l+2, num_hidden_layers_rgcn + num_hidden_layers_gat-2):
            if self.concat:
                self.layers.append(GATConv(num_hidden[ll-1]*num_heads[ll-1], num_hidden[ll], heads=num_heads[ll], concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias))
            else:
                self.layers.append(GATConv(num_hidden[ll-1], num_hidden[ll], heads=num_heads[ll], concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias))
        # GAT output layer
        if self.concat:
            self.gat_output = GATConv(num_hidden[ll]*num_heads[ll], n_classes, heads=num_heads[ll+1], concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias)
        else:
            self.gat_output = GATConv(num_hidden[ll], n_classes, heads=num_heads[ll+1], concat= self.concat, negative_slope=self.neg_slope, dropout = dropout, bias=bias)


    def forward(self, data, g):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.rgcn_input(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.dropout(x)
        #print(x.shape)
        for i in range(self.num_hidden_layers_rgcn-2):
            x = self.layers[i](x, edge_index, edge_type)
            x = self.activation(x)
            x = self.dropout(x)

        for i in range(self.num_hidden_layers_gat-1):
            x = self.layers[i+self.num_hidden_layers_rgcn-2](x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.gat_output(x, edge_index)
        logits = x
        logits = torch.mean(logits, 1, keepdim=True, out=None)
        logits = torch.tanh(logits)
        return logits
