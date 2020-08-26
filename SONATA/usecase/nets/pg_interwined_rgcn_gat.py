import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GATConv
import copy


class PIRGAT(torch.nn.Module):
    """
    """
    def __init__(self, num_features, n_classes, num_heads, num_rels, num_bases,
                 num_hidden, num_hidden_layers_rgcn, num_hidden_layers_gat,
                 dropout, activation, neg_slope, bias=True):
        super(PIRGAT, self).__init__()

        self.concat = False
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
        self.rgcn_input = RGCNConv(num_features, num_hidden[0], num_rels,
                                   num_bases, bias=bias)
        # aggr values ['add', 'mean', 'max'] default : add
        # RGCN Hidden layers
        self.layers_rgcn = nn.ModuleList()
        for l in range(1, num_hidden_layers_rgcn-1):
            self.layers_rgcn.append(RGCNConv(num_hidden[l-1], num_hidden[l], num_rels,
                                             num_bases, bias=bias))

        # RGCN Output layer
        self.rgcn_output = RGCNConv(num_hidden[l], n_classes, num_rels,
                                    num_bases, bias=bias)
        
        # GAT layers
        self.layers_gat = nn.ModuleList()
        # GAT Input Layer
        self.gat_input = GATConv(num_features, num_hidden[0], heads=num_heads[0],
                                 concat=self.concat,
                                 negative_slope=self.neg_slope,
                                 dropout=dropout, bias=bias)

        # GAT Hidden layers
        for l in range(1,num_hidden_layers_gat-1):
            if self.concat:
                self.layers_gat.append(GATConv(num_hidden[l-1]*num_heads[l],
                                               num_hidden[l],
                                               heads=num_heads[l],
                                               concat=self.concat,
                                               negative_slope=self.neg_slope,
                                               dropout=dropout, bias=bias))
            else:
                self.layers_gat.append(GATConv(num_hidden[l-1], num_hidden[l],
                                               heads=num_heads[l],
                                               concat=self.concat,
                                               negative_slope=self.neg_slope,
                                               dropout=dropout, bias=bias))

        # GAT output layer
        
        if self.concat:
            self.gat_output = GATConv(num_hidden[l]*num_heads[l-1],
                                      n_classes,
                                      heads=num_heads[-1],
                                      concat=self.concat,
                                      negative_slope=self.neg_slope,
                                      dropout=dropout, bias=bias)
        else:
            self.gat_output = GATConv(num_hidden[l], n_classes,
                                      heads=num_heads[-1],
                                      concat=self.concat,
                                      negative_slope=self.neg_slope,
                                      dropout=dropout,
                                      bias=bias)

    def forward(self, data, g):
        x_1, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x_2 = copy.deepcopy(x_1)
        x_1 = self.rgcn_input(x_1, edge_index, edge_type)
        x_1 = self.activation(x_1)
        x_1 = self.dropout(x_1)
        x_2 = self.gat_input(x_2, edge_index)
        x_2 = self.activation(x_2)
        x_2 = self.dropout(x_2)
        # print(x.shape)
        for i in range(self.num_hidden_layers_rgcn-2):
            x_1 = self.layers_rgcn[i](x_1, edge_index, edge_type)
            x_1 = self.activation(x_1)
            x_1 = self.dropout(x_1)
            x_2 = self.layers_gat[i](x_2, edge_index)
            x_2 = self.activation(x_2)
            x_2 = self.dropout(x_2)
            # Swapping output of intermediate layers
            x_1, x_2 = x_2, x_1
        
        x_1 = self.rgcn_output(x_1, edge_index, edge_type)
        x_2 = self.gat_output(x_2, edge_index)
        x_2 = torch.mean(x_2, 1, keepdim=True, out=None)
        # This could be changed Addition just for prototype
        logits = x_1 + x_2
        logits = torch.tanh(logits)
        return logits
