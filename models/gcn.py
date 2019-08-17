"""GCN using builtin functions that enables SPMV optimization.

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import math
import torch
import torch.nn as nn
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 freeze=0):
        super(GCNLayer, self).__init__()
        self.g = g
        self.freeze = 0
        self.weight = nn.Parameter(torch.Tensor(in_feats+freeze, out_feats))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(torch.cat((self.g.ndata['frozen'], h), 1), self.weight)
        # normalization by square root of src degree
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * self.g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 freeze=0):
        super(GCN, self).__init__()
        self.g = g
        self.freeze = freeze
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, 0., freeze=0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(g, n_hidden, n_hidden, activation, dropout, freeze=self.freeze))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, torch.tanh, dropout, freeze=self.freeze))


    def forward(self, features):
        # normalization
        degs = self.g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(device=features.device)
        self.g.ndata['norm'] = norm.unsqueeze(1)

        self.g.ndata['frozen'] = features[:, :self.freeze]

        h = features
        for layer in self.layers:
            h = layer(h)
        return h
