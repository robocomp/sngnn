"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.conv.gatconv import GATConv
import dgl
import sys
import torch.nn.functional as F


class GATMC(nn.Module):
    def __init__(self, g, gnn_layers, in_dim, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(GATMC, self).__init__()
        # print('hidden', num_hidden[-1], 'heads', heads,  'imgw', image_width, 'chann')
        # assert (heads[-1] == 1), 'The number of output heads must be the number of expected channels'
        assert (len(heads) == gnn_layers), 'The number of elements in the heads list must be the number of layers'
        assert (len(num_hidden) == gnn_layers), 'The number of elements in the num_hidden list must be the number of layers'
        
        self.first = True
        self.g = g
        self.num_hidden = num_hidden
        self.gnn_layers = gnn_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        print('Heads {}'.format(heads))
        # input projection (no residual)

        print(in_dim, num_hidden[0], heads[0], '(0)')
        self.layers.append(GATConv(
            in_dim, num_hidden[0], heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        # hidden layers
        for l in range(1, gnn_layers-1):
            print(num_hidden[l-1] * heads[l-1], num_hidden[l], heads[l], '('+str(l)+')')
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                num_hidden[l-1] * heads[l-1], num_hidden[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

        print(num_hidden[-1] * heads[-1], num_hidden[-1], heads[-1], '(*)')
        self.layers.append(GATConv(
            num_hidden[-2] * heads[-2], num_hidden[-1], heads[-1],
            feat_drop, attn_drop, negative_slope, residual, torch.tanh)) #

    def set_g(self, g):
        self.g = g

    def forward(self, inputs):
        
        h = inputs
        if self.first:
            print('H: ', h.size())
        for l in range(self.gnn_layers-1):
            h = self.layers[l](self.g, h).flatten(1)
            if self.first:
                print('out {}: {}'.format(l, h.size()))
        # output projection
        if self.first:
            print(h.shape)
        logits = self.layers[-1](self.g, h)
        if self.first:
            print('OUT {}'.format(logits.size()))
        # return logits
        # return logits[getMaskForBatch(self.g)]

        # print logits[getMaskForBatch(self.g)]
        self.first = False
        logits = torch.mean(logits, 1, keepdim=True, out=None)
        logits = torch.tanh(logits)
       	return logits

    

