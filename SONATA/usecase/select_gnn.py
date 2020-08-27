import torch
import sys
sys.path.append('./usecase/nets')
import torch.nn as nn
import torch.nn.functional as F
from tagcn import PTAG
from chebconv import PCHEB
from sagegcn import PSAGE
from gat_mcUpscale import GATMC
from pg_gat import PGAT
from pg_gcn import PGCN
from rgcnDGL import RGCN
from pg_rgcn import PRGCN
from pg_rgcn_gat import PRGAT
from pg_rgcn_gat2 import PRGAT2
from pg_rgcn_gat3 import PRGAT3
from pg_parallel_rgcn_gat import PPRGAT
from pg_interwined_rgcn_gat import PIRGAT
import dgl

class SELECT_GNN(torch.nn.Module):
    def __init__(self, num_features, n_classes, num_hidden, gnn_layers, dropout, activation, num_channels, gnn_type, K, num_heads,num_rels, num_bases,g, residual,attn_drop,num_hidden_layers_rgcn,num_hidden_layers_gat,num_hidden_layer_pairs,improved=True, concat=True, neg_slope=0.2,bias=True):
        super(SELECT_GNN, self).__init__()

        self.activation = activation
        self.num_hidden_layer_pairs = num_hidden_layer_pairs
        self.attn_drop = attn_drop
        self.num_hidden_layers_rgcn = num_hidden_layers_rgcn
        self.num_hidden_layers_gat = num_hidden_layers_gat
        self.num_rels = num_rels
        self.residual = residual
        self.num_bases = num_bases
        self.num_channels = num_channels
        self.n_classes = n_classes
        self.num_hidden = num_hidden
        self.gnn_layers = gnn_layers
        self.num_features = num_features
        self.dropout = dropout
        self.bias = bias
        self.improved = improved
        self.K = K
        self.g = g
        self.num_heads = num_heads
        self.concat = concat
        self.neg_slope = neg_slope
        self.dropout1 = dropout
        self.gnn_type = gnn_type

        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        #
        if self.gnn_type == 'ptag':
            print("GNN being used is PTAG")
            self.gnn_object = self.ptag()
        elif self.gnn_type == 'pcheb':
            print("GNN being used is PCHEB")
            self.gnn_object = self.pcheb()
        elif self.gnn_type == 'psage':
            print("GNN being used is PSAGE")
            self.gnn_object = self.psage()
        elif self.gnn_type == 'pgcn':
            print("GNN being used is PGCN")
            self.gnn_object = self.pgcn()
        elif self.gnn_type == 'pgat':
            print("GNN being used is PGAT")
            self.gnn_object = self.pgat()
        elif self.gnn_type == 'prgcn':
            print("GNN being used is PRGCN")
            self.gnn_object = self.prgcn()
        elif self.gnn_type == 'rgcn':
            print("GNN being used is RGCN")
            self.gnn_object = self.rgcn()
        elif self.gnn_type == 'gatmc':
            print("GNN being used is GATMC")
            self.gnn_object = self.gatmc()
        elif self.gnn_type == 'prgat':
            print("GNN being used is PRGAT")
            self.gnn_object = self.prgat()
        elif self.gnn_type == 'prgat2':
            print("GNN being used is PRGAT2")
            self.gnn_object = self.prgat2()
        elif self.gnn_type == 'prgat3':
            print("GNN being used is PRGAT2")
            self.gnn_object = self.prgat3()
        elif self.gnn_type == 'pprgat':
            print("GNN being used is PPRGAT")
            self.gnn_object = self.pprgat()
        elif self.gnn_type == 'pirgat':
            print("GNN being used is PIRGAT")
            self.gnn_object = self.pirgat()

    def ptag(self):
        return PTAG(self.num_features, self.n_classes, self.num_hidden, self.gnn_layers, self.dropout, self.activation)

    def pcheb(self):
        return PCHEB(self.num_features, self.n_classes, self.num_hidden, self.gnn_layers, self.K, self.dropout, self.activation)

    def psage(self):
        return PSAGE(self.num_features, self.n_classes, self.num_hidden, self.gnn_layers, self.dropout, self.activation)

    def pgat(self):
        return PGAT(self.num_features, self.n_classes, self.num_heads, self.num_hidden, self.gnn_layers, self.dropout1, self.dropout, self.activation)

    def pgcn(self):
        return PGCN(self.num_features, self.n_classes, self.num_hidden, self.gnn_layers, self.dropout, self.activation)

    def prgcn(self):
        return PRGCN(self.num_features, self.n_classes, self.num_rels, self.num_bases, self.num_hidden, self.gnn_layers, self.dropout1, self.activation)

    def rgcn(self):
        return RGCN(self.g, self.gnn_layers, self.num_features, self.num_hidden, self.num_rels, self.activation, self.dropout1)

    def gatmc(self):
        return GATMC(self.g, self.gnn_layers, self.num_features, self.num_hidden, self.num_heads, self.activation, self.dropout1, self.attn_drop, self.neg_slope, self.residual)

    def prgat(self):
        return PRGAT(self.num_features, self.n_classes, self.num_heads, self.num_rels, self.num_bases, self.num_hidden, self.num_hidden_layers_rgcn, self.num_hidden_layers_gat, self.dropout1, self.activation, self.neg_slope)

    def prgat2(self):
        return PRGAT2(self.num_features, self.n_classes, self.num_heads, self.num_rels, self.num_bases, self.num_hidden, self.num_hidden_layer_pairs, self.dropout1, self.activation, self.neg_slope)

    def prgat3(self):
        return PRGAT3(self.num_features, self.n_classes, self.num_heads, self.num_rels, self.num_bases, self.num_hidden, self.num_hidden_layer_pairs, self.dropout1, self.activation, self.neg_slope)

    def pprgat(self):
        return PPRGAT(self.num_features, self.n_classes, self.num_heads, self.num_rels, self.num_bases, self.num_hidden, self.num_hidden_layers_rgcn, self.num_hidden_layers_gat, self.dropout1, self.activation, self.neg_slope)

    def pirgat(self):
        return PIRGAT(self.num_features, self.n_classes, self.num_heads, self.num_rels, self.num_bases, self.num_hidden, self.num_hidden_layers_rgcn, self.num_hidden_layers_gat, self.dropout1, self.activation, self.neg_slope)

    def forward(self, data, g):
        if self.gnn_type in ['gatmc', 'prgat2', 'prgat3']:
            x = self.gnn_object(data)
        else:
            x = self.gnn_object(data, g)
        logits = x
        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        output = torch.Tensor(size=(len(unbatched), 3))
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            output[batch_number, :] = logits[base_index, :] # Output is just the room's node
            # output[batch_number, :] = logits[base_index:base_index+num_nodes, :].mean(dim=0) # Output is the average of all nodes
            base_index += num_nodes
            batch_number += 1
        return output
