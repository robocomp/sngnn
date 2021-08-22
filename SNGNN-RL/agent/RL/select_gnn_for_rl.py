import torch
import torch.nn as nn
from nets.DGL.rgcnDGL import RGCN
from nets.DGL.gat import GAT
from nets.DGL.mpnn_dgl import MPNN
import dgl
import time

class SELECT_GNN(nn.Module):
    def __init__(self, num_features, num_edge_feats, n_classes, num_hidden, gnn_layers, dropout,
                 activation, final_activation, gnn_type, num_heads, num_rels, num_bases, g, residual,
                 aggregator_type, attn_drop, concat=True, bias=True, norm=None, alpha=0.12, grid_nodes = 0,
                 central_grid_nodes = []):
        super(SELECT_GNN, self).__init__()

        self.activation = activation
        self.gnn_type = gnn_type
        if final_activation == 'relu':
            self.final_activation = torch.nn.ReLU()
        elif final_activation == 'tanh':
            self.final_activation = torch.nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation == 'leaky_relu':
            self.final_activation = torch.nn.LeakyReLU()
        elif final_activation is None:
            self.final_activation = None
        else:
            print('Unknown final activation:', self.final_activation)
            import sys
            sys.exit(-1)

        self.attn_drop = attn_drop
        self.num_rels = num_rels
        self.residual = residual
        self.aggregator = aggregator_type
        self.num_bases = num_bases
        self.n_classes = n_classes
        self.num_hidden = num_hidden
        self.gnn_layers = gnn_layers
        self.num_features = num_features
        self.num_edge_feats = num_edge_feats
        self.dropout = dropout
        self.bias = bias
        self.norm = norm
        self.g = g
        self.num_heads = num_heads
        self.concat = concat
        self.alpha = alpha
        self.robot_node_indexes = []
        self.grid_nodes = grid_nodes
        self.central_grid_nodes = central_grid_nodes
        
        if self.gnn_type == 'rgcn':
            print("GNN being used is RGCN")
            self.gnn_object = self.rgcn()
        elif self.gnn_type == 'gat':
            print("GNN being used is GAT")
            self.gnn_object = self.gat()
        elif self.gnn_type == 'mpnn':
            print("GNN being used is MPNN")
            self.gnn_object = self.mpnn()

    def rgcn(self):
        return RGCN(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_rels,
                    self.activation, self.final_activation, self.dropout, self.num_bases)

    def gat(self):
        return GAT(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_heads,
                   self.activation, self.final_activation,  self.dropout, self.attn_drop, self.alpha, self.residual)

    def mpnn(self):
        return MPNN(self.num_features, self.n_classes, self.num_hidden, self.num_edge_feats, self.final_activation,
                    self.aggregator, self.bias, self.residual, self.norm, self.activation)

    def set_robot_node_indexes(self, indexes):
        self.robot_node_indexes = indexes

    def forward(self, data, efeat):

        if self.gnn_type == "rgcn":
            # efeat is etypes; data is node features
            x = self.gnn_object(data, efeat)
        elif self.gnn_type == "gat":
            # data is node features
            x = self.gnn_object(data)
        elif self.gnn_type == "mpnn":
            # data is node features; efeat is edge features
            x = self.gnn_object(data, efeat)

        if not self.robot_node_indexes:
            indexes = []
            n_nodes = 0
            unbatched = dgl.unbatch(self.g)    
            for g in unbatched:
                indexes.append(n_nodes+self.grid_nodes)
                n_nodes += g.number_of_nodes()
        else:
            indexes = self.robot_node_indexes


        logits = torch.squeeze(x, 1).to(device=data.device)
        output = logits[indexes].to(device=data.device)
        # print("filtering by indexes", output.shape)
        outputS = output.shape
        nfeats = (1+len(self.central_grid_nodes))*outputS[1]
        # print("nfeats", nfeats)
        newShape = [(outputS[0]*outputS[1])//nfeats, nfeats]
        # print("final shape", newShape)
        output = output.view(newShape)
        return output
        # logits = x
        # base_index = 0
        # batch_number = 0
        # unbatched = dgl.unbatch(self.g)
        # output = torch.Tensor(size=(len(unbatched), self.n_classes)).to(device=data.device)
        # for g in unbatched:
        #     num_nodes = g.number_of_nodes()
        #     output[batch_number, :] = logits[base_index, :]  # Output is just the room's node
        #     base_index += num_nodes
        #     batch_number += 1
        # print("gnn output", output.shape)
        # return output
