import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import dgl.function as fn
import dgl
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from select_gnn_for_rl import SELECT_GNN


def activation_functions(activation_tuple_src):
    ret = []
    for x in activation_tuple_src:
        if x == 'relu':
            ret.append(F.relu)
        elif x == 'elu':
            ret.append(F.elu)
        elif x == 'tanh':
            ret.append(torch.tanh)
        elif x == 'leaky_relu':
            ret.append(F.leaky_relu)
        elif x is None:
            ret.append(None)
        else:
            print('Unknown activation function {}.'.format(x))
            sys.exit(-1)
    return tuple(ret)


def get_activation_function_class(activation_name):
    # add more activations
    if activation_name == "tanh":
        activation = nn.Tanh()
    elif activation_name == "relu":
        activation = nn.ReLU()
    elif activation_name == "leaky_relu":
        activation = nn.LeakyReLU()
    elif activation_name is None:
        activation = nn.Identity()
    else:
        print('Unknown activation function {}.'.format(activation_name))
        sys.exit(-1)

    return activation

def get_weight_init_class(weight_init_name):
    # add more weight init
    if weight_init_name == "xavier_uniform":
        weight_init = torch.nn.init.xavier_uniform_
    elif weight_init_name == "xavier_normal":
        weight_init = torch.nn.init.xavier_normal_

    return weight_init


class DuelingDQN(torch.nn.Module):
    def __init__(self, parameters):
        super(DuelingDQN, self).__init__()

        # API checks
        assert(parameters["gnn_layers"] == len(parameters["num_hidden"])+1)
        assert(parameters["gnn_layers"] == len(parameters["activation"])+1)
        assert(parameters["value_layers"]+1 == len(parameters["value_num_hidden"]))
        assert(parameters["value_layers"] == len(parameters["value_activation"]))
        assert(parameters["advantage_layers"]+1 == len(parameters["advantage_num_hidden"]))
        assert(parameters["advantage_layers"] == len(parameters["advantage_activation"]))

        # GNN last layer output dimension should match Value and Advantage first layer input dimension
        assert(parameters["n_classes"]*(1+len(parameters["central_grid_nodes"])) == parameters["value_num_hidden"][0])
        assert(parameters["n_classes"]*(1+len(parameters["central_grid_nodes"])) == parameters["advantage_num_hidden"][0])

        self.g = None
        self.gnn_activation = activation_functions(parameters["activation"])
        self.gnn_output = parameters["n_classes"]


        self.GNN = SELECT_GNN(
            num_features = parameters["num_features"], 
            num_edge_feats = parameters["num_edge_feats"], 
            n_classes = parameters["n_classes"], 
            num_hidden = parameters["num_hidden"], 
            gnn_layers = parameters["gnn_layers"], 
            dropout = parameters["dropout"],
            activation = self.gnn_activation, 
            final_activation = parameters["final_activation"], 
            gnn_type = parameters["gnn_type"], 
            num_heads = parameters["num_heads"], 
            num_rels = parameters["num_rels"], 
            num_bases = parameters["num_bases"], 
            g =  parameters["g"], 
            residual = parameters["residual"],
            aggregator_type = parameters["aggregator_type"], 
            attn_drop = parameters["attn_drop"], 
            concat = parameters["concat"], 
            bias = parameters["bias"], 
            norm = parameters["norm"], 
            alpha = parameters["alpha"],
            grid_nodes = parameters["grid_nodes"],
            central_grid_nodes = parameters["central_grid_nodes"]
            )


        print("VALUE NETWORK")
        self.value_layers = self.create_model(parameters["value_layers"], parameters["value_num_hidden"], parameters["value_activation"], parameters["value_weight_init"])

        print("ADVANTAGE NETWORK")
        self.advantage_layers = self.create_model(parameters["advantage_layers"], parameters["advantage_num_hidden"], parameters["advantage_activation"], parameters["advantage_weight_init"])


    def create_model(self, num_layers, layers, activation, weight_init):
        layers_ = []
        for i in range(num_layers):
            act = get_activation_function_class(activation[i])
            layers_ += [nn.Linear(layers[i], layers[i + 1])]
            init_weight = get_weight_init_class(weight_init[i])
            init_weight(layers_[-1].weight)
            layers_ += [act]

        return nn.Sequential(*layers_)

    def set_robot_node_indexes(self, indexes):
        self.GNN.set_robot_node_indexes(indexes)


    def forward(self, g, features, etypes):

        self.g = g
        self.GNN.g = g
        self.GNN.gnn_object.set_g(g)

        gnn_output = self.GNN.forward(features, etypes)
        
        value = self.value_layers(gnn_output)

        advantage = self.advantage_layers(gnn_output)

        advAverage = torch.mean(advantage, dim=1, keepdim=True)
        Q = value + advantage - advAverage

        return Q





class DuelingDQNRGCN(torch.nn.Module):
    def __init__(self, layers, in_dim, hidden_dimensions, num_rels, activation, weight_init, feat_drop, num_bases=-1):
        '''
        layers: gnn_layers, value_layers, advantage_layers
        hidden_dimensions: gnn, value, advantage,
        activation: gnn_activation, value_activation, advantage_activation
        '''
        super(DuelingDQNRGCN, self).__init__()

        # API checks
        assert(layers["gnn_layers"] == len(hidden_dimensions["gnn"]))
        assert(layers["gnn_layers"] == len(activation["gnn_activation"]))
        assert(layers["value_layers"]+1 == len(hidden_dimensions["value"]))
        assert(layers["value_layers"] == len(activation["value_activation"]))
        assert(layers["advantage_layers"]+1 == len(hidden_dimensions["advantage"]))
        assert(layers["advantage_layers"] == len(activation["advantage_activation"]))

        # GNN last layer output dimension should match Value and Advantage first layer input dimension
        assert(hidden_dimensions["gnn"][-1] == hidden_dimensions["value"][0])
        assert(hidden_dimensions["gnn"][-1] == hidden_dimensions["advantage"][0])

        self.g = None
        self.in_dim = in_dim

        self.gnn_num_layers = layers["gnn_layers"]
        self.value_num_layers = layers["value_layers"]
        self.advantage_num_layers = layers["advantage_layers"]

        self.gnn_hidden_dimensions = hidden_dimensions["gnn"]
        self.value_hidden_dimensions = hidden_dimensions["value"]
        self.advantage_hidden_dimensions = hidden_dimensions["advantage"]

        self.num_channels = hidden_dimensions["gnn"][-1]
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases

        self.gnn_activation = activation_functions(activation["gnn_activation"])
        self.value_activation = activation["value_activation"]
        self.advantage_activation = activation["advantage_activation"]

        self.value_weight_init = weight_init["value"]
        self.advantage_weight_init = weight_init["advantage"]

        self.gnn_output = hidden_dimensions["gnn"][-1]

        # create RGCN layers
        self.build_model()

    def build_model(self):
        self.gnn_layers = torch.nn.ModuleList()
        # input to hidden
        i2h = self.build_gnn_input_layer()
        self.gnn_layers.append(i2h)
        # hidden to hidden
        for i in range(self.gnn_num_layers-2):
            h2h = self.build_gnn_hidden_layer(i)
            self.gnn_layers.append(h2h)
        # hidden to output
        h2o = self.build_gnn_output_layer()
        self.gnn_layers.append(h2o)

        print("VALUE NETWORK")
        self.value_layers = self.create_model(self.value_num_layers, self.value_hidden_dimensions, self.value_activation, self.value_weight_init)

        print("ADVANTAGE NETWORK")
        self.advantage_layers = self.create_model(self.advantage_num_layers, self.advantage_hidden_dimensions, self.advantage_activation, self.advantage_weight_init)


    def build_gnn_input_layer(self):
        print(f'Building an INPUT layer for RGCN DeulingDQN of {self.in_dim}x{self.gnn_hidden_dimensions[0]} (activation:{self.gnn_activation[0]})')
        return RelGraphConv(self.in_dim, self.gnn_hidden_dimensions[0], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.gnn_activation[0])

    def build_gnn_hidden_layer(self, i):
        print(f'Building an HIDDEN layer for RGCN DeulingDQN of {self.gnn_hidden_dimensions[i]}x{self.gnn_hidden_dimensions[i+1]} (activation:{self.gnn_activation[i+1]})')
        return RelGraphConv(self.gnn_hidden_dimensions[i], self.gnn_hidden_dimensions[i+1], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.gnn_activation[i+1])

    def build_gnn_output_layer(self):
        print(f'Building an OUTPUT layer for RGCN DeulingDQN of {self.gnn_hidden_dimensions[-2]}x{self.gnn_hidden_dimensions[-1]} (activation:{self.gnn_activation[-1]})')
        return RelGraphConv(self.gnn_hidden_dimensions[-2], self.gnn_hidden_dimensions[-1], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.gnn_activation[-1])


    def create_model(self, num_layers, layers, activation, weight_init):
        layers_ = []
        for i in range(num_layers):
            act = get_activation_function_class(activation[i])
            layers_ += [nn.Linear(layers[i], layers[i + 1])]
            init_weight = get_weight_init_class(weight_init[i])
            init_weight(layers_[-1].weight)
            layers_ += [act]

        return nn.Sequential(*layers_)

    def forward(self, g, features, etypes):

        self.g = g

        h = features
        self.g.edata['norm'] = self.g.edata['norm'].to(device=features.device)

        for layer in self.gnn_layers:
            h = layer(self.g, h, etypes)

        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        gnn_output = torch.Tensor(size=(len(unbatched), self.gnn_output)).to(device=features.device)
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            gnn_output[batch_number, :] = h[base_index, :]  # Output is just the room's node
            # output[batch_number, :] = logits[base_index:base_index+num_nodes, :].mean(dim=0) # Output is the average of all nodes
            base_index += num_nodes
            batch_number += 1

        
        value = self.value_layers(gnn_output)

        advantage = self.advantage_layers(gnn_output)

        advAverage = torch.mean(advantage, dim=1, keepdim=True)
        Q = value + advantage - advAverage

        return Q
