from copy import copy
import os
import sys
import time
import random
import collections

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl

from icecream import ic

sys.path.append(os.path.join(os.path.dirname(__file__), '../socnav/'))
from socnav import SocNavDataset, get_features, get_relations, grid_nodes_number, central_grid_nodes

sys.path.append(os.path.join(os.path.dirname(__file__), '../RL/DuelingDQN/'))
from duelingDQN import DuelingDQN


# gamma = 0.99
# batch_size = 30
# replay_start_size = 10 * 1000
# replay_size = 80 * 1000
# learning_rate = 1e-4
# sync_target_frames = 1000
# epsilon_decay_last_frame = 60000
# epsilon_start = 1.0
# epsilon_final = 0.02
# tau = 0.001

action_matrix = [
                 (-0.225, 0.450), (0.000,  0.450), (0.225, 0.450),
(-0.450, 0.225), (-0.225, 0.225), (0.000,  0.225), (0.225, 0.225), (0.450, 0.225),
(-0.450, 0.000), (-0.225, 0.000), (0.000,  0.000), (0.225, 0.000), (0.450, 0.000),
                                  (0.000, -0.100)
]
act_dim = len(action_matrix)
print(f'Action space size: {act_dim}')


class RLAgent(object):
    def __init__(self, environment):
        super(RLAgent, self).__init__()
        print('Building agent DuelingDQN_GNN')
        self.time_str = str(int(time.time()))
        self.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
        self.environment = environment
        self.graph_ALT = '8'

        self.view = None

        _ , n_features = get_features(self.graph_ALT)
        _ , n_relations = get_relations(self.graph_ALT)
        self.grid_nodes = grid_nodes_number(self.graph_ALT)
        self.central_nodes = central_grid_nodes(self.graph_ALT, 1500.)
        ncentral_nodes = len(self.central_nodes) + 1


        # Parameters for RGCN
        parameters = {"num_features": n_features, "num_edge_feats": None, "n_classes": 42, "num_hidden": [42, 42], "gnn_layers":3, "dropout":0,
                 "activation": ['leaky_relu', 'leaky_relu'], "final_activation": "leaky_relu", "value_layers": 2, "value_num_hidden": [ncentral_nodes*42, 30, 1], 
                 "value_activation": ["leaky_relu", None], "value_weight_init": ["xavier_uniform", "xavier_uniform"], "advantage_layers": 2, "advantage_num_hidden": [ncentral_nodes*42, 30, act_dim], 
                 "advantage_activation": ["leaky_relu", None], "advantage_weight_init": ["xavier_uniform", "xavier_uniform"], "gnn_type":"rgcn", 
                 "num_heads":None, "num_rels":n_relations, "num_bases":-1, "g":None, "residual": None, "aggregator_type":None, "attn_drop":None, "concat":True, "bias":True, "norm":None, 
                 "alpha":0.12, "grid_nodes":self.grid_nodes, "central_grid_nodes":self.central_nodes}

        self.net = DuelingDQN(parameters).to(self.device)

        self.epsilon_restart = 0

        print("NETWORK")
        print(self.net)


        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = nn.DataParallel(self.net)

        if self.environment.check_play():
            self.net.load_state_dict(torch.load(self.environment.check_play(), map_location=lambda storage, loc: storage))
            self.compute = self.compute_play
            print(f'Test mode. Using model {self.environment.check_play()}.')
        else:
            print("Please use this format: 'python3 agent.py --play=model_file'")
            exit()

    def set_app(self, app):
        self.app = app

    def _reset(self):
        self.environment.reset()
        temp_observation, _, _ = self.environment.step([0., 0., 0.])
        self.current_graph = temp_observation
        self.total_reward = 0


    @torch.no_grad()
    def play_step(self):
        if not hasattr(self, 'current_graph'):
            temp_observation, is_done, _ = self.environment.step([0., 0., 0.])
            self.current_graph = temp_observation 
        # if np.random.random() < self.epsilon:
        #     action_index = random.randrange(act_dim)
        # else:
        subgraph, feats, _, robot_node_indexes = self.collate(self.current_graph)
        subgraph = subgraph.to(self.device)
        feats = feats.to(self.device)
        self.net.set_robot_node_indexes(robot_node_indexes)
        q_vals_v = self.net(subgraph, feats.float(), subgraph.edata['rel_type'].squeeze().to(self.device))
        _, act_v = torch.max(q_vals_v, dim=1)
        action_index = int(act_v.item())

        action_command = [action_matrix[action_index][0], 0., action_matrix[action_index][1]]
        temp_observation, is_done, _ = self.environment.step(action_command)
        self.next_graph = temp_observation

        self.current_graph = self.next_graph

        if is_done:
            self._reset()

        return is_done

    def collate(self, sample):
        graph, labels = sample[0]
        graphs = [graph]
        central_nodes = copy(self.central_nodes)
        robot_node_indexes = [self.grid_nodes] + central_nodes
        n_nodes = graph.number_of_nodes()
        central_nodes = [n + n_nodes for n in central_nodes]
        if len(sample)>1:
            for graph, label in sample[1:]:
                robot_node_indexes.append(n_nodes+self.grid_nodes)
                robot_node_indexes += central_nodes
                graphs.append(graph)
                labels = torch.cat([labels, label], dim=0)
                n_nodes += graph.number_of_nodes()
                central_nodes = copy(self.central_nodes)
                central_nodes = [n + n_nodes for n in central_nodes]
        graphs = dgl.batch(graphs).to(self.device)
        feats = graphs.ndata['h']
        return graphs, feats, labels, robot_node_indexes

    def compute_play(self):
        while True:
            if self.app is not None:
                self.app.processEvents()

            is_done = self.play_step()
