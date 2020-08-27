"""
This script generates the dataset.
"""

import sys
import os
import pickle
import json
import numpy as np
import copy
from collections import namedtuple
import math
import torch
import cv2
import random

from dgl import DGLGraph
import dgl

limit = 149999  # 31191

path_saves = 'saves/'


N_INTERVALS = 3
FRAMES_INTERVAL = 1.

MAX_ADV = 3.5
MAX_ROT = 4.
MAX_HUMANS = 15


class RoomGraph(DGLGraph):
    def __init__(self, data, alt, mode='train', debug=False, w_segments=[]):
        super(RoomGraph, self).__init__()

        self.labels = None
        self.features = None
        self.feature_dimensions = None
        self.num_rels = None
        self.rels = []
        self.all_features = []
        # We create a map to store the types of the nodes. We'll use it to compute edges' types
        self.typeMap = {}
        # Copy input data
        self.data = copy.deepcopy(data)
        self.alt = alt
        self.mode = mode
        self.debug = debug
        self.w_segments = w_segments

        self.set_n_initializer(dgl.init.zero_initializer)
        self.set_e_initializer(dgl.init.zero_initializer)

        if alt is "1":
            self.initializeWithAlternative1()
        elif alt is "2":
            pass
        else:
            print("Introduce a valid initializer alternative")

    def initializeWithAlternative1(self):

        rels_class = {'r_p', 'r_o', 'r_w', 'p_p', 'p_o', 'g_r'}
        # ^
        # |_p = person
        # |_r = room
        # |_o = object
        # |_w = wall
        # |_g = goal

        for e in list(rels_class):
            rels_class.add(e[::-1])
        rels_class = rels_class.union({'SP', 'SR', 'SO', 'SW', 'SG'})
        rels_class = sorted(list(rels_class))
        self.num_rels = len(rels_class)
        self.rels = rels_class

        # Feature dimensions
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'goal']
        time_one_hot = ['is_t_0', 'is_t_m1', 'is_t_m2']
        time_sequence_features = ['is_first_frame', 'time_left']
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        object_metric_features = ['obj_x_pos', 'obj_y_pos',  'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        room_metric_features = ['room_humans', 'room_humans2']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        self.all_features = node_types_one_hot + time_one_hot + time_sequence_features + human_metric_features + \
                           object_metric_features + room_metric_features + wall_metric_features + goal_metric_features
        self.feature_dimensions = len(self.all_features)

        # Relations are integers
        RelTensor = torch.LongTensor
        # Normalization factors are floats
        NormTensor = torch.Tensor

        # Compute data for walls
        Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
        walls = []
        i_w = 0
        for wall_segment in self.data['walls']:
            p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
            p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
            dist = np.linalg.norm(p1 - p2)
            if i_w >= len(self.w_segments):
                iters = int(dist / 400) + 1
                self.w_segments.append(iters)
            if self.w_segments[i_w] > 1:    #WE NEED TO CHECK THIS PART
                v = (p2 - p1) / self.w_segments[i_w]
                for i in range(self.w_segments[i_w]):
                    pa = p1 + v * i
                    pb = p1 + v * (i + 1)
                    inc2 = pb - pa
                    midsp = (pa + pb) / 2
                    walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
            else:
                inc = p2 - p1
                midp = (p2 + p1) / 2
                walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
            i_w+=1

        # Compute the number of nodes
        #      room +  room walls      + humans               + objects                 + Goal node
        n_nodes = 1 + len(walls) + len(self.data['people']) + len(self.data['objects']) + 1

        # Initialize features, labels and nodes
        self.features = np.zeros([n_nodes, self.feature_dimensions])
        self.labels = np.array(self.data['command'])
        self.labels[0] = self.labels[0]/MAX_ADV
        self.labels[2] = self.labels[2]/MAX_ROT
        self.add_nodes(n_nodes)

        # Initialise id counter
        max_used_id = 0

        # Room (Global node)
        room_id = 0
        self.typeMap[room_id] = 'r'  # 'r' for 'room'
        self.features[room_id, self.all_features.index('room')] = 1.
        self.features[room_id, self.all_features.index('room_humans')] = len(self.data['people']) / MAX_HUMANS
        self.features[room_id, self.all_features.index('room_humans2')] = (len(self.data['people']) ** 2) / (MAX_HUMANS ** 2)
        self.add_edge(room_id, room_id, {'rel_type': RelTensor([[self.rels.index('SR')]]), 'norm': NormTensor([[1]])})
        max_used_id += 1

        # humans
        for h in self.data['people']:
            self.add_edge(h['id'], room_id, {'rel_type': RelTensor([[self.rels.index('p_r')]]),
                                             #'norm': NormTensor([[1]])})
                                             'norm': NormTensor([[1. / len(self.data['people'])]])})
            self.add_edge(room_id, h['id'], {'rel_type': RelTensor([[self.rels.index('r_p')]]),
                                             'norm': NormTensor([[1.]])})
            self.add_edge(h['id'], h['id'], {'rel_type': RelTensor([[self.rels.index('SP')]]), 'norm': NormTensor([[1]])})
            self.typeMap[h['id']] = 'p'  # 'p' for 'person'
            max_used_id += 1
            xpos = h['x'] / 10.
            ypos = h['y'] / 10.
            dist = math.sqrt(xpos**2 + ypos**2)
            va = h['va'] / 10.
            vx = h['vx'] / 10.
            vy = h['vy'] / 10.

            orientation = h['a']
            while orientation > math.pi:
                orientation -= 2. * math.pi
            while orientation < -math.pi:
                orientation += 2. * math.pi
            if orientation > math.pi:
                orientation -= math.pi
            elif orientation < -math.pi:
                orientation += math.pi

            self.features[h['id'], self.all_features.index('human')] = 1.
            self.features[h['id'], self.all_features.index('hum_orientation_sin')] = math.sin(orientation)
            self.features[h['id'], self.all_features.index('hum_orientation_cos')] = math.cos(orientation)
            self.features[h['id'], self.all_features.index('hum_x_pos')] = xpos
            self.features[h['id'], self.all_features.index('hum_y_pos')] = ypos
            self.features[h['id'], self.all_features.index('human_a_vel')] = va
            self.features[h['id'], self.all_features.index('human_x_vel')] = vx
            self.features[h['id'], self.all_features.index('human_y_vel')] = vy
            self.features[h['id'], self.all_features.index('hum_dist')] = dist
            self.features[h['id'], self.all_features.index('hum_inv_dist')] = 1.-dist #/(1.+dist*10.)

        # objects
        for o in self.data['objects']:
            self.add_edge(o['id'], room_id, {'rel_type': RelTensor([[self.rels.index('o_r')]]),
                                             #'norm': NormTensor([[1.]])})
                                             'norm': NormTensor([[1. / len(self.data['objects'])]])})
            self.add_edge(room_id, o['id'], {'rel_type': RelTensor([[self.rels.index('r_o')]]),
                                             'norm': NormTensor([[1.]])})
            self.add_edge(o['id'], o['id'], {'rel_type': RelTensor([[self.rels.index('SO')]]), 'norm': NormTensor([[1]])})
            self.typeMap[o['id']] = 'o'  # 'o' for 'object'
            max_used_id += 1
            xpos = o['x'] / 10.
            ypos = o['y'] / 10.
            dist = math.sqrt(xpos**2 + ypos**2)
            va = o['va'] / 10.
            vx = o['vx'] / 10.
            vy = o['vy'] / 10.

            orientation = o['a']
            while orientation > math.pi:
                orientation -= 2. * math.pi
            while orientation < -math.pi:
                orientation += 2. * math.pi
            self.features[o['id'], self.all_features.index('object')] = 1
            self.features[o['id'], self.all_features.index('obj_orientation_sin')] = math.sin(orientation)
            self.features[o['id'], self.all_features.index('obj_orientation_cos')] = math.cos(orientation)
            self.features[o['id'], self.all_features.index('obj_x_pos')] = xpos
            self.features[o['id'], self.all_features.index('obj_y_pos')] = ypos
            self.features[o['id'], self.all_features.index('obj_a_vel')] = va
            self.features[o['id'], self.all_features.index('obj_x_vel')] = vx
            self.features[o['id'], self.all_features.index('obj_y_vel')] = vy
            self.features[o['id'], self.all_features.index('obj_x_size')] = o['size_x']
            self.features[o['id'], self.all_features.index('obj_y_size')] = o['size_y']
            self.features[o['id'], self.all_features.index('obj_dist')] = dist
            self.features[o['id'], self.all_features.index('obj_inv_dist')] = 1.-dist#/(1.+dist*10.)

        # Goal
        goal_id = max_used_id
        self.typeMap[goal_id] = 'g'  # 'g' for 'goal'
        self.add_edge(goal_id, room_id, {'rel_type': RelTensor([[self.rels.index('g_r')]]),
                                             'norm': NormTensor([[1.]])})
        self.add_edge(room_id, goal_id, {'rel_type': RelTensor([[self.rels.index('r_g')]]),
                                             'norm': NormTensor([[1.]])})
        self.add_edge(goal_id, goal_id, {'rel_type': RelTensor([[self.rels.index('SG')]]), 'norm': NormTensor([[1]])})
        xpos = self.data['goal'][0]['x'] / 10.
        ypos = self.data['goal'][0]['y'] / 10.
        dist = math.sqrt(xpos**2 + ypos**2)
        self.features[goal_id, self.all_features.index('goal')] = 1
        self.features[goal_id, self.all_features.index('goal_x_pos')] = xpos
        self.features[goal_id, self.all_features.index('goal_y_pos')] = ypos
        self.features[goal_id, self.all_features.index('goal_dist')] = dist
        self.features[goal_id, self.all_features.index('goal_inv_dist')] = 1.-dist#/(1.+dist*10.)

        max_used_id += 1

        # walls
        wids = dict()
        for wall in walls:
            wall_id = max_used_id
            wids[wall] = wall_id
            max_used_id += 1
            self.typeMap[wall_id] = 'w'  # 'w' for 'walls'
            self.add_edge(wall_id, room_id, {'rel_type': RelTensor([[self.rels.index('w_r')]]),
                                             #'norm': NormTensor([[1.]])})
                                             'norm': NormTensor([[1. / len(walls)]])})
            self.add_edge(room_id, wall_id, {'rel_type': RelTensor([[self.rels.index('r_w')]]),
                                             'norm': NormTensor([[1.]])})
            self.add_edge(wall_id, wall_id, {'rel_type': RelTensor([[self.rels.index('SW')]]), 'norm': NormTensor([[1]])})

            dist = math.sqrt((wall.xpos/1000.)**2 + (wall.ypos/1000.)**2)
            self.features[wall_id, self.all_features.index('wall')] = 1.
            self.features[wall_id, self.all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
            self.features[wall_id, self.all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
            self.features[wall_id, self.all_features.index('wall_x_pos')] = wall.xpos / 1000.
            self.features[wall_id, self.all_features.index('wall_y_pos')] = wall.ypos / 1000.
            self.features[wall_id, self.all_features.index('wall_dist')] = dist
            self.features[wall_id, self.all_features.index('wall_inv_dist')] = 1.-dist #1./(1.+dist*10.)

        # Interaction edges
        for link in self.data['interaction']:
            typeLdir = self.typeMap[link['dst']] + '_' + self.typeMap[link['src']]
            typeLinv = self.typeMap[link['src']] + '_' + self.typeMap[link['dst']]
            self.add_edge(link['dst'], link['src'], {'rel_type': RelTensor([[self.rels.index(typeLdir)]]),
                                                     'norm': NormTensor([[1.]])})
            self.add_edge(link['src'], link['dst'], {'rel_type': RelTensor([[self.rels.index(typeLinv)]]),
                                                     'norm': NormTensor([[1.]])})

        # ########################## #


class GenerateDataset(object):
    def __init__(self, path, mode, alt, init_line=-1, end_line=-1, verbose=True, debug=False, i_frame=0):
        super(GenerateDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.data = []
        self.num_rels = -1
        self.debug = debug
        try:
            if self.mode != 'run':
                self.load_from_file()
            else:
                self._load(alt, mode, i_frame)
        except FileNotFoundError:
            self._load(alt, mode, i_frame)
        self._preprocess()
        if verbose:
            print('{} scenarios loaded.'.format(len(self.data)))

    def _load(self, alt, mode, i_frame):
        if type(self.path) is str and self.path.endswith('.json'):
            linen = -1
            if self.debug:
                self.load_from_json(self.path, alt, mode, linen)
            else:
                with open(self.path) as json_file:
                    data = json.load(json_file)
                for frame in data:
                    if linen % 1000 == 0:
                        print(linen)
                    if linen + 1 >= limit:
                        print('Stop including more samples to speed up dataset loading')
                        break
                    linen += 1
                    if self.init_line >= 0 and linen < self.init_line:
                        continue
                    if linen > self.end_line >= 0:
                        continue
                    try:
                        self.data.append(RoomGraph(frame, alt, mode, debug=self.debug))
                    except Exception:
                        print(frame)
                        raise
        elif type(self.path) is str and self.path.endswith('.txt'):
            linen = -1
            with open(self.path) as set_file:
                ds_files = set_file.read().splitlines()
            print("number of files for ", self.path, len(ds_files))

            for ds in ds_files:
                linen = self.load_from_json(ds, alt, mode, linen)
                if linen + 1 >= limit:
                    break

        elif type(self.path) == list and type(self.path[0]) == str:
            self.data.append(RoomGraph(json.loads(self.path[0]), alt, mode))

        elif type(self.path) == list and len(self.path)>=1:
            self.load_one_graph(self.path, alt, mode, i_frame)
            #self.data.append(RoomGraph(self.path, alt, mode))
        self.num_rels = self.data[0].num_rels

        random.shuffle(self.data)

        if self.mode is not 'run':
            self.save_to_file()

    def get_dataset_name(self):
        return 'dataset_' + self.mode + '_s_' + str(limit) + '.pickle'

    def load_from_json(self, ds_file, alt, mode, linen):
        #print(ds_file)
        with open(ds_file) as json_file:
            data = json.load(json_file)

        new_features = ['is_t_0', 'is_t_m1', 'is_t_m2']
        w_segments = []
        frames_in_interval = []
        graphs_in_interval = []
        frame_new = data[0]
        frames_in_interval.append(data[0])
        graph_new = RoomGraph(frame_new, alt, mode, debug=self.debug, w_segments=w_segments)
        graph_new.features[:, graph_new.all_features.index('is_first_frame')] = 1
        graph_new.features[:, graph_new.all_features.index('time_left')] = 1.
        w_segments = graph_new.w_segments
        graphs_in_interval.append(graph_new)

        i_frame = 0
        for frame in data[1:]:
            frame_new = frame
            i_frame += 1
            if frame_new['timestamp']-frames_in_interval[0]['timestamp'] < FRAMES_INTERVAL:  # Truncated to N seconds
                continue
            if linen % 1000 == 0:
                print(linen)
            if linen + 1 >= limit:
                print('Stop including more samples to speed up dataset loading')
                break
            linen += 1
            if self.init_line >= 0 and linen < self.init_line:
                continue
            if linen > self.end_line >= 0:
                continue
            try:
                graph_new = RoomGraph(frame_new, alt, mode, debug=self.debug, w_segments=w_segments)
                graph_new.features[:, graph_new.all_features.index('time_left')] = 1./(i_frame+1.)
                frames_in_interval.insert(0, frame_new)
                graphs_in_interval.insert(0, graph_new)
                if len(graphs_in_interval) > N_INTERVALS:
                    graphs_in_interval.pop(N_INTERVALS)
                    frames_in_interval.pop(N_INTERVALS)
#                if len(graphs_in_interval)<N_INTERVALS:
#                    continue

                #print("nnodes new", graph_new.batch_num_nodes[0], "nnodes old", graph_old.batch_num_nodes[0])
                f_list = []
                for g in graphs_in_interval:
                    f_list.append(g.features)
                    for f in new_features:
                        if graphs_in_interval.index(g) == new_features.index(f):
                            g.features[:, g.all_features.index(f)] = 1
                        else:
                            g.features[:, g.all_features.index(f)] = 0

                # Create merged graph:
                final_graph = dgl.batch(graphs_in_interval)
                final_graph.flatten()

                # Add merged features amd number of relations:
                
                N = len(graphs_in_interval)
                final_features = np.concatenate(f_list, axis=0)
                setattr(final_graph, 'features', final_features)
                setattr(final_graph, "num_rels", graphs_in_interval[0].num_rels+(N_INTERVALS-1)*2)
                setattr(final_graph, "labels", graphs_in_interval[0].labels)
                if self.debug:
                    t_list = []
                    for g in graphs_in_interval:
                        t_list.append(g.typeMap)
                    setattr(final_graph, "typeMap", t_list)
                    setattr(final_graph, "all_features", graphs_in_interval[0].all_features)

                # Add temporal edges (one direction from new to old):
                offset = graphs_in_interval[0].batch_num_nodes[0]
                for noden in range(graphs_in_interval[0].batch_num_nodes[0]):
                    for i in range(1,N):
                        final_graph.add_edge(noden + i*offset, noden,
                                             {'rel_type': torch.LongTensor([[graph_new.num_rels+(i-1)*2]]),
                                              'norm': torch.Tensor([[1.]])})
                        final_graph.add_edge(noden, noden + i*offset,
                                             {'rel_type': torch.LongTensor([[graph_new.num_rels+(i-1)*2+1]]),
                                              'norm': torch.Tensor([[1.]])})


                

                self.data.append(final_graph)
            except Exception:
                print(frame)
                raise

        return linen

    def load_one_graph(self, path, alt, mode, i_frame):
        new_features = ['is_t_0', 'is_t_m1', 'is_t_m2']
        graph = RoomGraph(path[0], alt, mode, debug=self.debug)

        w_segments = graph.w_segments
        graphs_in_interval=[graph]
        for i in range(1, len(path)):
            graphs_in_interval.append(RoomGraph(path[i], alt, mode, debug=self.debug, w_segments = w_segments))

        f_list = []
        i_graph = 0
        for g in graphs_in_interval:
            if (i_frame-i_graph) == 0:
                g.features[:, g.all_features.index('is_first_frame')] = 1
            g.features[:, g.all_features.index('time_left')] = 1. / (i_frame-i_graph+1)
            i_graph += 1
            f_list.append(g.features)
            for f in new_features:
                if graphs_in_interval.index(g) == new_features.index(f):
                    g.features[:, g.all_features.index(f)] = 1
                else:
                    g.features[:, g.all_features.index(f)] = 0


        #print("nnodes new", graph_new.batch_num_nodes[0], "nnodes old", graph_old.batch_num_nodes[0])

        # Create merged graph:
        final_graph = dgl.batch(graphs_in_interval)
        final_graph.flatten()

        N = len(graphs_in_interval)

        # Add merged features amd number of relations:
        final_features = np.concatenate(f_list, axis=0)
        setattr(final_graph, 'features', final_features)
        setattr(final_graph, "num_rels", graph.num_rels+(N_INTERVALS-1)*2)
        setattr(final_graph, "labels", graph.labels)


        # Add temporal edges (one direction from new to old):
        offset = graph.batch_num_nodes[0]
        for noden in range(graph.batch_num_nodes[0]):
            for i in range(1,N):
                final_graph.add_edge(noden + i*offset, noden, 
                                     {'rel_type': torch.LongTensor([[graph.num_rels+(i-1)*2]]),
                                      'norm': torch.Tensor([[1.]])})
                final_graph.add_edge(noden, noden + i*offset, 
                                     {'rel_type': torch.LongTensor([[graph.num_rels+(i-1)*2+1]]),
                                      'norm': torch.Tensor([[1.]])})


        self.data.append(final_graph)


    def load_from_file(self):
        filename = self.get_dataset_name()

        with open(path_saves + filename, 'rb') as f:
            self.data = pickle.load(f)

        self.num_rels = self.data[0].num_rels
        for d in self.data:
            del d.num_rels

    def save_to_file(self):
        filename = self.get_dataset_name()
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        with open(path_saves + filename, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def _preprocess(self):
        pass  # We don't really need to do anything, all pre-processing is done in the _load method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.data[item].features, self.data[item].labels
