import os
import sys
import json
import copy
from collections import namedtuple
import math

import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import numpy as np
import time

N_INTERVALS = 3
FRAMES_INTERVAL = 1.
WITH_EDGE_FEATURES = True

grid_width = 11  # 30 #18
output_width = 73  # 121 #73
area_width = 1000.  # horizontal/vertical distance between two contiguous nodes of the grid. Previously it was taken as the spatial area of the grid

threshold_human_wall = 1.5
limit = 50000  # Limit of graphs to load
path_saves = 'saves/'  # This variable is necessary due to a bug in dgl.DGLDataset source code
graphData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'edge_feats', 'edge_types',
                                     'edge_norms', 'position_by_id', 'typeMap', 'labels', 'w_segments'])


#  human to wall distance
def dist_h_w(h, wall):
    if 'xPos' in h.keys():
        hxpos = float(h['xPos']) / 100.
        hypos = float(h['yPos']) / 100.
    else:
        hxpos = float(h['x'])
        hypos = float(h['y'])

    wxpos = float(wall.xpos) / 100.
    wypos = float(wall.ypos) / 100.
    return math.sqrt((hxpos - wxpos) * (hxpos - wxpos) + (hypos - wypos) * (hypos - wypos))

# return de number of grid nodes if a grid is used in the specified alternative
def grid_nodes_number(alt):
    if alt=='2' or alt=='7' or alt=='8':
        return grid_width*grid_width
    else:
        return 0

def central_grid_nodes(alt, r):
    if alt == '7' or alt == '8':
        grid_node_ids = np.zeros((grid_width, grid_width), dtype=int)
        for y in range(grid_width):
            for x in range(grid_width):
                grid_node_ids[x][y] = y * grid_width + x
        central_nodes = closest_grid_nodes(grid_node_ids, area_width, grid_width, r, 0, 0)
        return central_nodes
    else:
        return []

# Calculate the closet node in the grid to a given node by its coordinates
def closest_grid_node(grid_ids, w_a, w_i, x, y):
    c_x = int(round(x / w_a) + (w_i // 2))
    if c_x < 0: c_x = 0
    if c_x >= grid_width: c_x = grid_width - 1
    c_y = int(round(y / w_a) + (w_i // 2))
    if c_y < 0: c_y = 0
    if c_y >= grid_width: c_y = grid_width - 1
    return grid_ids[c_x][c_y]

    # if 0 <= c_x < grid_width and 0 <= c_y < grid_width:
    #     return grid_ids[c_x][c_y]
    # return None


def closest_grid_nodes(grid_ids, w_a, w_i, r, x, y):
    c_x = int(round(x / w_a) + (w_i // 2))
    c_y = int(round(y / w_a) + (w_i // 2))
    cols, rows = (int(math.ceil(r / w_a)), int(math.ceil(r / w_a)))
    rangeC = list(range(-cols, cols + 1))
    rangeR = list(range(-rows, rows + 1))
    p_arr = [[c, r] for c in rangeC for r in rangeR]
    grid_nodes = []
    # r_g = r / w_a
    for p in p_arr:
        g_x, g_y = c_x + p[0], c_y + p[1]
        gw_x, gw_y = (g_x-w_i//2)*w_a, (g_y-w_i//2)*w_a
        if math.sqrt((gw_x - x) * (gw_x - x) + (gw_y - y) * (gw_y - y)) <= r:
        # if math.sqrt(p[0] * p[0] + p[1] * p[1]) <= r_g:
            if 0 <= g_x < grid_width and 0 <= g_y < grid_width:
                grid_nodes.append(grid_ids[g_x][g_y])

    return grid_nodes


def get_relations(alt):
    rels = None
    if alt == '1':
        rels = {'p_r', 'o_r', 'l_r', 'l_p', 'l_o', 'p_p', 'p_o', 'w_l', 'w_p'}
        # p = person
        # r = robot
        # l = room (lounge)
        # o = object
        # w = wall
        # n = node (generic)
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '2':
        room_set = {'l_p', 'l_o', 'l_w', 'l_g', 'p_p', 'p_o', 'p_g', 'o_g', 'w_g'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # ^
        # |_p = person             g_ri = grid right
        # |_w = wall               g_le = grid left
        # |_l = lounge             g_u = grid up
        # |_o = object             g_d = grid down
        # |_g = grid node
        self_edges_set = {'P', 'O', 'W', 'L'}

        for e in list(room_set):
            room_set.add(e[::-1])
        relations_class = room_set | grid_set | self_edges_set
        rels = sorted(list(relations_class))
    elif alt == '3':
        rels = {'p_r', 'o_r', 'p_p', 'p_o', 'w_r', 't_r', 'w_p'} # add 'w_w' for links between wall nodes
        # p = person
        # r = room
        # o = object
        # w = wall
        # t = goal
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '4':
        rels = {'o_r', 'g_r'} 
        # r = room
        # o = object
        # g = goal
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '5' or alt == '6':
        rels = {'p_r', 't_r'} 
        # r = room
        # p = person
        # t = goal (target)
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '7':
        rels = {'p_r', 't_r', 'w_r', 'p_g', 't_g', 'r_g', 'w_g'} 
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # r = room
        # p = person
        # t = goal (target)
        # w = wall
        # g = grid
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = rels | grid_set
        rels = sorted(list(rels))
    elif alt == '8' or alt == '9':
        rels = {'p_r', 'o_r', 'p_p', 'p_o', 't_r', 'w_r', 'p_g', 'o_g', 't_g', 'r_g', 'w_g'} 
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # r = room
        # p = person
        # o = object
        # t = goal (target)
        # w = wall
        # g = grid
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = rels | grid_set
        rels = sorted(list(rels))
        

    num_rels = len(rels)

    return rels, num_rels


def get_features(alt):
    all_features = None
    if alt == '1':
        node_types_one_hot = ['robot', 'human', 'object', 'room', 'wall']
        human_metric_features = ['hum_distance', 'hum_distance2', 'hum_angle_sin', 'hum_angle_cos',
                                 'hum_orientation_sin', 'hum_orientation_cos', 'hum_robot_sin',
                                 'hum_robot_cos']
        object_metric_features = ['obj_distance', 'obj_distance2', 'obj_angle_sin', 'obj_angle_cos',
                                  'obj_orientation_sin', 'obj_orientation_cos']
        room_metric_features = ['room_min_human', 'room_min_human2', 'room_humans', 'room_humans2']
        wall_metric_features = ['wall_distance', 'wall_distance2', 'wall_angle_sin', 'wall_angle_cos',
                                'wall_orientation_sin', 'wall_orientation_cos']
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + \
                       wall_metric_features
    elif alt == '2':
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'grid']
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'hum_orientation_sin', 'hum_orientation_cos']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_orientation_sin', 'obj_orientation_cos']
        room_metric_features = ['room_humans', 'room_humans2']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']  # , 'flag_inside_room']  # , 'count']
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + \
                       wall_metric_features + grid_metric_features
    elif alt == '3':
        time_one_hot = ['is_t_0', 'is_t_m1', 'is_t_m2']
        # time_sequence_features = ['is_first_frame', 'time_left']
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        room_metric_features = ['room_humans', 'room_humans2']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'goal']
        all_features = node_types_one_hot + time_one_hot + human_metric_features + robot_features + \
                       object_metric_features + room_metric_features + wall_metric_features + goal_metric_features
    elif alt == '4':
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        node_types_one_hot = ['object', 'room', 'goal']
        all_features = node_types_one_hot + robot_features + \
                       object_metric_features + goal_metric_features
        if N_INTERVALS > 1:
            time_one_hot = ['is_t_0']
            for i in range(1, N_INTERVALS):
                time_one_hot.append('is_t_m' + str(i))
            all_features += time_one_hot    
    elif alt == '5':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        node_types_one_hot = ['human', 'room', 'goal']
        all_features = node_types_one_hot + robot_features + \
                       human_metric_features + goal_metric_features
        if N_INTERVALS > 1:
            time_one_hot = ['is_t_0']
            for i in range(1, N_INTERVALS):
                time_one_hot.append('is_t_m' + str(i))
            all_features += time_one_hot    
    elif alt == '6':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant = robot_features +  human_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features
        node_types_one_hot = ['human', 'room', 'goal']
        all_features += node_types_one_hot

    elif alt == '7':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant =  robot_features + human_metric_features + \
                                  wall_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features        
        node_types_one_hot = ['human', 'room', 'goal', 'wall', 'grid']        
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        all_features += node_types_one_hot + grid_metric_features

    elif alt == '8' or alt == '9':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant =  robot_features + human_metric_features + object_metric_features + \
                                  wall_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features        
        node_types_one_hot = ['human', 'object', 'room', 'goal', 'wall', 'grid']        
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        all_features += node_types_one_hot + grid_metric_features


    feature_dimensions = len(all_features)

    return all_features, feature_dimensions


#################################################################
# Different initialize alternatives:
#################################################################

# Generate data for a grid of nodes

def generate_grid_graph_data(alt = '2'):
    # Define variables for edge types and relations
    grid_rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Grid properties
    connectivity = 8  # Connections of each node
    node_ids = np.zeros((grid_width, grid_width), dtype=int)  # Array to store the IDs of each node
    typeMap = dict()
    coordinates_gridGraph = dict()  # Dict to store the spatial coordinates of each node
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Feature dimensions
    all_features, n_features = get_features(alt)

    # Compute the number of nodes and initialize feature vectors
    n_nodes = grid_width ** 2
    features_gridGraph = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    max_used_id = -1
    for y in range(grid_width):
        for x in range(grid_width):
            max_used_id += 1
            node_id = max_used_id
            node_ids[x][y] = node_id

            # Self edges
            src_nodes.append(node_id)
            dst_nodes.append(node_id)
            edge_types.append(grid_rels.index('g_c'))
            edge_norms.append([1.])
            edge_features = th.zeros(num_rels + 4)
            edge_features[grid_rels.index('g_c')] = 1
            edge_features[-1] = 0
            edge_feats_list.append(edge_features)


            if x < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + 1)
                edge_types.append(grid_rels.index('g_ri'))
                edge_norms.append([1.])
                edge_features = th.zeros(num_rels + 4)
                edge_features[grid_rels.index('g_ri')] = 1
                edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and y > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width + 1)
                    edge_types.append(grid_rels.index('g_uri'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[grid_rels.index('g_uri')] = 1
                    edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)

            if x > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - 1)
                edge_types.append(grid_rels.index('g_le'))
                edge_norms.append([1.])
                edge_features = th.zeros(num_rels + 4)
                edge_features[grid_rels.index('g_le')] = 1
                edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and y < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width - 1)
                    edge_types.append(grid_rels.index('g_dle'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[grid_rels.index('g_dle')] = 1
                    edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)

            if y < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + grid_width)
                edge_types.append(grid_rels.index('g_d'))
                edge_norms.append([1.])
                edge_features = th.zeros(num_rels + 4)
                edge_features[grid_rels.index('g_d')] = 1
                edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and x < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width + 1)
                    edge_types.append(grid_rels.index('g_dri'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[grid_rels.index('g_dri')] = 1
                    edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)

            if y > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - grid_width)
                edge_types.append(grid_rels.index('g_u'))
                edge_norms.append([1.])
                edge_features = th.zeros(num_rels + 4)
                edge_features[grid_rels.index('g_u')] = 1
                edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and x > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width - 1)
                    edge_types.append(grid_rels.index('g_ule'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[grid_rels.index('g_ule')] = 1
                    edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)


            typeMap[node_id] = 'g'  # 'g' for 'grid'
            x_pos = (x - grid_width // 2) * area_width
            y_pos = (y - grid_width // 2) * area_width
            # x_pos = (-area_width / 2. + (x + 0.5) * (area_width / grid_width))
            # y_pos = (-area_width / 2. + (y + 0.5) * (area_width / grid_width))
            features_gridGraph[node_id, all_features.index('grid')] = 1
            features_gridGraph[node_id, all_features.index('grid_x_pos')] = x_pos / 10000
            features_gridGraph[node_id, all_features.index('grid_y_pos')] = y_pos / 10000

            coordinates_gridGraph[node_id] = [x_pos / 10000, y_pos / 10000]

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)
    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features_gridGraph, edge_feats, edge_types, edge_norms, coordinates_gridGraph, typeMap, \
           node_ids, []


# So far there two different alternatives, the second one includes the grid
def initializeAlt1(data):
    # Initialize variables
    rels, num_rels = get_relations('1')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)
    closest_human_distance = -1  # Compute closest human distance

    # Compute data for walls
    Wall = namedtuple('Wall', ['dist', 'orientation', 'angle', 'xpos', 'ypos'])
    walls = []
    for wall_index in range(len(data['room']) - 1):
        p1 = np.array(data['room'][wall_index + 0])
        p2 = np.array(data['room'][wall_index + 1])
        dist = np.linalg.norm(p1 - p2)
        iters = int(dist / 400) + 1
        if iters > 1:
            v = (p2 - p1) / iters
            for i in range(iters):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(
                    Wall(np.linalg.norm(midsp) / 100., math.atan2(inc2[0], inc2[1]), math.atan2(midsp[0], midsp[1]),
                         midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(
                Wall(np.linalg.norm(inc / 2) / 100., math.atan2(inc[0], inc[1]), math.atan2(midp[0], midp[1]),
                     midp[0], midp[1]))

    # Compute the number of nodes
    # one for the robot + room walls      + humans               + objects          + room(global node)
    n_nodes = 1 + len(walls) + len(data['humans']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('1')
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    labels = th.zeros([1, 1])  # A 1x1 tensor
    labels[0][0] = th.tensor(float(data['score']) / 100.)

    # robot (id 0)
    robot_id = 0
    typeMap[robot_id] = 'r'  # 'r' for 'robot'
    features[robot_id, all_features.index('robot')] = 1.

    # humans
    for h in data['humans']:
        src_nodes.append(h['id'])
        dst_nodes.append(robot_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['humans'])])

        src_nodes.append(robot_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id = max(h['id'], max_used_id)
        xpos = float(h['xPos']) / 100.
        ypos = float(h['yPos']) / 100.

        position_by_id[h['id']] = [xpos, ypos]
        distance = math.sqrt(xpos * xpos + ypos * ypos)
        angle = math.atan2(xpos, ypos)
        orientation = float(h['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi
        # Compute point of view from humans
        if angle > 0:
            angle_hum = (angle - math.pi) - orientation
        else:
            angle_hum = (math.pi + angle) - orientation

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_distance')] = distance
        features[h['id'], all_features.index('hum_distance2')] = distance * distance
        features[h['id'], all_features.index('hum_angle_sin')] = math.sin(angle)
        features[h['id'], all_features.index('hum_angle_cos')] = math.cos(angle)
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_robot_sin')] = math.sin(angle_hum)
        features[h['id'], all_features.index('hum_robot_cos')] = math.cos(angle_hum)
        if closest_human_distance < 0 or closest_human_distance > distance:
            closest_human_distance = distance

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(robot_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1. / len(data['objects'])])

        src_nodes.append(robot_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id = max(o['id'], max_used_id)
        xpos = float(o['xPos']) / 100.
        ypos = float(o['yPos']) / 100.

        position_by_id[o['id']] = [xpos, ypos]
        distance = math.sqrt(xpos * xpos + ypos * ypos)
        angle = math.atan2(xpos, ypos)
        orientation = float(o['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_distance')] = distance
        features[o['id'], all_features.index('obj_distance2')] = distance * distance
        features[o['id'], all_features.index('obj_angle_sin')] = math.sin(angle)
        features[o['id'], all_features.index('obj_angle_cos')] = math.cos(angle)
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)

    # Room (Global node)
    max_used_id += 1
    room_id = max_used_id
    # print('Room will be {}'.format(room_id))
    typeMap[room_id] = 'l'  # 'l' for 'room' (lounge)
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_min_human')] = closest_human_distance
    features[room_id, all_features.index('room_min_human2')] = closest_human_distance * closest_human_distance
    features[room_id, all_features.index('room_humans')] = len(data['humans'])
    features[room_id, all_features.index('room_humans2')] = len(data['humans']) * len(data['humans'])

    # walls
    wids = dict()
    for wall in walls:
        max_used_id += 1
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_l'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('l_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]
        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_distance')] = wall.dist
        features[wall_id, all_features.index('wall_distance2')] = wall.dist * wall.dist
        features[wall_id, all_features.index('wall_angle_sin')] = math.sin(wall.angle)
        features[wall_id, all_features.index('wall_angle_cos')] = math.cos(wall.angle)
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)

    for h in data['humans']:
        number = 0
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(wids[wall])
                dst_nodes.append(h['id'])
                edge_types.append(rels.index('w_p'))
                edge_norms.append([1. / number])

    for wall in walls:
        number = 0
        for h in data['humans']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for h in data['humans']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(h['id'])
                dst_nodes.append(wids[wall])
                edge_types.append(rels.index('p_w'))
                edge_norms.append([1. / number])

    # interaction links
    for link in data['links']:
        typeLdir = typeMap[link[0]] + '_' + typeMap[link[1]]
        typeLinv = typeMap[link[1]] + '_' + typeMap[link[0]]

        src_nodes.append(link[0])
        dst_nodes.append(link[1])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link[1])
        dst_nodes.append(link[0])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

    # Edges for the room node (Global Node)
    for node_id in range(n_nodes):
        typeLdir = typeMap[room_id] + '_' + typeMap[node_id]
        typeLinv = typeMap[node_id] + '_' + typeMap[room_id]
        if node_id == room_id:
            continue

        src_nodes.append(room_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(node_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1. / max_used_id])

    # self edges
    for node_id in range(n_nodes - 1):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

    # Convert outputs to tensors
    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, None, edge_types, edge_norms, position_by_id, typeMap, labels, []


def initializeAlt2(data):
    # Define variables for edge types and relations
    rels, _ = get_relations('2')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    for wall_index in range(len(data['room']) - 1):
        p1 = np.array(data['room'][wall_index + 0])
        p2 = np.array(data['room'][wall_index + 1])
        dist = np.linalg.norm(p1 - p2)
        iters = int(dist / 400) + 1
        if iters > 1:
            v = (p2 - p1) / iters
            for i in range(iters):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))

    # Compute the number of nodes
    #      room +  room walls      + humans               + objects
    n_nodes = 1 + len(walls) + len(data['humans']) + len(data['objects'])

    # Feature dimensions
    all_features, n_features = get_features('2')
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    labels = th.zeros([1, 1])  # A 1x1 tensor
    labels[0][0] = th.tensor(float(data['score']) / 100.)

    # room (id 0) flobal node
    room_id = 0
    max_used_id = 0
    typeMap[room_id] = 'l'  # 'l' for 'room' (lounge)
    position_by_id[0] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['humans'])
    features[room_id, all_features.index('room_humans2')] = len(data['humans']) * len(data['humans'])

    # humans
    for h in data['humans']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_l'))
        edge_norms.append([1. / len(data['humans'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('l_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id = max(h['id'], max_used_id)
        xpos = float(h['xPos']) / 1000.
        ypos = float(h['yPos']) / 1000.

        position_by_id[h['id']] = [xpos, ypos]
        orientation = float(h['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi

        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = 2. * xpos
        features[h['id'], all_features.index('hum_y_pos')] = -2. * ypos

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_l'))
        edge_norms.append([1. / len(data['objects'])])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('l_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id = max(o['id'], max_used_id)
        xpos = float(o['xPos']) / 1000.
        ypos = float(o['yPos']) / 1000.

        position_by_id[o['id']] = [xpos, ypos]
        orientation = float(o['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = 2. * xpos
        features[o['id'], all_features.index('obj_y_pos')] = -2. * ypos

    # walls
    wids = dict()
    for wall in walls:
        max_used_id += 1
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_l'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('l_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 1000, wall.ypos / 1000]
        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = 2. * wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = -2. * wall.ypos / 1000.

    # interactions
    for link in data['links']:
        typeLdir = typeMap[link[0]] + '_' + typeMap[link[1]]
        typeLinv = typeMap[link[1]] + '_' + typeMap[link[0]]

        src_nodes.append(link[0])
        dst_nodes.append(link[1])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link[1])
        dst_nodes.append(link[0])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

    # self edges
    for node_id in range(n_nodes):
        r_type = typeMap[node_id].upper()

        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index(r_type))
        edge_norms.append([1.])

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, None, edge_types, edge_norms, position_by_id, typeMap, labels, []


MAX_ADV = 3.5
MAX_ROT = 4.
MAX_HUMANS = 15


def initializeAlt3(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('3')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    i_w = 0
    for wall_segment in data['walls']:
        p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
        p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
        dist = np.linalg.norm(p1 - p2)
        if i_w >= len(w_segments):
            iters = int(dist / 249) + 1
            w_segments.append(iters)
        if w_segments[i_w] > 1:  # WE NEED TO CHECK THIS PART
            v = (p2 - p1) / w_segments[i_w]
            for i in range(w_segments[i_w]):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
        i_w += 1

    # Compute the number of nodes
    # one for the robot + room walls   + humans    + objects              + room(global node)
    n_nodes = 1 + len(walls) + len(data['people']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('3')
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    # labels = th.zeros([1, 1])  # A 1x1 tensor
    # labels[0][0] = th.tensor(float(data['label_Q1']) / 100.)
    if 'label_Q1' in data.keys():
        labels = np.array([float(data['label_Q1']), float(data['label_Q2'])])
    else:
        labels = np.array([0, 0])
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['people']) / MAX_HUMANS
    features[room_id, all_features.index('room_humans2')] = (len(data['people']) ** 2) / (MAX_HUMANS ** 2)
    features[room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    features[room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1

    # humans
    for h in data['people']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['people'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id += 1
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        position_by_id[h['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
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

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = xpos
        features[h['id'], all_features.index('hum_y_pos')] = ypos
        features[h['id'], all_features.index('human_a_vel')] = va
        features[h['id'], all_features.index('human_x_vel')] = vx
        features[h['id'], all_features.index('human_y_vel')] = vy
        features[h['id'], all_features.index('hum_dist')] = dist
        features[h['id'], all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('p_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_p')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)



    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1.])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        position_by_id[o['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = xpos
        features[o['id'], all_features.index('obj_y_pos')] = ypos
        features[o['id'], all_features.index('obj_a_vel')] = va
        features[o['id'], all_features.index('obj_x_vel')] = vx
        features[o['id'], all_features.index('obj_y_vel')] = vy
        features[o['id'], all_features.index('obj_x_size')] = o['size_x']
        features[o['id'], all_features.index('obj_y_size')] = o['size_y']
        features[o['id'], all_features.index('obj_dist')] = dist
        features[o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('o_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_o')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 't'  # 't' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('t_r'))
    edge_norms.append([1.])
    # edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_t'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # Edge features
    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('t_r')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('r_t')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    # walls
    wids = dict()
    for w_i, wall in enumerate(walls, 0):
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'
        max_used_id += 1

		# # uncomment for links between wall nodes
        # if w_i == len(walls)-1:
        #     next_wall_id = max_used_id-len(walls)
        # else:
        #     next_wall_id = max_used_id
        # # ------------------------------------

        dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

        # Links to room node
        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_r'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('r_w'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('w_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_w')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        # # Links between wall nodes
        # wall_next = walls[(w_i + 1) % len(walls)]
        # dist_wnodes = math.sqrt((wall.xpos / 1000. - wall_next.xpos / 1000.) ** 2 +
        #                         (wall.ypos / 1000. - wall_next.ypos / 1000.) ** 2)

        # src_nodes.append(wall_id)
        # dst_nodes.append(next_wall_id)
        # edge_types.append(rels.index('w_w'))
        # edge_norms.append([1.])

        # src_nodes.append(next_wall_id)
        # dst_nodes.append(wall_id)
        # edge_types.append(rels.index('w_w'))
        # edge_norms.append([1.])

        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('w_w')] = 1
        # edge_features[-1] = dist_wnodes
        # edge_feats_list.append(edge_features)

        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('w_w')] = 1
        # edge_features[-1] = dist_wnodes
        # edge_feats_list.append(edge_features)
        # # ----------------------------------

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]

        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = wall.ypos / 1000.
        features[wall_id, all_features.index('wall_dist')] = dist
        features[wall_id, all_features.index('wall_inv_dist')] = 1. - dist  # 1./(1.+dist*10.)

    ## Links between walls and people
    # for h in data['people']:
    #     number = 0
    #     for wall in walls:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             number -= - 1
    #     for wall in walls:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             src_nodes.append(wids[wall])
    #             dst_nodes.append(h['id'])
    #             edge_types.append(rels.index('w_p'))
    #             edge_norms.append([1. / number])

    #             # Edge features
    #             edge_features = th.zeros(num_rels + 4)
    #             edge_features[rels.index('w_p')] = 1
    #             edge_features[-1] = dist
    #             edge_feats_list.append(edge_features)

    # for wall in walls:
    #     number = 0
    #     for h in data['people']:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             number -= - 1
    #     for h in data['people']:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             src_nodes.append(h['id'])
    #             dst_nodes.append(wids[wall])
    #             edge_types.append(rels.index('p_w'))
    #             edge_norms.append([1. / number])

    #             # Edge features
    #             edge_features = th.zeros(num_rels + 4)
    #             edge_features[rels.index('p_w')] = 1
    #             edge_features[-1] = dist
    #             edge_feats_list.append(edge_features)
    # # ----------------------------------


    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        dist = math.sqrt((position_by_id[link['src']][0] - position_by_id[link['dst']][0]) ** 2 +
                         (position_by_id[link['src']][1] - position_by_id[link['dst']][1]) ** 2)

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLdir)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLinv)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           labels, w_segments

def initializeAlt4(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('4')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute the number of nodes
    # one for the robot  + objects  + one for the goal
    n_nodes = 1 + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('4')
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    if 'label_Q1' in data.keys():
        labels = np.array([float(data['label_Q1']), float(data['label_Q2'])])
    else:
        labels = np.array([0, 0])
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    features[room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1


    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1.])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        position_by_id[o['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = xpos
        features[o['id'], all_features.index('obj_y_pos')] = ypos
        features[o['id'], all_features.index('obj_a_vel')] = va
        features[o['id'], all_features.index('obj_x_vel')] = vx
        features[o['id'], all_features.index('obj_y_vel')] = vy
        features[o['id'], all_features.index('obj_x_size')] = o['size_x']
        features[o['id'], all_features.index('obj_y_size')] = o['size_y']
        features[o['id'], all_features.index('obj_dist')] = dist
        features[o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('o_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_o')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 'g'  # 'g' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('g_r'))
    edge_norms.append([1.])
    # edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_g'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # Edge features
    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('g_r')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('r_g')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           labels, []

def initializeAlt5(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('5')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data['people']) + 1

    # Feature dimensions
    all_features, n_features = get_features('5')
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    if 'label_Q1' in data.keys():
        labels = np.array([float(data['label_Q1']), float(data['label_Q2'])])
    else:
        labels = np.array([0, 0])
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    features[room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1

    # humans
    for h in data['people']:
        h_id = max_used_id
        src_nodes.append(h_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['people'])])

        src_nodes.append(room_id)
        dst_nodes.append(h_id)
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h_id] = 'p'  # 'p' for 'person'
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        position_by_id[h_id] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = h['va'] / 10.
        vx = h['vx'] / 10.
        vy = h['vy'] / 10.

        max_used_id += 1
        orientation = h['a']

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h_id, all_features.index('human')] = 1.
        features[h_id, all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h_id, all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h_id, all_features.index('hum_x_pos')] = xpos
        features[h_id, all_features.index('hum_y_pos')] = ypos
        features[h_id, all_features.index('human_a_vel')] = va
        features[h_id, all_features.index('human_x_vel')] = vx
        features[h_id, all_features.index('human_y_vel')] = vy
        features[h_id, all_features.index('hum_dist')] = dist
        features[h_id, all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('p_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_p')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)


    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 'g'  # 'g' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('g_r'))
    edge_norms.append([1.])
    # edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_g'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # Edge features
    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('g_r')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('r_g')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           labels, []

Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
def generate_walls_information(data, w_segments):
    # Compute data for walls
    walls = []
    i_w = 0
    for wall_segment in data['walls']:
        p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
        p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
        dist = np.linalg.norm(p1 - p2)
        if i_w >= len(w_segments):
            iters = int(dist / 249) + 1
            w_segments.append(iters)
        if w_segments[i_w] > 1:  # WE NEED TO CHECK THIS PART
            v = (p2 - p1) / w_segments[i_w]
            for i in range(w_segments[i_w]):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
        i_w += 1
    return walls,  w_segments

# Initialize alternatives 6 and 7: 
# 6: people, goal and time features
# 7: people, walls, goal and time features. Can be combined with the grid
def initializeAlt6(data_sequence, alt = '6', w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data_sequence[0]['people']) + 1

    if alt == '7':
        walls, w_segments = generate_walls_information(data_sequence[0], w_segments)
        n_nodes += len(walls)


    # Feature dimensions
    all_features, n_features = get_features(alt)
    # print(all_features, n_features)
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    if 'label_Q1' in data_sequence[0].keys():
        labels = np.array([float(data_sequence[0]['label_Q1']), float(data_sequence[0]['label_Q2'])])
    else:
        labels = np.array([0, 0])
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    t_tag = ['']
    for i in range(1, N_INTERVALS):
        t_tag.append('_t'+str(i))

    n_instants = 0
    frames_in_interval = []
    first_frame = True
    for data in data_sequence:
        if n_instants == N_INTERVALS:
            break
        if not first_frame and math.fabs(data['timestamp'] - frames_in_interval[-1]['timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
            continue

        frames_in_interval.append(data)

        max_used_id = 0  # Initialise id counter (0 for the robot)
        # room (id 0)
        room_id = 0

        if first_frame:
            typeMap[room_id] = 'r'  # 'r' for 'room'
            position_by_id[room_id] = [0, 0]
            features[room_id, all_features.index('room')] = 1.
        
        features[room_id, all_features.index('step_fraction'+t_tag[n_instants])] = data['step_fraction']
        features[room_id, all_features.index('robot_adv_vel'+t_tag[n_instants])] = data['command'][0] / MAX_ADV
        features[room_id, all_features.index('robot_rot_vel'+t_tag[n_instants])] = data['command'][2] / MAX_ROT
        features[room_id, all_features.index('t'+str(n_instants))] = 1.

        max_used_id += 1        

        # humans
        for h in data['people']:
            h_id = max_used_id

            xpos = h['x'] / 10.
            ypos = h['y'] / 10.
            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = h['va'] / 10.
            vx = h['vx'] / 10.
            vy = h['vy'] / 10.
            orientation = h['a']

            if first_frame:
                src_nodes.append(h_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('p_r'))
                edge_norms.append([1. / len(data['people'])])

                src_nodes.append(room_id)
                dst_nodes.append(h_id)
                edge_types.append(rels.index('r_p'))
                edge_norms.append([1.])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('p_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_p')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)


                typeMap[h_id] = 'p'  # 'p' for 'person'
                position_by_id[h_id] = [xpos, ypos]                
                features[h_id, all_features.index('human')] = 1.

            max_used_id += 1


            features[h_id, all_features.index('step_fraction'+t_tag[n_instants])] = data['step_fraction']
            features[h_id, all_features.index('hum_orientation_sin'+t_tag[n_instants])] = math.sin(orientation)
            features[h_id, all_features.index('hum_orientation_cos'+t_tag[n_instants])] = math.cos(orientation)
            features[h_id, all_features.index('hum_x_pos'+t_tag[n_instants])] = xpos
            features[h_id, all_features.index('hum_y_pos'+t_tag[n_instants])] = ypos
            features[h_id, all_features.index('human_a_vel'+t_tag[n_instants])] = va
            features[h_id, all_features.index('human_x_vel'+t_tag[n_instants])] = vx
            features[h_id, all_features.index('human_y_vel'+t_tag[n_instants])] = vy
            features[h_id, all_features.index('hum_dist'+t_tag[n_instants])] = dist
            features[h_id, all_features.index('hum_inv_dist'+t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
            features[h_id, all_features.index('t'+str(n_instants))] = 1.


        # Goal
        goal_id = max_used_id
        max_used_id += 1

        xpos = data['goal'][0]['x'] / 10.
        ypos = data['goal'][0]['y'] / 10.
        dist = math.sqrt(xpos ** 2 + ypos ** 2)

        if first_frame:
            typeMap[goal_id] = 't'  # 't' for 'goal'
            src_nodes.append(goal_id)
            dst_nodes.append(room_id)
            edge_types.append(rels.index('t_r'))
            edge_norms.append([1.])
            # edge_norms.append([1. / len(data['objects'])])

            src_nodes.append(room_id)
            dst_nodes.append(goal_id)
            edge_types.append(rels.index('r_t'))
            edge_norms.append([1.])

            # Edge features
            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('t_r')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('r_t')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            position_by_id[goal_id] = [xpos, ypos]

            features[goal_id, all_features.index('goal')] = 1

        features[goal_id, all_features.index('step_fraction'+t_tag[n_instants])] = data['step_fraction']
        features[goal_id, all_features.index('goal_x_pos'+t_tag[n_instants])] = xpos
        features[goal_id, all_features.index('goal_y_pos'+t_tag[n_instants])] = ypos
        features[goal_id, all_features.index('goal_dist'+t_tag[n_instants])] = dist
        features[goal_id, all_features.index('goal_inv_dist'+t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
        features[goal_id, all_features.index('t'+str(n_instants))] = 1.

        # Walls
        if alt == '7':
            if not first_frame:
                walls, w_segments = generate_walls_information(data, w_segments)

            for wall in walls:
                wall_id = max_used_id
                max_used_id += 1

                if first_frame:
                    typeMap[wall_id] = 'w'  # 'w' for 'walls'

                    dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

                    # Links to room node
                    src_nodes.append(wall_id)
                    dst_nodes.append(room_id)
                    edge_types.append(rels.index('w_r'))
                    edge_norms.append([1. / len(walls)])

                    src_nodes.append(room_id)
                    dst_nodes.append(wall_id)
                    edge_types.append(rels.index('r_w'))
                    edge_norms.append([1.])

                    # Edge features
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('w_r')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('r_w')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                    position_by_id[wall_id] = [wall.xpos / 1000., wall.ypos / 1000.]
                    features[wall_id, all_features.index('wall')] = 1.

                features[wall_id, all_features.index('step_fraction'+t_tag[n_instants])] = data['step_fraction']
                features[wall_id, all_features.index('wall_orientation_sin'+t_tag[n_instants])] = math.sin(wall.orientation)
                features[wall_id, all_features.index('wall_orientation_cos'+t_tag[n_instants])] = math.cos(wall.orientation)
                features[wall_id, all_features.index('wall_x_pos'+t_tag[n_instants])] = wall.xpos / 1000.
                features[wall_id, all_features.index('wall_y_pos'+t_tag[n_instants])] = wall.ypos / 1000.
                features[wall_id, all_features.index('wall_dist'+t_tag[n_instants])] = dist
                features[wall_id, all_features.index('wall_inv_dist'+t_tag[n_instants])] = 1. - dist  # 1./(1.+dist*10.)


        n_instants += 1
        first_frame = False



    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           labels, []

# Initialize alternative 8: people, objects, goal, walls, interactions and time features. Can be combined with the grid 
def initializeAlt8(data_sequence, alt='8', w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data_sequence[0]['people']) + len(data_sequence[0]['objects']) + 1

    walls, w_segments = generate_walls_information(data_sequence[0], w_segments)
    n_nodes += len(walls)


    # Feature dimensions
    all_features, n_features = get_features(alt)
    # print(all_features, n_features)
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    if 'label_Q1' in data_sequence[0].keys():
        labels = np.array([float(data_sequence[0]['label_Q1']), float(data_sequence[0]['label_Q2'])])
    else:
        labels = np.array([0, 0])
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    t_tag = ['']
    for i in range(1, N_INTERVALS):
        t_tag.append('_t'+str(i))

    if 'step_fraction' in data_sequence[0].keys():
        step_fraction = data_sequence[0]['step_fraction']
    else:
        step_fraction = 0

    n_instants = 0
    frames_in_interval = []
    first_frame = True
    for data in data_sequence:
        if n_instants == N_INTERVALS:
            break
        if not first_frame and math.fabs(data['timestamp'] - frames_in_interval[-1]['timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
            continue

        if 'step_fraction' in data.keys():
            step_fraction = data['step_fraction']
        else:
            step_fraction = 0

        frames_in_interval.append(data)

        max_used_id = 0  # Initialise id counter (0 for the robot)
        # room (id 0)
        room_id = 0

        if first_frame:
            typeMap[room_id] = 'r'  # 'r' for 'room'
            position_by_id[room_id] = [0, 0]
            features[room_id, all_features.index('room')] = 1.
        
        features[room_id, all_features.index('step_fraction'+t_tag[n_instants])] = step_fraction
        features[room_id, all_features.index('robot_adv_vel'+t_tag[n_instants])] = data['command'][0] / MAX_ADV
        features[room_id, all_features.index('robot_rot_vel'+t_tag[n_instants])] = data['command'][2] / MAX_ROT
        features[room_id, all_features.index('t'+str(n_instants))] = 1.

        max_used_id += 1        

        # objects
        for o in data['objects']:
            o_id = o['id']

            xpos = o['x'] / 10.
            ypos = o['y'] / 10.

            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = o['va'] / 10.
            vx = o['vx'] / 10.
            vy = o['vy'] / 10.
            orientation = o['a']

            if first_frame:
                src_nodes.append(o_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('o_r'))
                edge_norms.append([1.])

                src_nodes.append(room_id)
                dst_nodes.append(o_id)
                edge_types.append(rels.index('r_o'))
                edge_norms.append([1.])
                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('o_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_o')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                typeMap[o_id] = 'o'  # 'o' for 'object'
                position_by_id[o_id] = [xpos, ypos]
                features[o_id, all_features.index('object')] = 1

            max_used_id += 1

            features[o_id, all_features.index('step_fraction'+t_tag[n_instants])] = step_fraction
            features[o_id, all_features.index('obj_orientation_sin'+t_tag[n_instants])] = math.sin(orientation)
            features[o_id, all_features.index('obj_orientation_cos'+t_tag[n_instants])] = math.cos(orientation)
            features[o_id, all_features.index('obj_x_pos'+t_tag[n_instants])] = xpos
            features[o_id, all_features.index('obj_y_pos'+t_tag[n_instants])] = ypos
            features[o_id, all_features.index('obj_a_vel'+t_tag[n_instants])] = va
            features[o_id, all_features.index('obj_x_vel'+t_tag[n_instants])] = vx
            features[o_id, all_features.index('obj_y_vel'+t_tag[n_instants])] = vy
            features[o_id, all_features.index('obj_x_size'+t_tag[n_instants])] = o['size_x']
            features[o_id, all_features.index('obj_y_size'+t_tag[n_instants])] = o['size_y']
            features[o_id, all_features.index('obj_dist'+t_tag[n_instants])] = dist
            features[o_id, all_features.index('obj_inv_dist'+t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)



        # humans
        for h in data['people']:
            h_id = h['id']

            xpos = h['x'] / 10.
            ypos = h['y'] / 10.
            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = h['va'] / 10.
            vx = h['vx'] / 10.
            vy = h['vy'] / 10.
            orientation = h['a']

            if first_frame:
                src_nodes.append(h_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('p_r'))
                edge_norms.append([1. / len(data['people'])])

                src_nodes.append(room_id)
                dst_nodes.append(h_id)
                edge_types.append(rels.index('r_p'))
                edge_norms.append([1.])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('p_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_p')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)


                typeMap[h_id] = 'p'  # 'p' for 'person'
                position_by_id[h_id] = [xpos, ypos]                
                features[h_id, all_features.index('human')] = 1.

            max_used_id += 1


            features[h_id, all_features.index('step_fraction'+t_tag[n_instants])] = step_fraction
            features[h_id, all_features.index('hum_orientation_sin'+t_tag[n_instants])] = math.sin(orientation)
            features[h_id, all_features.index('hum_orientation_cos'+t_tag[n_instants])] = math.cos(orientation)
            features[h_id, all_features.index('hum_x_pos'+t_tag[n_instants])] = xpos
            features[h_id, all_features.index('hum_y_pos'+t_tag[n_instants])] = ypos
            features[h_id, all_features.index('human_a_vel'+t_tag[n_instants])] = va
            features[h_id, all_features.index('human_x_vel'+t_tag[n_instants])] = vx
            features[h_id, all_features.index('human_y_vel'+t_tag[n_instants])] = vy
            features[h_id, all_features.index('hum_dist'+t_tag[n_instants])] = dist
            features[h_id, all_features.index('hum_inv_dist'+t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
            features[h_id, all_features.index('t'+str(n_instants))] = 1.


        # Goal
        goal_id = max_used_id
        max_used_id += 1

        xpos = data['goal'][0]['x'] / 10.
        ypos = data['goal'][0]['y'] / 10.
        dist = math.sqrt(xpos ** 2 + ypos ** 2)

        if first_frame:
            typeMap[goal_id] = 't'  # 't' for 'goal'
            src_nodes.append(goal_id)
            dst_nodes.append(room_id)
            edge_types.append(rels.index('t_r'))
            edge_norms.append([1.])
            # edge_norms.append([1. / len(data['objects'])])

            src_nodes.append(room_id)
            dst_nodes.append(goal_id)
            edge_types.append(rels.index('r_t'))
            edge_norms.append([1.])

            # Edge features
            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('t_r')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('r_t')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            position_by_id[goal_id] = [xpos, ypos]

            features[goal_id, all_features.index('goal')] = 1

        features[goal_id, all_features.index('step_fraction'+t_tag[n_instants])] = step_fraction
        features[goal_id, all_features.index('goal_x_pos'+t_tag[n_instants])] = xpos
        features[goal_id, all_features.index('goal_y_pos'+t_tag[n_instants])] = ypos
        features[goal_id, all_features.index('goal_dist'+t_tag[n_instants])] = dist
        features[goal_id, all_features.index('goal_inv_dist'+t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
        features[goal_id, all_features.index('t'+str(n_instants))] = 1.

        # Walls
        if not first_frame:
            walls, w_segments = generate_walls_information(data, w_segments)

        for wall in walls:
            wall_id = max_used_id
            max_used_id += 1

            if first_frame:
                typeMap[wall_id] = 'w'  # 'w' for 'walls'

                dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

                # Links to room node
                src_nodes.append(wall_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('w_r'))
                edge_norms.append([1. / len(walls)])

                src_nodes.append(room_id)
                dst_nodes.append(wall_id)
                edge_types.append(rels.index('r_w'))
                edge_norms.append([1.])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('w_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_w')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                position_by_id[wall_id] = [wall.xpos / 1000., wall.ypos / 1000.]
                features[wall_id, all_features.index('wall')] = 1.

            features[wall_id, all_features.index('step_fraction'+t_tag[n_instants])] = step_fraction
            features[wall_id, all_features.index('wall_orientation_sin'+t_tag[n_instants])] = math.sin(wall.orientation)
            features[wall_id, all_features.index('wall_orientation_cos'+t_tag[n_instants])] = math.cos(wall.orientation)
            features[wall_id, all_features.index('wall_x_pos'+t_tag[n_instants])] = wall.xpos / 1000.
            features[wall_id, all_features.index('wall_y_pos'+t_tag[n_instants])] = wall.ypos / 1000.
            features[wall_id, all_features.index('wall_dist'+t_tag[n_instants])] = dist
            features[wall_id, all_features.index('wall_inv_dist'+t_tag[n_instants])] = 1. - dist  # 1./(1.+dist*10.)
            features[wall_id, all_features.index('t'+str(n_instants))] = 1.            


        n_instants += 1
        first_frame = False

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        dist = math.sqrt((position_by_id[link['src']][0] - position_by_id[link['dst']][0]) ** 2 +
                         (position_by_id[link['src']][1] - position_by_id[link['dst']][1]) ** 2)

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLdir)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLinv)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           labels, []

def update_node_features(graphs, w_segments, data, ini_node, final_node, ini_edge, final_edge):
    # features should be updated in graphs.ndata['h']
    # w_segments is a list with the number of nodes uses to represent each wall
    # data is the json data for the current instant
    # ini_node is the index of the initial node to be updated
    # final_node is the index of the final node (not included) to be updated

    # Initialize variables
    # rels, num_rels = get_relations('3')
    # # print(num_rels)
    # edge_types = []  # List to store the relation of each edge
    # edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    i_w = 0
    for wall_segment in data['walls']:
        p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
        p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
        dist = np.linalg.norm(p1 - p2)
        v = (p2 - p1) / w_segments[i_w]
        for i in range(w_segments[i_w]):
            pa = p1 + v * i
            pb = p1 + v * (i + 1)
            inc2 = pb - pa
            midsp = (pa + pb) / 2
            walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        i_w += 1

    # Compute the number of nodes
    # one for the robot + room walls   + humans    + objects              + room(global node)
    # n_nodes = 1 + len(walls) + len(data['people']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('3')
    ##features = th.zeros(n_nodes, n_features)
    # edge_feats_list = []

    # Nodes variables
    # typeMap = dict()
    # position_by_id = {}

    labels = data['command']
    labels[0] = labels[0] / 100.
    labels[1] = labels[1] / 100.

    # room (id 0)
    room_id = 0
    # typeMap[room_id] = 'r'  # 'r' for 'room'
    # position_by_id[room_id] = [0, 0]
    # graphs.ndata['h'][ini_node+room_id, all_features.index('room_humans')] = len(data['people']) / MAX_HUMANS
    # graphs.ndata['h'][ini_node+room_id, all_features.index('room_humans2')] = (len(data['people']) ** 2) / (MAX_HUMANS ** 2)
    graphs.ndata['h'][ini_node+room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    graphs.ndata['h'][ini_node+room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1

    # humans
    for h in data['people']:
        # typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id += 1
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        # position_by_id[h['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = h['va'] / 10.
        vx = h['vx'] / 10.
        vy = h['vy'] / 10.

        orientation = h['a']

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        graphs.ndata['h'][ini_node+h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        graphs.ndata['h'][ini_node+h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        graphs.ndata['h'][ini_node+h['id'], all_features.index('hum_x_pos')] = xpos
        graphs.ndata['h'][ini_node+h['id'], all_features.index('hum_y_pos')] = ypos
        graphs.ndata['h'][ini_node+h['id'], all_features.index('human_a_vel')] = va
        graphs.ndata['h'][ini_node+h['id'], all_features.index('human_x_vel')] = vx
        graphs.ndata['h'][ini_node+h['id'], all_features.index('human_y_vel')] = vy
        graphs.ndata['h'][ini_node+h['id'], all_features.index('hum_dist')] = dist
        graphs.ndata['h'][ini_node+h['id'], all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # # Edge features
        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('p_r')] = 1
        # edge_features[-1] = dist
        # edge_feats_list.append(edge_features)

        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('r_p')] = 1
        # edge_features[-1] = dist
        # edge_feats_list.append(edge_features)
    # objects
    for o in data['objects']:
        # typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        # position_by_id[o['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']

        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_x_pos')] = xpos
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_y_pos')] = ypos
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_a_vel')] = va
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_x_vel')] = vx
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_y_vel')] = vy
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_x_size')] = o['size_x']
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_y_size')] = o['size_y']
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_dist')] = dist
        graphs.ndata['h'][ini_node+o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # # Edge features
        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('o_r')] = 1
        # edge_features[-1] = dist
        # edge_feats_list.append(edge_features)

        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('r_o')] = 1
        # edge_features[-1] = dist
        # edge_feats_list.append(edge_features)

    # Goal
    goal_id = max_used_id
    # typeMap[goal_id] = 'g'  # 'g' for 'goal'

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    # position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    graphs.ndata['h'][ini_node+goal_id, all_features.index('goal_x_pos')] = xpos
    graphs.ndata['h'][ini_node+goal_id, all_features.index('goal_y_pos')] = ypos
    graphs.ndata['h'][ini_node+goal_id, all_features.index('goal_dist')] = dist
    graphs.ndata['h'][ini_node+goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # # Edge features
    # edge_features = th.zeros(num_rels + 4)
    # edge_features[rels.index('g_r')] = 1
    # edge_features[-1] = dist
    # edge_feats_list.append(edge_features)

    # edge_features = th.zeros(num_rels + 4)
    # edge_features[rels.index('r_g')] = 1
    # edge_features[-1] = dist
    # edge_feats_list.append(edge_features)

    # walls
    wids = dict()
    for w_i, wall in enumerate(walls, 0):
        wall_id = max_used_id
        wids[wall] = wall_id
        # typeMap[wall_id] = 'w'  # 'w' for 'walls'
        max_used_id += 1

        dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

        # # Edge features
        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('w_r')] = 1
        # edge_features[-1] = dist
        # edge_feats_list.append(edge_features)

        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('r_w')] = 1
        # edge_features[-1] = dist
        # edge_feats_list.append(edge_features)


        # position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]

        graphs.ndata['h'][ini_node+wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        graphs.ndata['h'][ini_node+wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        graphs.ndata['h'][ini_node+wall_id, all_features.index('wall_x_pos')] = wall.xpos / 1000.
        graphs.ndata['h'][ini_node+wall_id, all_features.index('wall_y_pos')] = wall.ypos / 1000.
        graphs.ndata['h'][ini_node+wall_id, all_features.index('wall_dist')] = dist
        graphs.ndata['h'][ini_node+wall_id, all_features.index('wall_inv_dist')] = 1. - dist  # 1./(1.+dist*10.)


    # interaction links
    # for link in data['interaction']:
    #     typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
    #     typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

    #     dist = math.sqrt((position_by_id[link['src']][0] - position_by_id[link['dst']][0]) ** 2 +
    #                      (position_by_id[link['src']][1] - position_by_id[link['dst']][1]) ** 2)

    #     # Edge features
    #     edge_features = th.zeros(num_rels + 4)
    #     edge_features[rels.index(typeLdir)] = 1
    #     edge_features[-1] = dist
    #     edge_feats_list.append(edge_features)

    #     edge_features = th.zeros(num_rels + 4)
    #     edge_features[rels.index(typeLinv)] = 1
    #     edge_features[-1] = dist
    #     edge_feats_list.append(edge_features)


    # # self edges
    # for node_id in range(n_nodes):

    #     # Edge features
    #     edge_features = th.zeros(num_rels + 4)
    #     edge_features[rels.index('self')] = 1
    #     edge_features[-1] = 0
    #     edge_feats_list.append(edge_features)

    # # Convert outputs to tensors
    # edge_feats = th.stack(edge_feats_list)

    # graphs.edata['he'][ini_edge:final_edge:,] = edge_feats
    



#################################################################
# Class to load the dataset
#################################################################

class SocNavDataset(DGLDataset):
    grid_data = graphData(*generate_grid_graph_data('7'))
    default_alt = '7'

    def __init__(self, path, alt, mode='train', raw_dir='data/', prev_graph = None, init_line=-1, end_line=-1, loc_limit=limit,
                 force_reload=False, verbose=True, debug=False):
        if type(path) is str:
            self.path = raw_dir + path
        else:
            self.path = path
        self.mode = mode
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.prev_graph = prev_graph
        self.graphs = []
        self.labels = []
        self.data = dict()
        # self.grid_data = None
        self.data['typemaps'] = []
        self.data['coordinates'] = []
        self.data['identifiers'] = []
        self.data['wall_segments'] = []
        self.debug = debug
        self.limit = loc_limit
        self.force_reload = force_reload

        if self.mode == 'test':
            self.force_reload = True

        # Define device. GPU if it is available
        self.device = 'cpu'

        if self.debug:
            self.limit = 1 + (0 if init_line == -1 else init_line)
        
        if self.default_alt != self.alt:
            self.default_alt = self.alt
            if self.alt == '7' or self.alt == '8':
                self.grid_data = graphData(*generate_grid_graph_data(self.alt))

        if self.alt == '1':
            self.dataloader = initializeAlt1
        elif self.alt == '2':
            self.dataloader = initializeAlt2
            # self.grid_data = graphData(*generate_grid_graph_data('2'))
        elif self.alt == '3':
            self.dataloader = initializeAlt3
        elif self.alt == '4':
            self.dataloader = initializeAlt4
        elif self.alt == '5':
            self.dataloader = initializeAlt5
        elif self.alt == '6':
            self.dataloader = initializeAlt6
        elif self.alt == '7':
            self.dataloader = initializeAlt6
        elif self.alt == '8' or self.alt == '9':
            self.dataloader = initializeAlt8

        else:
            print('Introduce a valid initialize alternative')
            sys.exit(-1)

        super(SocNavDataset, self).__init__("SocNav", raw_dir=raw_dir, force_reload=self.force_reload, verbose=verbose)

    def get_dataset_name(self):
        graphs_path = 'graphs_' + self.mode + '_alt_' + self.alt + '_s_' + str(limit) + '.bin'
        info_path = 'info_' + self.mode + '_alt_' + self.alt + '_s_' + str(limit) + '.pkl'
        return graphs_path, info_path

    def generate_final_graph(self, raw_data):
        rels, num_rels = get_relations(self.alt)
        room_graph_data = graphData(*self.dataloader(raw_data, self.alt))
        if self.grid_data is not None:
            # Merge room and grid graph
            src_nodes = th.cat([self.grid_data.src_nodes,(room_graph_data.src_nodes + self.grid_data.n_nodes)], dim=0)
            dst_nodes = th.cat([self.grid_data.dst_nodes, (room_graph_data.dst_nodes + self.grid_data.n_nodes)], dim=0)
            edge_types = th.cat([self.grid_data.edge_types, room_graph_data.edge_types], dim=0)
            edge_norms = th.cat([self.grid_data.edge_norms, room_graph_data.edge_norms], dim=0)
            if WITH_EDGE_FEATURES:
                edge_feats = th.cat([self.grid_data.edge_feats, room_graph_data.edge_feats], dim=0)
                edge_feats_list = []

            # Link each node in the room graph to the correspondent grid graph.
            for r_n_id in range(0, room_graph_data.n_nodes):
                r_n_type = room_graph_data.typeMap[r_n_id]
                x, y = room_graph_data.position_by_id[r_n_id]
                closest_grid_nodes_id = closest_grid_nodes(self.grid_data.labels, area_width, grid_width, area_width, x * 10000, y * 10000)
                # closest_grid_nodes_id = [closest_grid_node(self.grid_data.labels,area_width, grid_width, x * 10000, y * 10000)]
                for g_id in closest_grid_nodes_id:
                    if g_id is not None:
                        src_nodes = th.cat([src_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)
                        dst_nodes = th.cat([dst_nodes, th.tensor([r_n_id + self.grid_data.n_nodes], dtype=th.int32)], dim=0)
                        edge_types = th.cat([edge_types, th.LongTensor([rels.index('g_' + r_n_type)])], dim=0)
                        edge_norms = th.cat([edge_norms, th.Tensor([[1.]])])
                        if WITH_EDGE_FEATURES:
                            new_edge_features = th.zeros(num_rels + 4)
                            new_edge_features[rels.index('g_'+ r_n_type)] = 1
                            # new_edge_features[-1] = dist
                            edge_feats_list.append(new_edge_features)


                        src_nodes = th.cat([src_nodes, th.tensor([r_n_id + self.grid_data.n_nodes], dtype=th.int32)], dim=0)
                        dst_nodes = th.cat([dst_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)
                        edge_types = th.cat([edge_types, th.LongTensor([rels.index(r_n_type + '_g')])], dim=0)
                        edge_norms = th.cat([edge_norms, th.Tensor([[1.]])])
                        if WITH_EDGE_FEATURES:
                            new_edge_features = th.zeros(num_rels + 4)
                            new_edge_features[rels.index(r_n_type + '_g')] = 1
                            # new_edge_features[-1] = dist
                            edge_feats_list.append(new_edge_features)


            # Compute typemaps, coordinates, number of nodes, features and labels for the merged graph.
            n_nodes = room_graph_data.n_nodes + self.grid_data.n_nodes
            typeMapRoomShift = dict()
            coordinates_roomShift = dict()
            if WITH_EDGE_FEATURES:
                edge_feats_list = th.stack(edge_feats_list)
                edge_feats = th.cat([edge_feats, edge_feats_list], dim=0)

            for key in room_graph_data.typeMap:
                typeMapRoomShift[key + len(self.grid_data.typeMap)] = room_graph_data.typeMap[key]
                coordinates_roomShift[key + len(self.grid_data.position_by_id)] = room_graph_data.position_by_id[key]

            position_by_id = {**self.grid_data.position_by_id, **coordinates_roomShift}
            typeMap = {**self.grid_data.typeMap, **typeMapRoomShift}
            labels = room_graph_data.labels
            features = th.cat([self.grid_data.features, room_graph_data.features], dim=0)
        else:
            src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, \
            position_by_id, typeMap, labels, wall_segments = room_graph_data

        self.data['typemaps'].append(typeMap)
        self.data['coordinates'].append(position_by_id)
        self.data['identifiers'].append(raw_data[0]['ID'])

        self.labels.append(labels)

        try:
            final_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=n_nodes, idtype=th.int32, device=self.device)
            final_graph.ndata['h'] = features.to(self.device)
            edge_types.to(self.device, dtype=th.long)
            edge_norms.to(self.device, dtype=th.float64)
            if WITH_EDGE_FEATURES:
                edge_feats.to(self.device, dtype=th.float64)
                final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms, 'he': edge_feats})
            else:
                final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms})
            return final_graph
        except Exception:
            raise

    def load_one_graph(self, data):
        # For alternatives 6, 7, 8 and 9 do not create several-instants graphs
        if self.alt == '6'or self.alt == '7' or self.alt == '8' or self.alt == '9':
            graph_data = graphData(*self.dataloader(data))
            src_nodes, dst_nodes, n_nodes, feats, edge_feats, edge_types, edge_norms, coordinates, typeMap, labels, w_segments = graph_data
        else:
            graph_data = graphData(*self.dataloader(data[0]))
            w_segments = graph_data.w_segments
            graphs_in_interval = [graph_data]
            frames_in_interval = [data[0]]
            for frame in data[1:]:
                if len(graphs_in_interval) == N_INTERVALS:
                    break
                if math.fabs(frame['timestamp'] - frames_in_interval[-1][
                    'timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
                    continue
                graphs_in_interval.append(graphData(*self.dataloader(frame, w_segments)))
                frames_in_interval.append(frame)

            src_nodes, dst_nodes, edge_types, edge_norms, n_nodes, feats, edge_feats, typeMap, coordinates = \
                self.merge_graphs(graphs_in_interval)
            labels = graphs_in_interval[0].labels
        try:
            # Create merged graph:
            final_graph = dgl.graph((src_nodes, dst_nodes),
                                    num_nodes=n_nodes,
                                    idtype=th.int32, device=self.device)

            # Add merged features and update edge labels:
            final_graph.ndata['h'] = feats.to(self.device)
            final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms, 'he': edge_feats})

            # Append final data
            self.graphs.append(final_graph)
            self.labels.append(labels)
            self.data['typemaps'].append(typeMap)
            self.data['coordinates'].append(coordinates)
            self.data['identifiers'].append(data[0]['ID'])
            self.data['wall_segments'].append(w_segments)

        except Exception:
            print("Error loading one graph")
            raise

    def generate_from_previous_graph(self, data):
        self.graphs = copy.deepcopy(self.prev_graph.graphs)
        self.labels = copy.deepcopy(self.prev_graph.labels)
        self.data = copy.deepcopy(self.prev_graph.data)

        # print("Graph:",self.graphs)
        # print("******************************")
        # print("Labels:",self.labels)
        # print("******************************")
        # print("Data:",self.data)
        # print("******************************")
        #

        num_edges = len(self.graphs[0].edges()[0])
        ini_edge = 0
        final_edge = num_edges
        nnodes_in_interval = int(self.graphs[0].number_of_nodes()/N_INTERVALS)
        update_node_features(self.graphs[0], self.data['wall_segments'][0],
                             data[0], 0, nnodes_in_interval, 0, num_edges)
        # print("NODES IN THE GRAPH::",nnodes_in_interval)
        # print("******************************")
        n_graphs = 1

        initial_node = nnodes_in_interval
        final_node = initial_node + nnodes_in_interval
        frames_in_interval = [data[0]]
        # print(frames_in_interval)
        for frame in data[1:]:
            if n_graphs == N_INTERVALS:
                break
            if math.fabs(frame['timestamp'] - frames_in_interval[-1][
                'timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
                continue
            update_node_features(self.graphs[0], self.data['typemaps'][0], self.data['wall_segments'][0],
                                 frame, initial_node, final_node, ini_edge, final_edge)
            frames_in_interval.append(frame)
            n_graphs += 1
            initial_node += nnodes_in_interval
            final_node += nnodes_in_interval
            ini_edge += num_edges
            final_edge += num_edges


    def merge_graphs(self, graphs_in_interval):
        all_features, n_features = get_features(self.alt)
        new_features = ['is_t_0', 'is_t_m1', 'is_t_m2']
        f_list = []
        src_list = []
        dst_list = []
        edge_types_list = []
        edge_norms_list = []
        edge_feats_list = []
        typeMap = dict()
        coordinates = dict()
        n_nodes = 0
        rels, num_rels = get_relations(self.alt)
        g_i = 0
        offset = graphs_in_interval[0].n_nodes
        for g in graphs_in_interval:
            # Shift IDs of the typemap and coordinates lists
            for key in g.typeMap:
                typeMap[key + (offset * g_i)] = g.typeMap[key]
                coordinates[key + (offset * g_i)] = g.position_by_id[key]
            n_nodes += g.n_nodes
            f_list.append(g.features)
            edge_feats_list.append(g.edge_feats)
            # Add temporal edges
            src_list.append(g.src_nodes + (offset * g_i))
            dst_list.append(g.dst_nodes + (offset * g_i))
            edge_types_list.append(g.edge_types)
            edge_norms_list.append(g.edge_norms)
            if g_i > 0:
                # Temporal connections and edges labels
                new_src_list = []
                new_dst_list = []
                new_etypes_list = []
                new_enorms_list = []
                new_edge_feats_list = []
                for node in range(g.n_nodes):
                    new_src_list.append(node + offset * (g_i - 1))
                    new_dst_list.append(node + offset * g_i)
                    new_etypes_list.append(num_rels + (g_i - 1) * 2)
                    new_enorms_list.append([1.])

                    new_src_list.append(node + offset * g_i)
                    new_dst_list.append(node + offset * (g_i - 1))
                    new_etypes_list.append(num_rels + (g_i - 1) * 2 + 1)
                    new_enorms_list.append([1.])

                    # Edge features
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[num_rels + (g_i - 1) * 2] = 1
                    edge_features[-1] = 0
                    new_edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[num_rels + (g_i - 1) * 2 + 1] = 1
                    edge_features[-1] = 0
                    new_edge_feats_list.append(edge_features)

                new_edge_feats = th.stack(new_edge_feats_list)

                src_list.append(th.IntTensor(new_src_list))
                dst_list.append(th.IntTensor(new_dst_list))
                edge_types_list.append(th.LongTensor(new_etypes_list))
                edge_norms_list.append(th.Tensor(new_enorms_list))
                edge_feats_list.append(new_edge_feats)
            if N_INTERVALS>1:
                for f in new_features:
                    if g_i == new_features.index(f):
                        g.features[:, all_features.index(f)] = 1
                    else:
                        g.features[:, all_features.index(f)] = 0
            g_i += 1

        src_nodes = th.cat(src_list, dim=0)
        dst_nodes = th.cat(dst_list, dim=0)
        edge_types = th.cat(edge_types_list, dim=0)
        edge_norms = th.cat(edge_norms_list, dim=0)
        edge_feats = th.cat(edge_feats_list, dim=0)
        feats = th.cat(f_list, dim=0)

        return src_nodes, dst_nodes, edge_types, edge_norms, n_nodes, feats, edge_feats, typeMap, coordinates

    #################################################################
    # Implementation of abstract methods
    #################################################################

    def download(self):
        # No need to download any data
        pass

    def process(self):
        if type(self.path) is str and self.path.endswith('.json'):
            linen = -1
            for line in open(self.path).readlines():
                if linen % 1000 == 0:
                    print(linen)

                if linen + 1 >= self.limit:
                    print('Stop including more samples to speed up dataset loading')
                    break
                linen += 1
                if self.init_line >= 0 and linen < self.init_line:
                    continue
                if linen > self.end_line >= 0:
                    continue

                raw_data = json.loads(line)
                final_graph = self.generate_final_graph(raw_data)
                self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)

        elif type(self.path) is str and self.path.endswith('.txt'):
            linen = -1
            print(self.path)
            with open(self.path) as set_file:
                ds_files = set_file.read().splitlines()

            print("number of files for ", self.path, len(ds_files))

            for ds in ds_files:
                if linen % 1000 == 0:
                    print(linen)
                with open(ds) as json_file:
                    data = json.load(json_file)
                if self.alt != '7' and self.alt != '8':
                    self.load_one_graph(data)
                else:
                    final_graph = self.generate_final_graph(data)
                    self.graphs.append(final_graph)
                if linen + 1 >= limit:
                    print('Stop including more samples to speed up dataset loading')
                    break
                linen += 1
            self.labels = th.tensor(self.labels, dtype=th.float64)

        elif type(self.path) == list and len(self.path) >= 1:
            if self.prev_graph is None:
                if self.alt != '7' and self.alt != '8':
                    self.load_one_graph(self.path)
                else:
                    final_graph = self.generate_final_graph(self.path)
                    self.graphs.append(final_graph)
            else:
                num_instants = sum(map(('r').__eq__, self.prev_graph.data['typemaps'][0].values()))
                if num_instants==N_INTERVALS:
                    self.generate_from_previous_graph(self.path)
                else:
                    self.load_one_graph(self.path)
            self.labels = th.tensor(self.labels, dtype=th.float64)
        elif type(self.path) == list and type(self.path[0]) == str:
            raw_data = json.loads(self.path)
            final_graph = self.generate_final_graph(raw_data)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)
        else:
            final_graph = self.generate_final_graph(self.path)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug:
            return
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        # Save graphs
        save_graphs(graphs_path, self.graphs, {'labels': self.labels})

        # Save additional info
        save_info(info_path, {'typemaps': self.data['typemaps'],
                              'coordinates': self.data['coordinates'],
                              'identifiers': self.data['identifiers']})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        # Load graphs
        self.graphs, label_dict = load_graphs(graphs_path)
        self.labels = label_dict['labels']

        # Load info
        self.data['typemaps'] = load_info(info_path)['typemaps']
        self.data['coordinates'] = load_info(info_path)['coordinates']
        self.data['identifiers'] = load_info(info_path)['identifiers']

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)
