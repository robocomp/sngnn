import sys
import random
from datetime import datetime

from socnav import SocNavDataset
import networkx as nx
import matplotlib.pyplot as plt

from tabulate import tabulate


def split_by_types(graph):
    ret = dict()
    for t in [ 'r', 'p', 'o', 'l', 'w', 'i' ]:
        ret[t] = []
    for k in sorted(graph.typeMap.keys()):
        ret[graph.typeMap[k]].append(k)
    return ret

def colours_for_types():
    ret = dict()
    ret['r'] = 'r'
    ret['p'] = 'b'
    ret['o'] = 'g'
    ret['l'] = 'm'
    ret['w'] = 'c'
    ret['i'] = 'y'
    return ret

def convert_one_hot_to_character(row, x, graph_type):
    if row[x] < 0.5:
        return ' '
    if graph_type == 'raw' or graph_type == '1' or graph_type == '2':
        return 'rhoi'[x]
    if graph_type == '3' or graph_type == '4':
        return 'rhoilw'[x]
    if graph_type == 'relational':
        return 'rholw'[x]

def one_hots(graph_type):
    if graph_type == 'raw' or graph_type == '1' or graph_type == '2':
        return 4
    if graph_type == '3' or graph_type == '4':
        return 6
    if graph_type == 'relational':
        return 5

def table_for_graph(graph, graph_type):
    node_descriptor_table = []
    node_descriptor_table.append(graph.node_descriptor_header)
    for row in graph.features:
        first = [convert_one_hot_to_character(row, x, graph_type) for x in range(one_hots(graph_type))]
        last = ['{:1.3f}'.format(x) for x in row[one_hots(graph_type):]]
        node_descriptor_table.append(first + last)
    node_descriptor_table = tabulate(node_descriptor_table)
    return node_descriptor_table


training_file = sys.argv[1]
graph_type = sys.argv[2]

train_dataset = SocNavDataset(training_file, mode='train', alt=graph_type, end_line=0, init_line=0)

for index in range(len(train_dataset)):
    graph, features, labels = train_dataset[index]
    print(table_for_graph(graph, graph_type=graph_type))
    nx_G = graph.to_networkx()
    random.seed(datetime.now())
    #Function call for node colouring : Add parameter : node_color
    pos = nx.spring_layout(nx_G, k=30, pos=None, fixed=None, iterations=15000, threshold=0.000001, seed=random.getrandbits(32))

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    # nx.draw(nx_G, pos, with_labels=True)
    # nx.draw(nx_G, pos, node_color = colours_for_types(graph.typeMap),with_labels=True)

    types_split = split_by_types(graph)

    for r in types_split.keys():
        nx.draw_networkx_nodes(nx_G, pos, with_labels=True,
                       nodelist=types_split[r],
                       node_color=colours_for_types()[r],
                       node_size=200,
                       alpha=0.8)

    nx.draw_networkx_edges(nx_G, pos, width=1.0, alpha=0.5)

    nx.draw_networkx_labels(nx_G, pos, graph.typeMap, font_size=12)
    plt.axis('off')
    plt.show()
