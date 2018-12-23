import networkx as nx
import os
from collections import Counter
import numpy as np
import random

from main import replace

def connect_graph(graph):
    '''
    Makes a graph connected.
    If a graph has more than 1 connected componet,
    it makes an arbitrary edge between two components.
    It does so between neighboring connected components.
    :param graph: undirected graph
    :return:
    '''
    ccs = map(list, nx.connected_components(graph))
    curr = next(ccs)
    for foll in ccs:
        u, v = random.choice(curr), random.choice(foll)
        graph.add_edge(u, v)
    return graph

def generate_ER_graphs(n_graphs,
                       n_vertices,
                       prob,
                       save_to_files=None, graph_dir=None):
    '''Generate Erdos-Renyi graphs of the same order.
    Optionally saves graphs to the folder.
    It will make sure to connect components by indtroducing new edges.
    However, one may think to set the parameters n_vertices and prob,
    to control the number of vertices in the largest component.
    For example, it's known that if p >= 1/n, then a graph will most
    likely have a single connected component with all n vertices.

    :param n_graphs: number of graphs to generate
    :param n_vertices: number of vertices in each graph (ER parameter)
    :param prob: probability of having edge between arbitrary 2 nodes (ER parameter)
    :param save_to_files: True or None
    :param graph_dir: directory to save files (only if save_to_files=True)
    :return:
    '''
    graphs = [connect_graph(nx.erdos_renyi_graph(n_vertices, prob))
              for _ in range(n_graphs)]
    if save_to_files:
        if not os.path.exists(graph_dir):
            os.mkdir(graph_dir)
        [nx.write_edgelist(G, '{}/{}.edgelist'.format(graph_dir, ix)) for ix, G in enumerate(graphs)]
    return graphs

def generate_anonymous_walks(n_vertices, save_to_file=None, path=None):
    '''Generates anonymous walks that are tested by the algorithm.
    All of these walks potentially can be tested by the algorithm.
    However, clearly if walk 0-1-0 does not exist in your graph,
    then walk 0-1-2-1-0 also does not exist. So this set will be
    excessive for some real graphs. In fact, the cardinality of
    returned set is exponential to the number of vertices n, while
    the number of requests to this set will be quadratic to n.
    Also, some of the anonymous walks that can be tested by the
    algorithm also not included in this set. In particular,
    it's not tested when previous nodes are connected to some
    combination of the following nodes. Also, it's not included,
    when some interconnected edges go with edges that go from
    the previous to the new indeces.
    :param n_vertices: number of vertices in a graph
    :return:
    '''
    walks = []
    # 1. longest walks
    walks += [list(range(ix)) + list(range(ix-2, -1, -1)) for ix in range(1, n_vertices+1)]

    # 2. we go back in a walk, and try to replace a node with a new index
    branched_walks = []
    for w in walks:
        max_index = max(w)

        # check if previous nodes can be connected to new nodes (upcoming edges)
        new_walk = w.copy()
        for source in range(max_index - 1, -1, -1):
            for target in range(max_index + 1, max_index+(max_index-source) + 1):
                if target < n_vertices:
                    new_walk = replace(new_walk, source, target)
                    branched_walks.append(new_walk)

        # cover interconnections between already added nodes (interconnected edges)
        new_walk = w.copy()
        for source in range(max_index - 1, -1, -1):
            for target in range(source + 2, max_index+1):
                new_walk = replace(new_walk, source, target)
                branched_walks.append(new_walk)

    walks += branched_walks

    if save_to_file:
        with open(path, 'w+') as f:
            for w in walks:
                f.write(','.join(list(map(str, w))) + '\n')

    return walks + branched_walks

if __name__ == '__main__':

    NG = 100 # number of graphs
    NV = 6 # number of vertices


    graphs = generate_ER_graphs(NG, NV, prob=0.2, save_to_files=True, graph_dir='er_graphs')
    print('Statistics on generated graphs')
    print('Nodes histogram:', Counter([_.order() for _ in graphs]))
    print('Nodes mean:', np.mean([_.order() for _ in graphs]))
    print('Edges histogram:', Counter([_.size() for _ in graphs]))
    print('Edges mean:', np.mean([_.size() for _ in graphs]))

    if not os.path.exists('aw/'):
        os.mkdir('aw/')
    aw = generate_anonymous_walks(NV, save_to_file=True, path='aw/aw{}.txt'.format(NV))


    console = []