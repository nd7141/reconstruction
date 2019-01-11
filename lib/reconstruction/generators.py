import networkx as nx
import os
from collections import Counter
import numpy as np
import random
import pandas as pd


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


def generate_regular_graphs(n_graphs, n_vertices, degree, graph_dir=None):
    '''
    Generates random graphs with same degree.
    See here for the number of distinct graphs for n and d.
    For example for n = 8 and d = 3, there are 5 graphs.
    :param n_graphs: number of graphs to generate
    :param n_vertices: number of vertices in each graphs
    :param degree: degree of each vertex
    :param save_to_files: where to save files
    :param graph_dir: graph directory
    :return:
    '''
    graphs = [nx.random_regular_graph(degree, n_vertices)
              for _ in range(n_graphs)]
    if graph_dir:
        print(graph_dir)
        if not os.path.exists(graph_dir):
            os.mkdir(graph_dir)
        [nx.write_edgelist(G, '{}/{}.edgelist'.format(graph_dir, ix)) for ix, G in enumerate(graphs)]
    return graphs


def read_dimacs_graph(filename):
    G = nx.Graph()
    with open(filename) as f:
        for line in f:
            if line.startswith('p'):
                continue
            elif line.startswith('e'):
                splitted = line.strip().split(' ')
                G.add_edge(splitted[1], splitted[2])
            else:
                print('Found bad line start: ', line)
                raise Exception
    return G

def relabel_graph(G):
    nodes = list(G.nodes())
    new_order = random.sample(G.nodes(), G.order())
    mapping = {nodes[i]: new_order[i] for i in range(len(new_order))}
    return nx.relabel_nodes(G, mapping)

def generate_regular_dataset(graph_dir, nv_range, ng_range, degree):
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    sizes = nv_range
    orders = ng_range
    D = degree
    for ix in range(10):
        print(ix)
        NG = orders[ix]
        NV = sizes[ix]
        graphs = generate_regular_graphs(NG, NV, degree=D)
        if NV <= 50:
            total = 0
            for l in range(len(graphs) - 1):
                is_iso = nx.is_isomorphic(graphs[l], graphs[l + 1])
                total += is_iso * 1
            if total > 0:
                raise Exception("Found iso graphs in random regular graphs {} {}.\nRepeat experiment.".format(NG, NV))

        for t in range(len(graphs)):
            for c in range(5):
                g_copy = relabel_graph(graphs[t])
                nx.write_edgelist(g_copy, './regulars/regular_n{}_d{}_t{}_c{}.edgelist'.format(NV, D, t, c))


if __name__ == '__main__':

    NG = 1000 # number of graphs
    NV = 80 # number of vertices
    D = 3 # degree of vertices

    # graphs = generate_ER_graphs(NG, NV, prob=0.2, save_to_files=True, graph_dir='er_graphs_n70')
    # graphs = generate_regular_graphs(NG, NV, D, save_to_files=True, graph_dir='reg_graphs_n8_d3')
    # print('Statistics on generated graphs')
    # print('Nodes histogram:', Counter([_.order() for _ in graphs]))
    # print('Nodes mean:', np.mean([_.order() for _ in graphs]))
    # print('Edges histogram:', Counter([_.size() for _ in graphs]))
    # print('Edges mean:', np.mean([_.size() for _ in graphs]))

    # if not os.path.exists('aw/'):
    #     os.mkdir('aw/')
    # aw = generate_anonymous_walks(NV, save_to_file=True, path='aw/aw{}.txt'.format(NV))

    # with open('aw/aw20.txt') as f:
    #     ls= [len(line.strip().split(',')) for line in f]
    # print(Counter(ls))

    # generate regular dataset
    # D = 3
    # sizes = list(range(10, 101, 10))
    # orders = [5, 15, 20, 20, 20, 20, 20, 20, 20, 40]
    # generate_regular_dataset('regulars/', sizes, orders, D)


    # generate_regular_graphs(10, NV, D, graph_dir='./reg_graphs_n80_d3/')

    # data_dir = '../../cmz/'
    # save_dir = '../../cmz-edges/'
    #
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    #
    # sizes = []
    # for fn in os.listdir(data_dir):
    #     if fn.startswith('cmz'):
    #         G = read_dimacs_graph(data_dir + fn)
    #         sizes.append((G.order(), G.size()))
    #         nx.write_edgelist(G, save_dir + fn)
    #
    #
    # print(Counter(sizes))

    # # G = read_dimacs_graph('data/cfi-rigid-z2/cfi-rigid-z2-0088-01-1')
    #
    # if not os.path.exists(data_dir + '-edges/'):
    #     os.mkdir(data_dir + '-edges/')
    #
    # files = os.listdir(data_dir)
    # for i, fn in enumerate(files):
    #     print(i)
    #     G = read_dimacs_graph(data_dir + '/' + fn)
    #     nx.write_edgelist(G, '{}-edges/{}'.format(data_dir, fn))
    #
    # print(G.order(), G.size())