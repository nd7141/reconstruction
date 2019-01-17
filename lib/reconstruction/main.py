import os
import time

import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
import sklearn

import re

from lib.reconstruction.AnonymousWalkKernel import AnonymousWalks as AW
from collections import defaultdict as ddict, Counter
import random

def support(P):
    '''Reconstruct graph from the walk.'''
    G = nx.Graph()
    G.add_node(P[0])
    for i in range(len(P) - 1):
        G.add_edge(P[i], P[i + 1])
    return G


def replace(P, source, target):
    '''Replace last occurrence of source with source-target-source.'''
    assert source in P
    ix = len(P) - P[::-1].index(source)
    return P[:ix] + [target, P[ix - 1]] + P[ix:]


def reconstruct(Dl, radius):
    '''Reconstruct graph from distribution of anonymous walks.'''
    P = [0]  # initial support walk
    balls = [nx.Graph()]  # reconstructed balls
    S = [] # supports for each radius
    for r in range(radius):
        # grow the ball with the new radius
        B = support(P)  # ball of radius r
        S.append(P)
        balls.append(B)
        border = sorted(set(balls[r+1].nodes()).difference(balls[r].nodes()))  # nodes in the periphery of the ball
        for i in range(len(border)):  # pick new border nodes
            # check the neighbors of the border nodes
            x = max(P) + 1  # next node that does not yet exist in the ball
            rpl = replace(P, border[i], x)  # replacement of border node u with u-x-u
            while tuple(rpl) in Dl:
                # check that replacement exists (i.e. there is an edge between x and border node)
                P = rpl  # change the supporting walk if replacement exists
                for u in set(border[i + 1:]).union(range(len(B), x)):
                    # check if new node x is connected to any other border nodes or
                    # to nodes that were connected previously for radius r0
                    rpl = replace(P, x, u)
                    if tuple(rpl) in Dl:
                        P = rpl
                x = max(P) + 1  # go to another node x that is potentially a neighbor of u
                rpl = replace(P, border[i], x)  # check that x is replacement

    balls.append(support(P))
    S.append(P)
    return balls, S


def reconstruct_dfs(Dl):
    P = [0] # supporting walk
    S = [0] # stack of nodes to check
    checked = dict() # nodes that has been checked for edge
    while len(S) > 0: # grow supporting walk in DFS manner
        curr = S[-1]
        x = max(P) + 1 # next node to check
        for u in range(curr+1, x): # u is already in the supporting walk
            # check if there is connection to already discovered nodes
            if u not in checked[curr]: # see if we already checked this edge
                rpl = replace(P, curr, u)
                if tuple(rpl) in Dl:
                    P = rpl # add additional edge to the support walk
                checked.setdefault(curr, set()).add(u)

        # check if the current node is connected to a not-yet-discovered node
        rpl = replace(P, curr, x)
        if tuple(rpl) in Dl: # move one level up in the walk
            checked.setdefault(curr, set()).add(x)
            S.append(x)
            P = rpl
        else: # move one level down in the walk
            S.pop()
    return P


def reconstruct_dfs2(graph, source, verbose=False):
    P = [0] # supporting walk
    S = [0] # stack of nodes to check
    checked = dict() # nodes that has been checked for edge
    while len(S) > 0: # grow supporting walk in DFS manner
        curr = S[-1]
        x = max(P) + 1 # next node to check
        for u in range(curr+1, x): # u is already in the supporting walk
            # check if there is connection to already discovered nodes
            if u not in checked[curr]: # see if we already checked this edge
                rpl = replace(P, curr, u)
                if check_aw_in_graph(graph, source, rpl):
                    P = rpl # add additional edge to the support walk
                checked.setdefault(curr, set()).add(u)

        # check if the current node is connected to a not-yet-discovered node
        rpl = replace(P, curr, x)
        if check_aw_in_graph(graph, source, rpl): # move one level up in the walk
            checked.setdefault(curr, set()).add(x)
            S.append(x)
            P = rpl
        else: # move one level down in the walk
            S.pop()
        if verbose:
            print('Current support:', P)
    return P

def covering_walk(graph, source):
    P = [0]  # supporting walk
    S = [0]  # stack of nodes to check
    node2anon = {source: 0}
    anon2node = {0: source}
    checked = dict()  # nodes that has been checked for edge
    degrees = graph.degree()
    while len(S) > 0:  # grow supporting walk in DFS manner
        curr = S[-1]
        x = max(P) + 1  # next node to check

        # check if there is a node in the neighborhood that has not been explored yet
        Ncurr = list(nx.neighbors(graph, anon2node[curr]))
        if random.uniform(0, 1) < 0.99:
            random.shuffle(Ncurr)  # option 1: random order
        else:
            Ncurr = sorted(Ncurr, key=lambda v: degrees[v], reverse=True)  # option 2: top-degree
            # Ncurr = sorted(Ncurr, key=lambda v: degrees[v], reverse=False)  # option 3: low-degree
        # print(anon2node[curr], Ncurr)
        for neighbor in Ncurr:
            if neighbor in node2anon:
                continue  # already visited
            else:
                node2anon[neighbor] = x
                anon2node[x] = neighbor
                S.append(x)
                checked.setdefault(curr, set()).add(x)
                P = replace(P, curr, x)  # move to it
                break
        else:
            S.pop()  # move back in the stack

        for u in range(x-1, curr, -1):  # u is already in the supporting walk
            # check if there is connection to already discovered nodes
            if u not in checked[curr]:  # see if we already checked this edge
                if anon2node[u] in graph[anon2node[curr]]:
                    P = replace(P, curr, u)
                checked.setdefault(curr, set()).add(u)

    cover = [anon2node[v] for v in P]
    return cover, P

def cover2(graph, source):
    random_walk = [source]
    checked = ddict(list)
    stack = [source]
    visited = {source}
    ranks = {0: source} # to attempt to get maximal cover (possible to do without rank, but then no guarantees on maximality)
    revranks = {source: 0}

    while len(stack) > 0:
        last = stack[-1]
        lastrank = revranks[last]
        maxrank = max(ranks.keys()) + 1
        Nlast = list(nx.neighbors(graph, last))
        np.random.shuffle(Nlast) # here you can set any policy you want in which order to check neighbors

        # going in depth
        for neighbor in Nlast:
            if neighbor not in visited: # found new node, then add it to the walk
                random_walk.append(neighbor)
                stack.append(neighbor)
                checked[last].append(neighbor)
                visited.add(neighbor)
                ranks[maxrank] = neighbor
                revranks[neighbor] = maxrank
                break
        else: # we didn't find any new neighbor and rollback
            stack.pop()
            if len(stack) > 0:
                random_walk.append(stack[-1])
                checked[last].append(stack[-1])

        # interconnecting nodes that are already in walk
        for r in range(maxrank-1, lastrank+1, -1):
            node = ranks[r]
            if node not in checked[last] and node in Nlast:
                checked[last].append(node)
                random_walk.extend([node, last])

    covering_anonymous_walk = [revranks[u] for u in random_walk]
    return covering_anonymous_walk, random_walk


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

def canonical_labeling_covers(graph, degrees, samples_per_node = 10):
    if degrees is None:
        degrees = graph.degree()

    degree_x_freq = sorted(Counter(degrees.values()).items(), key=lambda v: (v[1], v[0]))
    target_degree, _ = degree_x_freq[0]

    # print('Need {} nodes to check'.format(target_degree))
    canonical_nodes = filter(lambda v: v[1] == target_degree, degrees.items())

    covers = set()
    for node, freq in canonical_nodes:
        for _ in range(samples_per_node):
            _, anon = covering_walk(graph, node)
            covers.add(tuple(anon))
    return covers


def graph_isomorphism_algorithm_covers(graph1, graph2, samples_per_node = 10):

    degrees1 = dict(graph1.degree())
    degrees2 = dict(graph2.degree())

    if sorted(degrees1.values()) != sorted(degrees2.values()):
        print('Stop on sequence degree')
        return False

    covers1 = canonical_labeling_covers(graph1, degrees1, samples_per_node)
    covers2 = canonical_labeling_covers(graph2, degrees2, samples_per_node)

    overlap = covers1.intersection(covers2)
    if overlap:
        return True
    else:
        return False


def all_aw(steps, keep_last = False):
    '''Get all possible anonymous walks of length up to steps.'''
    paths = []
    last_step_paths = [[0, 1]]
    for i in range(2, steps+1):
        current_step_paths = []
        for j in range(i + 1):
            for walks in last_step_paths:
                if walks[-1] != j and j <= max(walks) + 1:
                    paths.append(walks + [j])
                    current_step_paths.append(walks + [j])
        last_step_paths = current_step_paths
    # filter only on n-steps walks
    if keep_last:
        paths = list(filter(lambda path: len(path) ==  steps + 1, paths))
    return paths


# condition j <= i + 1
def number_of_aw_restricted(length):
    levels = dict()
    levels[0] = {0: 1} # node, frequency
    for step in range(1, length+1):
        all_numbers = levels[step-1]
        new_level = ddict(int)
        for node, freq in all_numbers.items():
            children = list(range(0, node+2))
            for child in children:
                new_level[child] += freq
        levels[step] = dict(new_level)
    return levels


def number_of_aw_in_path(n, source, steps):
    P = nx.path_graph(n)
    aw = AW(P)
    aw.create_random_walk_graph()

    walks = aw.get_dl(source, steps, verbose=False, keep_last=False)
    c = Counter(list(map(len, walks)))
    return list(map(lambda v: v[1], sorted(c.items())))


def check_aw_in_graph(graph, source, aw, verbose=False):
    '''
    Returns random walks that correspond to anonymous walk.
    :param G: graph
    :param source: starting vertex
    :param aw: anonymous walk to check
    :return: random walks and their mapping to anonymous walk.
    '''

    curr_walks = [([source], {source: 0})] # list of tuples: rw, {node->anon}
    for ix in range(1, len(aw)):
        new_walks = []
        prev_target_anonymized = aw[ix - 1] # previous element of aw to check
        target_anonymized = aw[ix] # current element of aw to check
        for walk, mapping in curr_walks:

            # check if we already traversed this edge
            if target_anonymized in mapping.values(): # we already have this index in the mapping
                nodes = list(map(lambda node: mapping[node], walk)) # anon nodes
                pairs = list(zip(nodes, nodes[1:])) # edge pairs
                sorted_pairs = list(map(lambda p: sorted(p), pairs)) # all anon pairs
                if sorted([prev_target_anonymized, target_anonymized]) in sorted_pairs: # presence of this edge
                    reverse_mapping = {v: k for k, v in mapping.items()}
                    new_walks.append((walk + [reverse_mapping[target_anonymized]], mapping)) # add the node to the walk
                    continue # move on to next walk as no other nodes can continue this walk

            new_idx = max(mapping.values()) + 1 # anonymous index for new neighbor
            for neighbor in graph[walk[-1]]: # check each neighbor
                if neighbor in mapping: # already encountered this node
                    if mapping[neighbor] == target_anonymized: # equals to what we seek
                        new_walks.append((walk + [neighbor], mapping))
                else: # new neighbor
                    if new_idx == target_anonymized: # new value
                        new_mapping = mapping.copy() # update mapping
                        new_mapping[neighbor] = new_idx
                        new_walks.append((walk + [neighbor], new_mapping))
        curr_walks = new_walks.copy()
        if verbose:
            print('Iteration {}. Found {} random walks that correspond to {}'.format(ix,
                                                                                    len(curr_walks),
                                                                                    list(map(str, aw[:ix+1]))))
    return curr_walks


def check_corpus_of_aw(graph, source, aws):
    ''' Check every aw in graph.

    :param graph: graph
    :param source: starting vertex
    :param aws: list of aw
    :return:
    '''
    start = time.time()
    d = dict()
    for ix, aw in enumerate(aws):
        walks = check_aw_in_graph(graph, source, aw, verbose=False)
        d[tuple(aw)] = int(len(walks) > 0)
        print(ix)
    print('Time: {:.2f}'.format(time.time() - start))
    return d


def graph_canonical_labeling(graph, degrees = None):
    '''
    Return the minimal frequency degree canonical labeling of a graph.
    It will select the degree that is the least frequent
    (and if tied the least by value among ties).

    :param graph: graph to compute canonical labeling
    :param degrees: if given use precomputed degrees instead
    :return:
    '''

    if degrees is None:
        degrees = graph.degree()

    degree_x_freq = sorted(Counter(degrees.values()).items(), key=lambda v: (v[1], v[0]))
    target_degree, _ = degree_x_freq[0]

    canonical_nodes = filter(lambda v: v[1] == target_degree, degrees.items())

    alphas = sorted([reconstruct_dfs2(graph, node) for node, _ in canonical_nodes])

    return alphas


def graph_isomorphism_algorithm(G1, G2):
    '''
    Tests if two graphs are isomorphic using Anonymous Canonical Labeling.
    The test uses the least frequent degree canonical labelings.
    :param G1:
    :param G2:
    :return: bool, if two graphs are isomorphic
    '''

    degrees1 = G1.degree()
    degrees2 = G2.degree()

    if sorted(degrees1.values()) != sorted(degrees2.values()):
        print('Stop on sequence degree')
        return False

    cl1 = graph_canonical_labeling(G1, degrees1)
    cl2 = graph_canonical_labeling(G2, degrees2)

    return np.all(np.array(cl1) == np.array(cl2))

def label_dataset(graphs):

    labels = dict()
    curr_label = 0
    for i in range(len(graphs)):
        if i not in labels:
            labels[i] = curr_label
        else:
            continue
        for j in range(i + 1, len(graphs)):
            res = nx.is_isomorphic(graphs[i], graphs[j])
            if res:
                labels[j] = curr_label
        curr_label += 1
        print(i)

    return labels

def relabel_graph(G):
    nodes = list(G.nodes())
    new_order = random.sample(G.nodes(), G.order())
    mapping = {nodes[i]: new_order[i] for i in range(len(new_order))}
    return nx.relabel_nodes(G, mapping)

# https://gist.github.com/zachguo/10296432
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = 5
    empty_cell = " " * columnwidth
    # Print header
    print("truth\\pred", end=' ')
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=' ')
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)):
            cell = "%{0}s".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()

def print_cm2(cm):
    tn, fp, fn, tp = cm.ravel()
    print('p\\t\t1\t0')
    print('1\t', tp, fp)
    print('0\t', fn, tn)




if __name__ == '__main__':
    random.seed(0)
    G1 = nx.Graph()
    # G1.add_edges_from([(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7)])
    G1.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5)])
    # pos = nx.spring_layout(G)
    pos = {0: np.array([0.25, 1]),
           1: np.array([0, 0.75]),
           2: np.array([0.5, 0.75]),
           3: np.array([0, 0.25]),
           4: np.array([0.5, 0.25]),
           5: np.array([0.75, 0]),
           #  6: np.array([1, 0.5]),
           #  7: np.array([1, 0.75])
           }

    G2 = nx.Graph()
    # G1.add_edges_from([(0,1), (0,2), (1,2), (1,3), (2,4), (3,5), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7)])
    G2.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])
    print('Original Graph')
    # pos = nx.spring_layout(G)
    pos = {0: np.array([0.5, 1]),
           1: np.array([0, 0.5]),
           2: np.array([1, 0.5]),
           3: np.array([0, 0]),
           4: np.array([1, 0]),
           }

    aw = AW(G2)
    aw.create_random_walk_graph()
    # Dl = aw.get_dl(0, 10)

    # G = nx.read_edgelist('er_graphs_n10/0.edgelist')
    # print('G:', G.edges())
    # print(reconstruct_dfs2(G, '0'))
    # with open('aw/aw6.txt') as f:
    #     aws = list(map(lambda line: list(map(int, line.strip().split(','))), f.readlines()))

    # curr_walks = check_aw_in_graph(G2, 0, [0,1,2,1,0], verbose=True)
    # aws_in_graph = check_corpus_of_aw(G, '0', aws)
    # print(Counter(aws_in_graph.values()))

    # G1 = nx.read_edgelist('../../test/regular6_1.txt')
    # G2 = nx.read_edgelist('../../test/regular6_2.txt')
    # G3 = nx.read_edgelist('../../test/so_1.txt')
    # G4 = nx.read_edgelist('../../test/so_2.txt')

    # G1 = nx.convert_node_labels_to_integers(G1)
    # G2 = nx.convert_node_labels_to_integers(G2)
    # G3 = nx.convert_node_labels_to_integers(G3)
    # G4 = nx.convert_node_labels_to_integers(G4)

    G = nx.read_edgelist('../../reg_graphs_n8_d3/1.edgelist')
    # print(covering_walk(G, '0'))

    G = nx.Graph()
    G.add_path(['0','1','2','3'])
    G.add_edge('0','2')
    G.add_edge('0', '3')

    print(cover2(G1, 0))
    for _ in range(10):
        print(cover2(G1, 0))


    # print(graph_isomorphism_algorithm_covers(G3, G3))
    # print(graph_isomorphism_algorithm_covers(G4, G4))
    # print(graph_isomorphism_algorithm_covers(G3, G4))
    #
    # print(graph_isomorphism_algorithm_covers(G1, G1))
    # print(graph_isomorphism_algorithm_covers(G2, G2))
    # print(graph_isomorphism_algorithm_covers(G1, G2))

    # experiment 0:
    # graph_dir = '../../../reconstruction-data/reg_graphs_n80_d3/'
    # names = os.listdir(graph_dir)
    # regulars = [nx.read_edgelist(graph_dir + fn) for fn in names]
    # preds = []
    # truth = []
    # for it in range(1000):
    #     i, j = random.choices(range(len(regulars)), k=2)
    #     gt = (i==j)*1
    #     pred = graph_isomorphism_algorithm_covers(regulars[i], regulars[j], 10)
    #     print(it, gt, pred)
    #     preds.append(pred)
    #     truth.append(gt)
    #
    # cm = sklearn.metrics.confusion_matrix(truth, preds)
    # print_cm(cm, set(truth))

    # experiment: generate regular graphs of different size and test when algo does not work
    # for N in range(10, 101, 10):
    #     G = nx.random_regular_graph(3, N)
    #     G2 = relabel_graph(G)
    #     print(N, graph_isomorphism_algorithm_covers(G, G2, 10))

    # graph_dir = '../../../reconstruction-data/regulars/'
    # fns = os.listdir(graph_dir)
    # file_mapping = ddict(list)
    # for n_vert, fn in [(int(re.findall('\d+', fn)[0]), fn) for fn in fns]:
    #     file_mapping[n_vert].append(fn)
    #
    # n_pairs = 50
    # ground = []
    # preds = []
    # for n in range(10, 101, 10):
    #     for _ in range(n_pairs):
    #         fn1, fn2 = random.sample(file_mapping[n], 2)
    #         t1, t2 = int(re.findall('t\d+', fn1)[0][1:]), int(re.findall('t\d+', fn2)[0][1:])
    #         ground.append((t1 == t2)*1)
    #         G1, G2 = nx.read_edgelist(graph_dir + fn1), nx.read_edgelist(graph_dir + fn2)
    #         preds.append(graph_isomorphism_algorithm_covers(G1, G2)*1)
    #         # print(n, _, ground[-1], preds[-1])
    #     # print(n, ground[-n_pairs:], preds[-n_pairs:])
    #     cm = sklearn.metrics.confusion_matrix(ground, preds)
    #     print('N: ', n)
    #     print_cm2(cm)


    # Experiment 1: small regular graphs
    # names = os.listdir('../../reg_graphs_n8_d3/')
    # regulars = []
    # for i, fn in enumerate(names):
    #     if fn.endswith('edgelist'):
    #         regulars.append(nx.read_edgelist('../../reg_graphs_n8_d3/' + fn))
    #     # if i > 10:
    #     #     break
    # print('Read all graphs')


    # truth = []
    # with open('../../reg_graphs_n8_d3/labels.txt') as f:
    #     for line in f:
    #         if line:
    #             truth.append(line.strip())

    # ix = 0
    # G = regulars[ix]
    # true_label = truth[ix]
    # tp = fp = tn = fn = 0
    # preds = []
    # total = 0
    # for i, test_graph in enumerate(regulars):
    #     start = time.time()
    #     is_iso = graph_isomorphism_algorithm_covers(G, test_graph, samples_per_node=1)
    #     end = time.time()
    #     total += end-start
    #     if is_iso:
    #         if truth[i] == true_label:
    #             tp += 1
    #         else:
    #             fp += 1
    #     else:
    #         if truth[i] == true_label:
    #             fn += 1
    #         else:
    #             tn += 1
    #     preds.append((is_iso)*1)
    #     # print(i, is_iso, truth[i])
    #
    # converted_truth = [(c == true_label)*1 for c in truth]
    # cm = sklearn.metrics.confusion_matrix(converted_truth, preds)
    # print("time:", end-start)
    # print('confusion matrix')
    # print(tp, fp)
    # print(fn, tn)
    # print_cm(cm, set(converted_truth))
    #RESULTS#####################
    # output (10 samples per node)
    # time: False 112.35075759887695
    # confusion matrix
    #               0     1
    #         0   880     0
    #         1     0   120
    # output (1 sample per node)
    # time: 0.001979351043701172
    # confusion matrix
    #               0     1
    #         0   880     0
    #         1     2   118
    # degree-policy experiment (n=10)
    # time: 0.001985311508178711
    # confusion matrix
    # truth\pred     0     1
    #         0   880     0
    #         1     6   114


    # experiment 2: hard graphs of medium size (~ 100 nodes)
    # save_dir = '../../cfi-rigid-r2-iso-edges/'
    # fns = sorted(filter(lambda fn: fn.startswith('cfi'), os.listdir(save_dir)))

    # G1, G2 = nx.read_edgelist(save_dir + fns[2]), nx.read_edgelist(save_dir + fns[2 + 1])
    # start = time.time()
    # is_iso = graph_isomorphism_algorithm_covers(G1, G2, samples_per_node=10)
    # end = time.time()
    # print(is_iso, end-start)
    # RESULTS#######################
    # output: False 112.35075759887695

    # times = []
    # preds = []
    # for i in range(0, len(fns), 2):
    #     G1, G2 = nx.read_edgelist(save_dir + fns[i]), nx.read_edgelist(save_dir + fns[i + 1])
    #     start = time.time()
    #     # is_iso = nx.is_isomorphic(G1, G2)
    #     is_iso = graph_isomorphism_algorithm_covers(G1, G2, samples_per_node=1)
    #     end = time.time()
    #     preds.append(is_iso)
    #     times.append(end-start)
    #     print(i, is_iso, end-start, G1.order(), G1.size())
    # print(sum(preds))
    # print(sum(times)/len(times))
    ####RESULTS#########################
    # covers algo: up to 72 pair result false for all ~10sec-10min (72<= n<=720)
    # nx.is_isomorphic: true for 2 pairs, 8th pairs takes more than 30 minutes


    # experiment 3: easy graphs of medium size (70 nodes)
    # truth = []
    # preds = []
    # save_dir = '../../er_graphs_n70/'
    # fns = sorted(filter(lambda fn: fn.endswith('edgelist'), os.listdir(save_dir)))
    # ix = 0
    # G = nx.read_edgelist(save_dir + fns[ix])
    # tp = fp = tn = fn = 0
    # for i in range(len(fns)):
    #     test_graph = nx.read_edgelist(save_dir + fns[i])
    #     pred = graph_isomorphism_algorithm_covers(G, test_graph, 10)
    #     true = nx.is_isomorphic(G, test_graph) * 1
    #     if pred == 1:
    #         if true == 1:
    #             tp += 1
    #         else:
    #             fp += 1
    #     else:
    #         if true == 1:
    #             fn += 1
    #         else:
    #             tn += 1
    #     preds.append(pred*1)
    #     truth.append(true*1)
    #
    # print(tp, fp)
    # print(fn, tn)

    # cm = sklearn.metrics.confusion_matrix(truth, preds)
    # print('confusion matrix')
    # print_cm(cm, [0,1])
    #RESULTS####################
    # mistake only on the same graph

    # for n in range(10, 101, 5):
    #     G = nx.watts_strogatz_graph(n, 3, 0.01)
    #     print(n, graph_isomorphism_algorithm_covers(G, G, 10))
    #RESULTS##############
    # output: true for all

    # for n in range(10, 101, 5):
    #     G = connect_graph(nx.erdos_renyi_graph(n, 0.3))
    #     print(n, graph_isomorphism_algorithm_covers(G, G, 1))
    #RESULTS###############
    # output: true only for n=10 (10 samples) and n-10,15 (for 100 samples)
    # output: true for all if we select degree policy



    # total = 0
    # for i in range(0, len(fns), 2):
    #     try:
    #         i = 2
    #         G1, G2 = nx.read_edgelist(save_dir + fns[i]), nx.read_edgelist(save_dir + fns[i+1])
    #     except:
    #         print('Found incompatible files', fns[i], fns[i+1])
    #
    #     is_iso = graph_isomorphism_algorithm_covers(G1, G2, samples_per_node=100)
    #     total += is_iso*1
    #     print(total, len(fns))
    #     break

    # print(nx.is_isomorphic(G1, G2))

    # alphas1 = graph_canonical_labeling(G1)
    # print(alphas1)
    #
    # alphas2 = graph_canonical_labeling(G2)
    # print(alphas2)

    # p1 = reconstruct_dfs2(G1, 3)
    # p2 = reconstruct_dfs2(G2, 3)

    # print(p1)
    # print(p2)
    # alpha = [0, 1, 2, 3, 4, 5, 4, 3, 2, 5, 2, 1, 0]
    # curr_walks = check_aw_in_graph(G1, 0, alpha, verbose=True)
    # print(curr_walks)


    console = []