import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from AnonymousWalkKernel import AnonymousWalks as AW
from collections import defaultdict as ddict, Counter

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

if __name__ == '__main__':
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
    Dl = aw.get_dl(0, 10)

    # balls, supports = reconstruct(Dl, 3)
    balls = reconstruct_dfs(Dl)
    # print(supports)

    paths10 = all_aw(10)
    # cardinalities = number_of_aw_restricted(10)

    # number of anonymous walks in a path
    print(number_of_aw_in_path(10, 0, 15))

    console = []