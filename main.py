import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from AnonymousWalkKernel import AnonymousWalks as AW


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

    aw = AW(G1)
    aw.create_random_walk_graph()
    Dl = aw.get_dl(0, 14)

    balls, supports = reconstruct(Dl, 3)
    print(supports)

    for ball in balls:
        print(ball.edges(), ball.nodes())