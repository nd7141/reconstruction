import random
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

from generators import connect_graph, generate_anonymous_walks


class GraphProblem:
    def __init__(self, graph_edges, initial_vertex=0, num_edges=None, nx_graph=None):
        """
        A problem of traversing a graph following a given anonymous walk
        :param graph_edges: dict { vertex_id -> next vertices avilable from vertex_id }
        :param initial_vertex: index of the initial vertex
        """
        self.edges = graph_edges
        self.initial_vertex = initial_vertex
        self.path = [self.initial_vertex]
        self.num_edges = num_edges
        self.nx_graph = nx_graph
    
    def get_actions(self):
        return sorted(list(self.edges.keys()))
    
    def get_action_size(self):
        return len(self.edges.keys())

    def get_valid_actions(self, vertex):
        actions = self.get_actions()
        valid = list(self.edges[vertex])
        v = np.zeros(len(actions))
        for i in valid:
            v[i] = 1
        return v

    def get_next_state(self, path, vertex):
        return path + [vertex]

    def render(self):
        """ displays a graph with current walk on it """
        plt.title('Graph')
        nx.draw(nx.Graph(self.edges))
        plt.show()

    def get_state(self):
        return list(self.path)

    def load_state(self, state):
        self.path = list(state)

    def reset(self):
        self.path = [self.initial_vertex]


def generate_erdos_renyi_problems(num_vertices=10, edge_prob=0.2):

    while True:
        graph = connect_graph(nx.erdos_renyi_graph(num_vertices, edge_prob))
        graph_edges = defaultdict(set)
        num_edges = 0
        nx_graph = graph
        for v1, v2 in graph.edges:
            graph_edges[v1].add(v2)
            graph_edges[v2].add(v1)
            num_edges += 1

        initial_vertex = random.choice(sorted(graph_edges.keys()))
        problem = GraphProblem(graph_edges, initial_vertex, num_edges, nx_graph)

        problem.reset()
        yield problem

def generate_regular_problems(num_vertices=10, degree=3):

    while True:
        graph = connect_graph(nx.random_regular_graph(degree, num_vertices))
        graph_edges = defaultdict(set)
        num_edges = 0
        nx_graph = graph
        for v1, v2 in graph.edges:
            graph_edges[v1].add(v2)
            graph_edges[v2].add(v1)
            num_edges += 1

        initial_vertex = random.choice(sorted(graph_edges.keys()))
        problem = GraphProblem(graph_edges, initial_vertex, num_edges, nx_graph)

        problem.reset()
        yield problem
