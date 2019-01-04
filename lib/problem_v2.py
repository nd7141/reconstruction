import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

from .reconstruction.generators import connect_graph, generate_anonymous_walks
from .heapsearch import solve_heapsearch


class GAWProblem:
    def __init__(self, graph_edges, anonymous_walk, initial_vertex=0):
        """
        A problem of traversing a graph following a given anonymous walk
        :param graph_edges: dict { vertex_id -> next vertices avilable from vertex_id }
        :param anonymous_walk: a list of anonymous vertex ids of the desired walk.
        :param initial_vertex: index of the initial vertex
        """
        self.edges = graph_edges
        self.walk = anonymous_walk
        self.initial_vertex = initial_vertex
        self.path = [self.initial_vertex]

    def get_valid_actions(self):
        """ Returns a set of available next vertices that do not violate the walk """
        if len(self.path) == len(self.walk): return set()  # problem is solved
        anonymous_next_vertex = self.walk[len(self.path)]
        if anonymous_next_vertex not in self.walk[:len(self.path)]:
            # if next vertex hasn't appeared in the walk before,
            # allow any vertex that hasn't been visited yet
            return set(self.edges[self.path[-1]]).difference(set(self.path))
        else:
            # if next vertex was already mentioned in AW before, accept that vertex only
            valid_vertex = self.path[self.walk.index(anonymous_next_vertex)]
            return {valid_vertex} if valid_vertex in self.edges[self.path[-1]] else set()

    def step(self, chosen_next_vertex):
        """ Take an action by adding a vertex to current anonymous walk. Gym-compliant interface. """
        assert chosen_next_vertex in self.get_valid_actions()
        self.path.append(chosen_next_vertex)
        is_done = len(self.get_valid_actions()) == 0
        reward = 1.0 / len(self.walk)
        if len(self.path) == len(self.walk):
            reward += 100
        return chosen_next_vertex, reward, is_done, {}

    def render(self):
        """ displays a graph with current walk on it """
        plt.title('walk: ' + repr(self.walk))
        nx.draw(nx.Graph(self.edges),
                labels=dict(zip(self.path, self.walk)),
                )
        plt.show()

    def get_state(self):
        return list(self.path)

    def load_state(self, state):
        self.path = list(state)

    def reset(self):
        self.path = [self.initial_vertex]


def generate_erdos_renyi_problems(num_vertices=10, edge_prob=0.2,
                                  walk_length_range=None, verify_heapsearch_budget=None):
    walks = generate_anonymous_walks(num_vertices)
    if walk_length_range is not None:
        walks = [walk for walk in walks if walk_length_range[0] <= len(walk) <= walk_length_range[1]]

    while True:
        walk = random.choice(walks)
        graph = connect_graph(nx.erdos_renyi_graph(num_vertices, edge_prob))
        graph_edges = defaultdict(set)
        for v1, v2 in graph.edges:
            graph_edges[v1].add(v2)
            graph_edges[v2].add(v1)

        initial_vertex = random.choice(sorted(graph_edges.keys()))
        problem = GAWProblem(graph_edges, walk, initial_vertex)

        if verify_heapsearch_budget is not None:
            solution, _ = solve_heapsearch(problem, max_steps=verify_heapsearch_budget)
            if len(solution) < len(walk):
                continue

        problem.reset()
        yield problem

def generate_regular_problems(num_vertices=10, degree=3,
                                  walk_length_range=None, verify_heapsearch_budget=None):
    walks = generate_anonymous_walks(num_vertices)
    if walk_length_range is not None:
        walks = [walk for walk in walks if walk_length_range[0] <= len(walk) <= walk_length_range[1]]

    while True:
        walk = random.choice(walks)
        graph = connect_graph(nx.random_regular_graph(degree, num_vertices))
        graph_edges = defaultdict(set)
        for v1, v2 in graph.edges:
            graph_edges[v1].add(v2)
            graph_edges[v2].add(v1)

        initial_vertex = random.choice(sorted(graph_edges.keys()))
        problem = GAWProblem(graph_edges, walk, initial_vertex)

        if verify_heapsearch_budget is not None:
            solution, _ = solve_heapsearch(problem, max_steps=verify_heapsearch_budget)
            if len(solution) < len(walk):
                continue

        problem.reset()
        yield problem
