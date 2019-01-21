from collections import deque
import numpy as np
import random
import torch
import networkx as nx
from problem_mcts import GraphProblem, convert_graph
import glob
from collections import defaultdict

class PathsBuffer(object):
    def __init__(self, capacity=1000, threshold = 0.75):
        self.capacity = capacity
        self.buffer = []
        self.threshold = threshold
    
    def push(self, path):
        self.buffer.append(path)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def flush(self):
        self.buffer = []

    def rank_path(self, path):
        buffer_copy = self.buffer[:]
        buffer_copy.append(path)
        if len(buffer_copy) > self.capacity:
            buffer_copy.pop(0)
        buffer_copy.sort(key = lambda x: convert_to_walk(x))
        path_num = round(self.threshold*len(buffer_copy))
        if convert_to_walk(path) > convert_to_walk(buffer_copy[path_num]):
            return 1.0
        elif convert_to_walk(path) == convert_to_walk(buffer_copy[path_num]):
            return float(random.sample([1, -1], 1)[0])
        return -1.0
    
    def is_paths_buffer_valid(self):
        self.buffer.sort(key = lambda x: convert_to_walk(x))
        for i in range(len(self.buffer)-1):
            if convert_to_walk(self.buffer[i]) >= convert_to_walk(self.buffer[i+1]):
                return False
        return True
    
    def __len__(self):
        return len(self.buffer)

def is_valid_path_new(path, edges):
    for i in range(len(path)-1):
        if path[i+1] not in edges[path[i]]:
            return False
    return True

def is_valid_path(path, problem):
    edges = problem.edges
    for i in range(len(path)-1):
        if path[i+1] not in edges[path[i]]:
            return False
    return True

def convert_to_walk(path):
    mapping = dict()
    walk = []
    index = 0
    for vertex in path:
        if vertex not in mapping:
            mapping[vertex] = index
            walk.append(index)
            index += 1
        else:
            walk.append(mapping[vertex])
    return walk

def get_states_emb(paths, graph_emb):
    paths_emb = []
    for path in paths:
        path_emb = []
        for node in path:
            path_emb.append(graph_emb[node].data.numpy())
        avg = torch.mean(torch.tensor(path_emb), 0)
        paths_emb.append(torch.cat((avg, graph_emb[path[-1]])).data.numpy())
    return torch.tensor(paths_emb)

class ReplayBuffer(object):
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = 1000
    
    def push(self, records):
        self.buffer.extend(records)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def flush(self):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)

def graph_isomorphism_algorithm_covers(graph1, graph2, agent, policy, samples_per_node = 10):

    degrees1 = dict(graph1.degree())
    degrees2 = dict(graph2.degree())

    if sorted(degrees1.values()) != sorted(degrees2.values()):
        print('Stop on sequence degree')
        return False

    covers1 = canonical_labeling_covers(agent, graph1, policy, samples_per_node)
    covers2 = canonical_labeling_covers(agent, graph2, policy, samples_per_node)

    overlap = covers1.intersection(covers2)
    if overlap:
        return True
    else:
        return False

def canonical_labeling_covers(agent, graph, policy, samples_per_node = 10):
    covers = set()
    graph = convert_graph(graph)
    for node in graph.edges.keys():
        for _ in range(samples_per_node):
          random_walk = policy(agent, graph, node)
          covers.add(tuple(convert_to_walk(random_walk)))
    return covers

def relabel_graph(G):
    nodes = list(G.nodes())
    new_order = random.sample(G.nodes(), G.order())
    mapping = {nodes[i]: new_order[i] for i in range(len(new_order))}
    return nx.relabel_nodes(G, mapping)

def check_that_cover(random_walk, problem):
    checked = {tuple(sorted(e)): 0 for e in problem.nx_graph.edges()}
    for i in range(len(random_walk) - 1):
        r_edge = tuple(sorted([random_walk[i], random_walk[i+1]]))
        if r_edge not in checked:
            print('Found edge that does not exist', r_edge)
            return False
        else:
            checked[r_edge] = 1

    if sum(checked.values()) == problem.nx_graph.size():
        return True
    else:
        print('Not all edges are found', checked)
        return False

def generate_graphs(path, n_graphs):
    graphs = []
    samples = random.choices(glob.glob(path+"/*"), k=n_graphs)
    for sample in samples:
        graph_path = random.choices(glob.glob(sample+"/9?.edgelist"),k=1)
        with open(graph_path[0], "rb") as f:
            graph = nx.read_edgelist(f)
            graph_edges = defaultdict(set)
            num_edges = 0
            nx_graph = graph
            for v1, v2 in graph.edges:
                graph_edges[int(v1)].add(int(v2))
                graph_edges[int(v2)].add(int(v1))
                num_edges += 1
        graphs.append(GraphProblem(graph_edges, 0, num_edges, nx_graph))
    return graphs   
