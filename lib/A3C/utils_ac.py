from collections import deque
import numpy as np
import random
import torch
import networkx as nx
from problem_ac import convert_graph

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
        #rank buffer by alphabetical walk order and select with reward 1
        buffer_copy = self.buffer[:]
        buffer_copy.append(path)
        if len(buffer_copy) > self.capacity:
            buffer_copy.pop(0)
        buffer_copy.sort(key = lambda x: convert_to_walk(x))
        path_num = round(self.threshold*len(buffer_copy))
        if convert_to_walk(path) > convert_to_walk(buffer_copy[path_num]):
            return 1.0
        elif convert_to_walk(path) == convert_to_walk(buffer_copy[path_num]):
            return random.sample([1, -1], 1)[0]
        return -1.0

    def is_paths_buffer_valid(self, edges):
        return(all(is_valid_path_new(path, edges) for path in self.buffer))

    def is_paths_buffer_sorted(self):
        self.buffer.sort(key = lambda x: convert_to_walk(x))
        for i in range(len(self.buffer)-1):
            if not convert_to_walk(self.buffer[i]) >= convert_to_walk(self.buffer[i+1]):
                return False
        return True
    
    def __len__(self):
        return len(self.buffer)

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
    
    def push(self, records):
        self.buffer.extend(records)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def is_valid_path(path, problem):
    edges = problem.edges
    for i in range(len(path)-1):
        if path[i+1] not in edges[path[i]]:
            return False
    return True

def is_valid_path_new(path, edges):
    for i in range(len(path)-1):
        if path[i+1] not in edges[path[i]]:
            return False
    return True

def replace(P, source, target):
    '''Replace last occurrence of source with source-target-source.'''
    assert source in P
    ix = len(P) - P[::-1].index(source)
    return P[:ix] + [target, P[ix - 1]] + P[ix:]

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

def random_walk(problem, path_length, vertex):
    path = [vertex]
    for _ in range(path_length-1):
        next_vertex = random.sample(problem.edges[path[-1]],1)[0]
        path = problem.get_next_state(path, next_vertex)
    return path

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
