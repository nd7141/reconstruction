from collections import deque
import numpy as np
import random
import torch

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
    
