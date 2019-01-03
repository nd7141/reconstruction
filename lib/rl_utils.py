from collections import deque
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, problem, state, chosen_vertex, reward, is_done):
        self.buffer.append((problem, state, chosen_vertex, reward, is_done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

