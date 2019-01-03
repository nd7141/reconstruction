import random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from lib import nn_utils, rl_utils
from heapq import heappush, heappop, heapify, nsmallest


class SimpleAgent(nn.Module):
    def __init__(self, hid_size, gcn_size=None, lstm_size=None, walk_emb_size=None, vertex_emb_size=None,
                 activation=nn.ELU(), layernorm=False, capacity=1000):
        super().__init__()
        gcn_size = gcn_size or hid_size
        lstm_size = lstm_size or hid_size
        self.vertex_emb_size = vertex_emb_size = vertex_emb_size or hid_size
        self.walk_emb_size = walk_emb_size = walk_emb_size or vertex_emb_size

        self.gcn = nn_utils.GraphConvolutionBlock(
            vertex_emb_size, gcn_size, out_size=hid_size, num_convolutions=2,
            activation=activation, normalize_hid=layernorm
        )

        self.walk_lstm = nn.LSTM(walk_emb_size, lstm_size, batch_first=True, bidirectional=True)
        self.walk_out = nn.Linear(2 * lstm_size, hid_size)

        # v-- (current walk vector, current graph vertex, next_action_vertex) -> Q(s, take_next_action)
        self.q_network = nn.Sequential(
            nn.Linear(3 * hid_size, hid_size), activation, nn.Linear(hid_size, 1)
        )
        
        self.buffer = rl_utils.ReplayBuffer(capacity=capacity)

    def get_q_values(self, problem, graph_emb, walk_emb, possible_actions=None):
        valid_next_vertices = possible_actions or sorted(problem.get_valid_actions())
        q_network_inputs = torch.stack([
            torch.cat(
                [walk_emb[len(problem.path)], graph_emb[problem.path[-1]], graph_emb[next_vertex]]
            ) for next_vertex in valid_next_vertices
        ])
        q_values = self.q_network(q_network_inputs)[:, 0]
        return dict(zip(valid_next_vertices, q_values))

    def embed_graph(self, graph_edges, *, device=None, **kwargs):
        """
        :param edges: {vertex_id -> available edges from that vertex_id}
        :return: {vertex_id -> vector representation of that vertex id}
        """
        assert len(graph_edges) == max(graph_edges) + 1, "graph vertices must be labeled 0...N without gaps"
        vertices = sorted(graph_edges.keys())
        vertices = torch.tensor(vertices, device=device, dtype=torch.int64)
        vertex_emb = nn_utils.encode_indices(vertices, self.vertex_emb_size)
        # ^-- [num_vertices, vertex_emb_size]
        adj = nn_utils.make_adjacency_matrix(graph_edges, device=device)  # sparse [num_vertices x num_vertices]
        vertex_emb = self.gcn(vertex_emb, adj)  # [num_vertices x hid_size]
        return dict(zip(map(int, vertices), vertex_emb))

    def embed_walks(self, walks, *, device=None, **kwargs):
        """
        :param walk: a tensor of anonymous vertex indices, shape: [batch, length]
        :return: matrix [batch, length, hid_size]
        """
        walks = walks.to(device=device)
        walk_embs = nn_utils.encode_indices(walks, self.walk_emb_size)  # [batch, len, walk_emb_size]
        walk_embs, _ = self.walk_lstm(walk_embs)  # [batch, len, 2 * lstm_size]
        walk_embs = self.walk_out(walk_embs)   # [batch, len, hid_size]
        return walk_embs

    def embed_walk(self, walk, **kwargs):
        return self.embed_walks(torch.tensor([walk]), **kwargs)[0]

    def solve(self, problem, epsilon=0.0, max_backtracks=0, reset=True, **kwargs):
        """
        Attempts to find a given anonymous walk by greedily selecting action with highest q-value.
        :type problem: GAWProblem
        :param epsilon: probability of picking random action instead of one with highest q-value
        :param reset: if True, resets problem before playing
        :return: (best solution, transitions)
        """
        assert max_backtracks == 0, "backtracking not yet implemented"
        if reset:
            problem.reset()

        graph_emb = self.embed_graph(problem.edges, **kwargs)  # {vertex_id -> vector[num_units]}
        walk_emb = self.embed_walk(problem.walk, **kwargs)     # matrix[walk_length, units]
        #transitions = []  # [s, a, r, is_done] tuples

        while True:
            if random.random() >= epsilon:
                next_vertex_qvalues = self.get_q_values(problem, graph_emb, walk_emb)
                chosen_vertex = max(next_vertex_qvalues.keys(), key=next_vertex_qvalues.get)
            else:
                possible_actions = problem.get_valid_actions()
                chosen_vertex = random.choice(list(possible_actions))

            state = problem.get_state()
            _, reward, is_done, _ = problem.step(chosen_vertex)
            self.buffer.push(problem, state, chosen_vertex, reward, is_done)

            if is_done:
                #TODO backtrack here
                break

        return problem.get_state()#, transitions
