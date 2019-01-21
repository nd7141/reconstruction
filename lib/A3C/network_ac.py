import random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import nn_utils_ac
import utils_ac

class ActorCriticPDist(nn.Module):
    def __init__(self, hid_size, gcn_size=None, lstm_size=None, walk_emb_size=None, vertex_emb_size=None,
                 activation=nn.ELU(), layernorm=False, num_vertices=None):
        super().__init__()
        gcn_size = gcn_size or hid_size
        self.vertex_emb_size = vertex_emb_size = vertex_emb_size or hid_size

        self.gcn = nn_utils_ac.GraphConvolutionBlock(
            vertex_emb_size, gcn_size, out_size=hid_size, num_convolutions=2,
            activation=activation, normalize_hid=layernorm
        )

        self.critic = nn.Sequential(
            nn.Linear(2*hid_size, hid_size),
            activation,
            nn.Linear(hid_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(2*hid_size, hid_size),
            activation,
            nn.Linear(hid_size, num_vertices),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value

    def embed_graph(self, graph_edges, *, device=None, **kwargs):
        """
        :param edges: {vertex_id -> available edges from that vertex_id}
        :return: {vertex_id -> vector representation of that vertex id}
        """
        assert len(graph_edges) == max(graph_edges) + 1, "graph vertices must be labeled 0...N without gaps"
        vertices = sorted(graph_edges.keys())
        vertices = torch.tensor(vertices, device=device, dtype=torch.int64)
        vertex_emb = nn_utils_ac.encode_indices(vertices, self.vertex_emb_size)
        # ^-- [num_vertices, vertex_emb_size]
        adj = nn_utils_ac.make_adjacency_matrix(graph_edges, device=device)  # sparse [num_vertices x num_vertices]
        vertex_emb = self.gcn(vertex_emb, adj)  # [num_vertices x hid_size]
        return dict(zip(map(int, vertices), vertex_emb))


class ActorCriticAct(nn.Module):
    def __init__(self, hid_size, gcn_size=None, lstm_size=None, walk_emb_size=None, vertex_emb_size=None,
                 activation=nn.ELU(), layernorm=False):
        super().__init__()
        gcn_size = gcn_size or hid_size
        self.vertex_emb_size = vertex_emb_size = vertex_emb_size or hid_size

        self.gcn = nn_utils_ac.GraphConvolutionBlock(
            vertex_emb_size, gcn_size, out_size=hid_size, num_convolutions=2,
            activation=activation, normalize_hid=layernorm
        )

        self.critic = nn.Sequential(
            nn.Linear(2*hid_size, hid_size),
            activation,
            nn.Linear(hid_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(3*hid_size, hid_size),
            activation,
            nn.Linear(hid_size, 1),
            nn.Softmax(dim=0)
        )

    def embed_graph(self, graph_edges, *, device=None, **kwargs):
        """
        :param edges: {vertex_id -> available edges from that vertex_id}
        :return: {vertex_id -> vector representation of that vertex id}
        """
        assert len(graph_edges) == max(graph_edges) + 1, "graph vertices must be labeled 0...N without gaps"
        vertices = sorted(graph_edges.keys())
        vertices = torch.tensor(vertices, device=device, dtype=torch.int64)
        vertex_emb = nn_utils_ac.encode_indices(vertices, self.vertex_emb_size)
        # ^-- [num_vertices, vertex_emb_size]
        adj = nn_utils_ac.make_adjacency_matrix(graph_edges, device=device)  # sparse [num_vertices x num_vertices]
        vertex_emb = self.gcn(vertex_emb, adj)  # [num_vertices x hid_size]
        return dict(zip(map(int, vertices), vertex_emb))
        
    def get_dist(self, paths, graph_emb, edges):
        paths_embs = []
        for path in paths:
            path_emb = []
            for item in path:
                path_emb.append(graph_emb[item])
            paths_embs.append(hidden[0].view(-1))
        values = self.critic(paths_embs)
        states = []
        for i, path in enumerate(paths):
            next_embs = []
            for next_vertex in edges[path[-1]]:
                next_vertex_emb = graph_emb[next_vertex]
                next_embs.append(torch.cat([paths_embs[i], next_vertex_emb]))
            states.append(torch.stack(next_embs))
        predicts = []
        for i in states:
            predicts.append(self.actor(i).view(-1))
        return predicts, values

class ActorCriticActLSTM(nn.Module):
    def __init__(self, hid_size, gcn_size=None, lstm_size=None, walk_emb_size=None, vertex_emb_size=None,
                 activation=nn.ELU(), layernorm=False):
        super().__init__()
        gcn_size = gcn_size or hid_size
        self.vertex_emb_size = vertex_emb_size = vertex_emb_size or hid_size

        self.gcn = nn_utils_ac.GraphConvolutionBlock(
            vertex_emb_size, gcn_size, out_size=hid_size, num_convolutions=2,
            activation=activation, normalize_hid=layernorm
        )

        self.lstm = nn.LSTM(hid_size, hid_size, batch_first=True)

        self.critic = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            activation,
            nn.Linear(hid_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(2*hid_size, hid_size),
            activation,
            nn.Linear(hid_size, 1),
            nn.Softmax(dim=0)
        )

    def embed_graph(self, graph_edges, *, device=None, **kwargs):
        """
        :param edges: {vertex_id -> available edges from that vertex_id}
        :return: {vertex_id -> vector representation of that vertex id}
        """
        assert len(graph_edges) == max(graph_edges) + 1, "graph vertices must be labeled 0...N without gaps"
        vertices = sorted(graph_edges.keys())
        vertices = torch.tensor(vertices, device=device, dtype=torch.int64)
        vertex_emb = nn_utils_ac.encode_indices(vertices, self.vertex_emb_size)
        # ^-- [num_vertices, vertex_emb_size]
        adj = nn_utils_ac.make_adjacency_matrix(graph_edges, device=device)  # sparse [num_vertices x num_vertices]
        vertex_emb = self.gcn(vertex_emb, adj)  # [num_vertices x hid_size]
        return dict(zip(map(int, vertices), vertex_emb))
        
    def get_dist(self, paths, graph_emb, edges):
        # get paths embedings
        paths_embs = []
        for path in paths:
            path_emb = []
            for item in path:
                path_emb.append(graph_emb[item])
            _, hidden = self.lstm(torch.stack(path_emb).unsqueeze(0))
            paths_embs.append(hidden[0].view(-1))
        states = []
        for i, path in enumerate(paths):
            next_embs = []
            for next_vertex in edges[path[-1]]:
                next_vertex_emb = graph_emb[next_vertex]
                next_embs.append(torch.cat([paths_embs[i], next_vertex_emb]))
            states.append(torch.stack(next_embs))
        predicts = []
        for i in states:
            predicts.append(self.actor(i).view(-1))
        values =[]
        for j in range(len(paths)):
            values.append(self.critic(paths_embs[j]))
        return predicts, torch.stack(values).view(-1)