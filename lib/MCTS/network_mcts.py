import random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import nn_utils

class Agent(nn.Module):
    def __init__(self, hid_size, gcn_size=None, vertex_emb_size=None,
                 activation=nn.ELU(), layernorm=False, num_vertices=None):
        super().__init__()
        gcn_size = gcn_size or hid_size
        self.vertex_emb_size = vertex_emb_size = vertex_emb_size or hid_size

        self.gcn = nn_utils.GraphConvolutionBlock(
            vertex_emb_size, gcn_size, out_size=hid_size, num_convolutions=2,
            activation=activation, normalize_hid=layernorm
        )

        self.value = nn.Sequential(
            nn.Linear(2*hid_size, hid_size),
            activation,
            nn.Linear(hid_size, 1)
        )

        self.policy = nn.Sequential(
            nn.Linear(2*hid_size, hid_size),
            activation,
            nn.Linear(hid_size, num_vertices),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        value = self.value(x)
        log_probs = self.policy(x)
        return log_probs, value

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

    def predict(self, embs):
        with torch.no_grad():
            pi, v = self.forward(embs)
            #print(torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]) list with v value
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]