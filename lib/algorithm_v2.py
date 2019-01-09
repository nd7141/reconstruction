import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class QLearningTrainer:
    def __init__(self, agent, Optimizer=lambda params, **kw: torch.optim.Adam(params, lr=1e-4, **kw)):
        """
        :param agent: lib.agent.SimpleAgent
        """
        self.agent = agent
        self.opt = Optimizer(agent.parameters())
        self.target_q_network = copy.deepcopy(agent.q_network)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.agent.q_network.state_dict())

    def train_on_batch(self, batch_size = 100, step=True, **kwargs):
        #graph_emb = self.agent.embed_graph(problem.edges, **kwargs)  # {vertex_id -> vector[num_units]}
        #walk_emb = self.agent.embed_walk(problem.walk, **kwargs)     # matrix[walk_length, units]
        
        #sample from replay buffer

        batch = self.agent.buffer.sample(batch_size)

        # compute next state values, V(s') = argmax_a' Q(s', a')
        batch_indices, next_state_inputs = [], []
        for i, (problem, path, next_vertex, reward, is_done) in enumerate(batch):
            graph_emb = self.agent.embed_graph(problem.edges, **kwargs)
            walk_emb = self.agent.embed_walk(problem.walk, **kwargs) 
            problem.load_state(path)
            problem.step(next_vertex)
            next_actions = problem.get_valid_actions()
            batch_indices.extend([i] * len(next_actions))
            next_state_inputs.extend([
                torch.cat([walk_emb[len(problem.path)], graph_emb[problem.path[-1]], graph_emb[valid_next_vertex]])
                for valid_next_vertex in next_actions
            ])

        with torch.no_grad():
            if len(next_state_inputs):
                next_q_values = self.target_q_network(torch.stack(next_state_inputs))[:, 0]
            else:
                next_q_values = torch.zeros(0)

        next_qvalues_by_batch_index = defaultdict(list)
        for i, q in zip(batch_indices, next_q_values.data.cpu().numpy()):
            next_qvalues_by_batch_index[i].append(q)

        # {batch index -> V(s')}
        next_state_values_by_batch_index = {i: max(next_qvalues_by_batch_index[i])
                                            for i in next_qvalues_by_batch_index}

        # build training batch
        q_network_inputs, q_network_targets = [], []

        for i, (problem, path, next_vertex, reward, is_done) in enumerate(batch):
            graph_emb = self.agent.embed_graph(problem.edges, **kwargs)
            walk_emb = self.agent.embed_walk(problem.walk, **kwargs)
            problem.load_state(path)
            q_network_inputs.append(torch.cat([
                walk_emb[len(problem.path)], graph_emb[problem.path[-1]], graph_emb[next_vertex]
            ]))
            q_network_targets.append(reward if is_done else (reward + next_state_values_by_batch_index[i]))

        # compute loss and backprop
        predicted_qvalues = self.agent.q_network(torch.stack(q_network_inputs))[:, 0]
        target_qvalues = torch.tensor(q_network_targets, device=predicted_qvalues.device)
        mse = torch.mean((predicted_qvalues - target_qvalues) ** 2)
        mse.backward()
        if step:
            self.opt.step()
            self.opt.zero_grad()

        return mse.item()
