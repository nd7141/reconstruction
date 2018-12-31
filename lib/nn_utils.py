import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 .
    Shamelessly stolen from https://github.com/tkipf/pygcn/
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionBlock(nn.Module):
    def __init__(self, inp_size, hid_size, out_size=None, num_convolutions=1, activation=nn.ELU(),
                 residual=False, normalize_hid=False, normalize_out=False):
        """ Graph convolution layer with some options """
        nn.Module.__init__(self)
        out_size = out_size or inp_size
        assert (out_size == inp_size) or not residual
        self.convs = nn.ModuleList([GraphConvolution(inp_size if i == 0 else hid_size, hid_size)
                                    for i in range(num_convolutions)])
        if normalize_hid:
            self.hid_norms = [nn.LayerNorm(hid_size) for _ in range(num_convolutions)]
        self.activation = activation
        self.dense = nn.Linear(hid_size, out_size)
        self.residual = residual
        if normalize_out:
            self.out_norm = nn.LayerNorm(out_size)

    def forward(self, inp, adj):
        hid = inp
        for i in range(len(self.convs)):
            hid = self.convs[i](hid, adj)
            if hasattr(self, 'hid_norm'):
                hid = self.hid_norms[i](hid)
            hid = self.activation(hid)
        hid = self.dense(hid)
        if self.residual:
            hid += inp
        if hasattr(self, 'out_norm'):
            hid = self.out_norm(hid)
        return hid


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    Stolen from https://github.com/Diego999/pyGAT with slight modification in `edge = adj.t()`.
    """

    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj.t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*out x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]),
                                     torch.ones(size=(N, 1)).cuda()) + 1e-15
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(
            self.out_features) + ')'


class SpGraphAttentionBlock(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.0, alpha=0.2):
        """Sparse version of GAT."""
        super(SpGraphAttentionBlock, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(in_features,
                                                 out_features,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(out_features * num_heads,
                                             out_features,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)
        self.dense = nn.Linear(out_features, out_features)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.dense(x)

        return F.elu(x)


def make_adjacency_matrix(edges, device='cuda', dtype=torch.float32):
    edge_ix_pairs = [(from_i, to_i) for from_i in edges for to_i in edges[from_i]]
    adj_edges = torch.tensor(edge_ix_pairs, device=device, dtype=torch.int64)
    adj_values = torch.ones([len(adj_edges)], device=device)
    adj = torch.sparse_coo_tensor(adj_edges.t(), adj_values, dtype=dtype, device=device)
    return adj


def encode_indices(indices, size, scale=1.0, dtype=torch.float32):
    """
    Uses sinusoid position encoding to represent indices
    :param indices: integer tensor [*input_dims]
    :param size: output encoding size
    :returns: encoded tensor [*input_dims, size]
    """
    unit_index = torch.arange(size, dtype=dtype, device=indices.device)
    unit_index = unit_index.view(*([1] * len(indices.shape) + [size]))
    indices = indices[..., None].to(dtype=dtype)
    phase = indices / 10000.0 ** (float(scale) * (unit_index // 2) / size)
    encoding = (unit_index % 2) * torch.sin(phase) + (1 - unit_index % 2) * torch.cos(phase)
    return encoding
