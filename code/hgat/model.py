"""Modern PyTorch re-implementation of HGAT (Linmei et al., 2019).

Two node-attention levels, faithful to the original design but implemented with
sparse, edge-wise attention (segment-softmax over each node's neighbours) instead
of the original dense N x N materialisation -- same mechanism, far less memory.

  * node-level attention : GAT-style additive attention over each adjacency block,
                           blended with the normalised adjacency by ``gamma``
  * type-level attention : self-attention that weights the message from each
                           source node type when updating a target node type
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of ``scores`` within groups given by ``index`` (0..n-1)."""
    m = scores.new_full((n,), float("-inf")).scatter_reduce(0, index, scores, reduce="amax")
    scores = torch.exp(scores - m[index])
    denom = scores.new_zeros(n).index_add(0, index, scores)
    return scores / (denom[index] + 1e-12)


class NodeAttention(nn.Module):
    """Node-level attention on one (target, source) adjacency block.

    With ``learn=False`` this is plain normalised-adjacency propagation and no
    attention parameters are created (used by layer 2, matching the original
    HGAT where the second layer is a graph convolution).
    """

    def __init__(self, dim: int, gamma: float, learn: bool = True):
        super().__init__()
        self.learn = learn
        if learn:
            self.a_tgt = nn.Parameter(torch.empty(dim, 1))
            self.a_src = nn.Parameter(torch.empty(dim, 1))
            nn.init.xavier_uniform_(self.a_tgt)
            nn.init.xavier_uniform_(self.a_src)
        self.gamma = gamma
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h_tgt, h_src, block):
        n_tgt = h_tgt.size(0)
        if block._nnz() == 0:
            return h_tgt.new_zeros(n_tgt, h_src.size(1))
        idx = block.indices()
        w = block.values()                       # normalised adjacency weight per edge
        i, j = idx[0], idx[1]
        if self.learn:
            e = self.leaky((h_tgt @ self.a_tgt).squeeze(-1)[i]
                           + (h_src @ self.a_src).squeeze(-1)[j])
            att = segment_softmax(e, i, n_tgt)
            coef = self.gamma * att + (1.0 - self.gamma) * w
        else:
            coef = w                             # plain adjacency propagation
        msg = coef.unsqueeze(-1) * h_src[j]
        return h_tgt.new_zeros(n_tgt, h_src.size(1)).index_add(0, i, msg)


class TypeAttention(nn.Module):
    """Type-level attention: weight each source-type message for a target type."""

    def __init__(self, dim: int, hidden: int, n_types: int, self_idx: int):
        super().__init__()
        self.self_idx = self_idx
        self.n_types = n_types
        self.linear = nn.Linear(dim, hidden)
        self.a = nn.Parameter(torch.empty(2 * hidden, 1))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        # msgs: (n, n_types, dim)
        x = self.linear(msgs)                                    # (n, T, hidden)
        self_x = x[:, self.self_idx:self.self_idx + 1, :].expand(-1, x.size(1), -1)
        u = self.leaky(torch.cat([x, self_x], dim=-1) @ self.a).squeeze(-1)   # (n, T)
        w = F.softmax(u, dim=1).unsqueeze(-1)                    # (n, T, 1)
        return (w * msgs).sum(dim=1) * self.n_types             # scale by #types


class HeteroLayer(nn.Module):
    """One HGAT layer: per-type projection -> node attention -> type attention."""

    def __init__(self, in_dims, out_dim, gamma, use_node_attention: bool):
        super().__init__()
        self.n_types = len(in_dims)
        self.proj = nn.ModuleList([nn.Linear(d, out_dim, bias=False) for d in in_dims])
        self.node_att = nn.ModuleList(
            [NodeAttention(out_dim, gamma, learn=use_node_attention) for _ in in_dims]
        )
        self.type_att = nn.ModuleList(
            [TypeAttention(out_dim, 32, self.n_types, t) for t in range(self.n_types)]
        )

    def forward(self, feats, adj):
        h = [self.proj[t](feats[t]) for t in range(self.n_types)]
        out = []
        for t1 in range(self.n_types):
            msgs = [self.node_att[t1](h[t1], h[t2], adj[t1][t2])
                    for t2 in range(self.n_types)]
            out.append(self.type_att[t1](torch.stack(msgs, dim=1)))
        return out


class HGAT(nn.Module):
    def __init__(self, dims, hidden, n_classes, dropout, gamma):
        super().__init__()
        self.dropout = dropout
        self.layer1 = HeteroLayer(dims, hidden, gamma, use_node_attention=True)
        self.layer2 = HeteroLayer([hidden] * len(dims), n_classes, gamma,
                                  use_node_attention=False)

    def forward(self, feats, adj):
        h = self.layer1(feats, adj)
        h = [F.dropout(F.relu(x), self.dropout, training=self.training) for x in h]
        out = self.layer2(h, adj)
        return F.log_softmax(out[0], dim=1)          # text (document) node logits
