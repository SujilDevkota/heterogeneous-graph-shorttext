"""Build the heterogeneous document/topic/entity graph and its adjacency blocks.

Node order is [documents | entities | topics]. Edges:

  * document-entity : a document links to each entity it mentions
  * entity-entity   : cosine(entity_vec_i, entity_vec_j) > threshold, with a
                      top-k back-fill so no entity is left isolated
  * document-topic  : a document links to its top-P LDA topics

The full symmetric adjacency (with self-loops) is symmetric-normalised
(D^-1/2 (A+I) D^-1/2) and then sliced into 3x3 type-pair blocks that the HGAT
model consumes, exactly as in the paper's data pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .preprocess import ProcessedCorpus
from .topics import TopicModel

TYPES = ["text", "entity", "topic"]


@dataclass
class GraphData:
    feats: list[torch.Tensor]              # one (n_type, d_type) matrix per node type
    adj: list[list[torch.Tensor]]          # 3x3 sparse normalised adjacency blocks
    labels: torch.Tensor                   # (n_text,)
    idx_train: torch.Tensor
    idx_val: torch.Tensor
    idx_test: torch.Tensor
    dims: list[int]
    counts: list[int]                      # [n_text, n_entity, n_topic]


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + 1e-12)


def _sym_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    """D^-1/2 (A) D^-1/2 for a symmetric adjacency that already has self-loops."""
    deg = np.asarray(adj.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, out=d_inv_sqrt, where=deg > 0)
    D = sp.diags(d_inv_sqrt)
    return (D @ adj @ D).tocsr()


def _entity_entity_edges(names: list[str], vecs: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Cosine-similarity edges among entity nodes with a min-degree back-fill."""
    thr = cfg["entities"]["entity_entity_sim"]
    min_deg = cfg["entities"]["min_degree"]
    unit = _l2(vecs)
    sims = unit @ unit.T                       # (E, E) cosine similarities
    np.fill_diagonal(sims, -1.0)

    edges: set[tuple[int, int]] = set()
    for i in range(len(names)):
        # all neighbours above the threshold ...
        above = np.where(sims[i] >= thr)[0]
        # ... plus top-k most similar so degree >= min_deg (fixes the isolated-entity case)
        if len(above) < min_deg:
            above = np.argsort(-sims[i])[:min_deg]
        for j in above:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a != b:
                edges.add((a, b))
    return list(edges)


def build_graph(dataset, corpus: ProcessedCorpus, topics: TopicModel, cfg: dict) -> GraphData:
    n_text = dataset.n_docs

    # ---- entity nodes: keep entities that appear in >=1 document and have a vector ----
    used = sorted({e for ents in corpus.doc_entities for e in ents
                   if e in corpus.entity_vectors})
    eid = {name: k for k, name in enumerate(used)}
    n_entity = len(used)
    entity_vecs = np.stack([corpus.entity_vectors[n] for n in used]).astype(np.float32) \
        if used else np.zeros((0, corpus.dim), np.float32)

    n_topic = topics.n_topics
    N = n_text + n_entity + n_topic

    def t_off(i):    # global index of the i-th text node
        return i

    def e_off(i):
        return n_text + i

    def p_off(i):
        return n_text + n_entity + i

    # ---- edges ----
    rows, cols = [], []

    def add(u, v):
        rows.append(u); cols.append(v)
        rows.append(v); cols.append(u)

    # document-entity
    for d, ents in enumerate(corpus.doc_entities):
        for e in ents:
            if e in eid:
                add(t_off(d), e_off(eid[e]))

    # entity-entity
    for i, j in _entity_entity_edges(used, entity_vecs, cfg):
        add(e_off(i), e_off(j))

    # document-topic (top-P topics per document)
    P = cfg["topics"]["top_topics_per_doc"]
    top = np.argsort(-topics.doc_topic, axis=1)[:, :P]
    for d in range(n_text):
        for t in top[d]:
            add(t_off(d), p_off(int(t)))

    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    adj.data[:] = 1.0                                  # collapse any duplicate edges
    adj = adj + sp.eye(N, format="csr")                # self-loops
    adj.data[:] = np.minimum(adj.data, 1.0)
    adj_norm = _sym_normalize(adj)

    # ---- slice into 3x3 type blocks ----
    spans = [(0, n_text), (n_text, n_text + n_entity), (n_text + n_entity, N)]
    blocks: list[list[torch.Tensor]] = []
    for (r0, r1) in spans:
        row_blocks = []
        for (c0, c1) in spans:
            row_blocks.append(_to_torch_sparse(adj_norm[r0:r1, c0:c1]))
        blocks.append(row_blocks)

    # ---- features (L2 for embeddings, L1 for the topic distribution) ----
    feats = [
        torch.tensor(_l2(corpus.doc_vectors), dtype=torch.float32),
        torch.tensor(_l2(entity_vecs) if n_entity else entity_vecs, dtype=torch.float32),
        torch.tensor(topics.topic_word, dtype=torch.float32),
    ]
    dims = [f.shape[1] for f in feats]

    return GraphData(
        feats=feats,
        adj=blocks,
        labels=torch.tensor(dataset.labels, dtype=torch.long),
        idx_train=torch.tensor(dataset.idx_train, dtype=torch.long),
        idx_val=torch.tensor(dataset.idx_val, dtype=torch.long),
        idx_test=torch.tensor(dataset.idx_test, dtype=torch.long),
        dims=dims,
        counts=[n_text, n_entity, n_topic],
    )


def _to_torch_sparse(m: sp.spmatrix) -> torch.Tensor:
    m = m.tocoo()
    if m.nnz == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=m.shape,
        ).coalesce()
    idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    val = torch.tensor(m.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=m.shape).coalesce()
