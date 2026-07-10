"""Node-feature construction — the variable this work studies.

Two modes, on the *identical* graph structure, so any accuracy difference
isolates the node-feature initialization (the paper's central comparison):

  * ``embedding`` (proposed) : 300-d pretrained vectors — documents use the
    mean of their token vectors, entities their span vector; topic nodes use
    their LDA topic-word distribution.
  * ``tfidf`` (baseline)     : sparse TF-IDF vectors over the corpus vocabulary
    for documents and for entity names; topic nodes unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import ProcessedCorpus
from .topics import TopicModel


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + 1e-12)


def build_features(corpus: ProcessedCorpus, topics: TopicModel,
                   used_entities: list[str], cfg: dict, mode: str) -> list[np.ndarray]:
    """Return [doc_feats, entity_feats, topic_feats] for the requested mode."""
    if mode == "embedding":
        ent = (np.stack([corpus.entity_vectors[n] for n in used_entities])
               .astype(np.float32) if used_entities
               else np.zeros((0, corpus.dim), np.float32))
        return [
            _l2(corpus.doc_vectors.astype(np.float32)),
            _l2(ent) if len(used_entities) else ent,
            topics.topic_word,
        ]

    if mode == "tfidf":
        vec = TfidfVectorizer(max_features=cfg["features"]["tfidf_vocab"],
                              min_df=2)                      # rows are L2-normalised
        doc = vec.fit_transform(corpus.lemmas).toarray().astype(np.float32)
        ent_text = [n.replace("_", " ") for n in used_entities]
        ent = (vec.transform(ent_text).toarray().astype(np.float32)
               if used_entities else np.zeros((0, doc.shape[1]), np.float32))
        if len(used_entities):
            n_zero = int((np.abs(ent).sum(axis=1) == 0).sum())
            print(f"tfidf baseline: {n_zero}/{len(used_entities)} entity rows are "
                  f"all-zero (name tokens outside the {doc.shape[1]}-word vocabulary); "
                  f"those nodes still propagate structure through their edges")
        return [doc, ent, topics.topic_word]

    raise ValueError(f"unknown feature mode: {mode!r} (use 'embedding' or 'tfidf')")
