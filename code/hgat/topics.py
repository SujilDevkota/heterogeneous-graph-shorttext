"""LDA topic modelling (scikit-learn), matching the paper's topic stage.

Fits Latent Dirichlet Allocation on a bag-of-words of the corpus (K=15,
alpha=beta=0.1 for AG News). Returns:

  * ``doc_topic``   - document-topic distribution (n_docs, K)   -> top-P doc-topic edges
  * ``topic_word``  - topic-word distribution (K, vocab)         -> topic node features
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class TopicModel:
    doc_topic: np.ndarray       # (n_docs, K)
    topic_word: np.ndarray      # (K, vocab)  -> L1-normalised topic node features
    n_topics: int


def fit_topics(lemmas: list[str], cfg: dict, seed: int) -> TopicModel:
    tc = cfg["topics"]
    vectorizer = CountVectorizer(max_features=tc["vocab_size"], min_df=tc["min_df"])
    bow = vectorizer.fit_transform(lemmas)

    lda = LatentDirichletAllocation(
        n_components=tc["n_topics"],
        doc_topic_prior=tc["doc_topic_prior"],
        topic_word_prior=tc["topic_word_prior"],
        max_iter=tc["max_iter"],
        learning_method="batch",
        random_state=seed,
    )
    doc_topic = lda.fit_transform(bow).astype(np.float32)

    # Topic node feature = topic-word distribution (row-normalised), as in the paper.
    topic_word = lda.components_.astype(np.float32)
    topic_word /= topic_word.sum(axis=1, keepdims=True) + 1e-12

    return TopicModel(doc_topic=doc_topic, topic_word=topic_word, n_topics=tc["n_topics"])
