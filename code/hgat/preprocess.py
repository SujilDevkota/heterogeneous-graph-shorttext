"""Single spaCy pass over the corpus.

Produces everything the downstream stages need, computed once:

  * ``lemmas``          - cleaned lemma strings per document (for the LDA bag-of-words)
  * ``doc_vectors``     - mean of the document's token GloVe vectors (300-d) -> document node features
  * ``doc_entities``    - normalised named entities per document -> document-entity edges
  * ``entity_vectors``  - 300-d vector per unique entity -> entity node features + entity-entity edges

The mean-pooled document vector matches the paper's document feature (a centroid
of word vectors). Named entities from spaCy's NER stand in for the paper's TAGME
Wikipedia entities; spaCy's 300-d GloVe vectors stand in for Google-News word2vec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

# spaCy NER labels that behave like the "concept" entities TAGME links.
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                  "WORK_OF_ART", "FAC", "NORP", "LAW", "LANGUAGE"}


@dataclass
class ProcessedCorpus:
    lemmas: list[str]                              # space-joined lemmas per document
    doc_vectors: np.ndarray                        # (n_docs, 300)
    doc_entities: list[list[str]]                  # normalised entities per document
    entity_vectors: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.doc_vectors.shape[1])


def _norm_entity(text: str) -> str:
    return "_".join(text.lower().split())


def process_corpus(texts: list[str], nlp, cfg: dict) -> ProcessedCorpus:
    min_len = cfg["entities"]["min_len"]
    dim = nlp.vocab.vectors_length or 300

    lemmas: list[str] = []
    doc_vectors = np.zeros((len(texts), dim), dtype=np.float32)
    doc_entities: list[list[str]] = []
    entity_vectors: dict[str, np.ndarray] = {}

    for i, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=128),
                                 total=len(texts), desc="spaCy")):
        # Lemmas for the LDA bag-of-words.
        toks = [t.lemma_.lower() for t in doc
                if t.is_alpha and not t.is_stop and len(t) > 1]
        lemmas.append(" ".join(toks))

        # Document embedding: mean of in-vocabulary token vectors.
        vecs = [t.vector for t in doc if t.has_vector and t.is_alpha and not t.is_stop]
        if vecs:
            doc_vectors[i] = np.mean(vecs, axis=0)

        # Named entities -> entity nodes (kept only if they carry a vector).
        ents: list[str] = []
        for ent in doc.ents:
            if ent.label_ not in _ENTITY_LABELS:
                continue
            name = _norm_entity(ent.text)
            if len(name) < min_len or name.replace("_", "").isdigit():
                continue
            if not ent.has_vector or ent.vector_norm == 0:
                continue
            ents.append(name)
            if name not in entity_vectors:
                entity_vectors[name] = np.asarray(ent.vector, dtype=np.float32)
        doc_entities.append(sorted(set(ents)))

    return ProcessedCorpus(
        lemmas=lemmas,
        doc_vectors=doc_vectors,
        doc_entities=doc_entities,
        entity_vectors=entity_vectors,
    )
