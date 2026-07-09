"""Modern re-implementation of the HGAT short-text news classification pipeline.

Reproduces the method described in the paper *Pretrained-Embedding Node
Initialization for Heterogeneous-Graph Semi-Supervised Short-Text News
Classification* (Devkota & Shakya, 2021):

    documents + LDA topics + linked entities  ->  heterogeneous graph
    -> HGAT with node-level and type-level attention  ->  transductive classification.

The graph structure and the HGAT model are faithful to the original; two
external dependencies of the 2021 code are replaced with fully-offline
equivalents so the pipeline runs end-to-end on a laptop:

  * entity linking : TAGME (Wikipedia)      -> spaCy named-entity recognition
  * word vectors   : Google-News word2vec   -> spaCy 300-d GloVe vectors

Both substitutions keep the three node types, the graph construction, and the
model identical; only the source of entities and of the 300-d embeddings differ.
"""

__all__ = ["data", "preprocess", "topics", "features", "graph", "model", "train"]
