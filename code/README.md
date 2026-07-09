# Modern reproduction of the HGAT pipeline

A clean, runnable re-implementation of the paper's method in modern Python
(tested on **Python 3.13, PyTorch 2.x**). It builds a document / topic / entity
heterogeneous graph from AG News and classifies it with HGAT
(node-level + type-level attention) in a transductive semi-supervised setting.

```
code/
├── run.py                # end-to-end: data → graph → HGAT → results
├── config.yaml           # all settings (matches the paper where possible)
├── requirements.txt
└── hgat/
    ├── data.py           # AG News → balanced 6,000-doc subset + few-label split
    ├── preprocess.py     # one spaCy pass: lemmas, doc vectors, entities
    ├── topics.py         # scikit-learn LDA (K=15, α=β=0.1)
    ├── graph.py          # D/T/E graph + symmetric-normalised adjacency blocks
    ├── model.py          # HGAT: sparse node-level + type-level attention
    └── train.py          # transductive training, multi-seed mean ± std
```

## Quick start

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python run.py                      # ~a few minutes on a laptop (CPU/MPS/CUDA)
```

Results (per-seed and mean ± std accuracy / macro-F1) print to the console and
are written to `artifacts/results.json`.

## What matches the paper, and what is substituted

The **graph structure and the HGAT model are faithful** to the original:
three node types (Document, Topic, Entity); document–topic, document–entity, and
entity–entity edges; symmetric-normalised adjacency; two HGAT layers with
node-level (GAT-style additive) and type-level (self) attention; transductive
training with a few labels per class.

Two 2021 external dependencies are replaced with **fully-offline equivalents** so
the pipeline runs without an API key or a multi-GB download:

| Stage | Paper (2021) | This reproduction |
|-------|--------------|-------------------|
| Entity linking | TAGME → Wikipedia (API token) | spaCy named-entity recognition |
| Word vectors (300-d) | Google-News word2vec (3.3 GB) | spaCy `en_core_web_lg` GloVe vectors |
| Topic model | LDA (K=15) | LDA (K=15) — unchanged |
| Model | HGAT dual-level attention | HGAT dual-level attention — unchanged |

Both substitutions keep the method identical in structure; only the source of the
entities and of the 300-d embeddings differs. To use the exact TAGME + word2vec
inputs, provide a TAGME token and a word2vec file and swap the two functions in
`preprocess.py` (the interfaces are isolated for this).

## Improvements over the 2021 code (from the paper's limitations)

- **Multi-seed evaluation** with mean ± std (the 2021 run was single-seed).
- **Sparse, edge-wise node attention** instead of the dense *O(N²)* materialisation
  that caused the original's ~12 GB memory requirement.
- **L2-normalised** embedding features (the 2021 code applied an L1 normaliser
  written for non-negative TF-IDF to signed embeddings).
- Fixed the entity–entity top-k back-fill (the original had a loop-index bug).
- Clean package, config file, fixed seeds, and CPU/MPS/CUDA support — no hard-coded
  Colab paths.

## Results

Running `python run.py` with the default `config.yaml` on the 6,000-document
AG News subset (160 train / 160 val / 5,680 test — 40 labels per class),
graph = **6,000** document + **9,513** entity + **15** topic nodes:

| Metric | Result (5 seeds) |
|--------|------------------|
| Accuracy | **84.24% ± 0.47** (best 84.74) |
| Macro-F1 | **84.28% ± 0.50** |
| Runtime | ~5 min on Apple-silicon MPS |

Per-seed accuracy: 84.52 / 83.38 / 84.74 / 84.15 / 84.40.

> These numbers are **not** directly comparable to the paper's single-run 76.30%:
> the entity linker and word embeddings differ (spaCy NER + GloVe vs. TAGME +
> Google-News word2vec), the results are averaged over 5 seeds rather than one
> run, and several implementation bugs were fixed. The purpose of this repository
> is a faithful, reproducible, **runnable** version of the method — that it lands
> in a sensible range for AG News is the confirmation that the pipeline is sound.
