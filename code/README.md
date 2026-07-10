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
    ├── features.py       # node features: embeddings (proposed) vs TF-IDF (baseline) — the studied variable
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
python run.py                      # proposed (embedding) mode — a few minutes on a laptop
python run.py --features tfidf     # baseline mode; prints the paired comparison when both exist
```

Results (per-seed and mean ± std accuracy / macro-F1) print to the console and
are written to `artifacts/results_<mode>.json` (`results_embedding.json`,
`results_tfidf.json`).

## What matches the paper, and what is substituted

The **graph structure and the HGAT model are faithful** to the original:
three node types (Document, Topic, Entity); document–topic, document–entity, and
entity–entity edges; symmetric-normalised adjacency; two HGAT layers — layer 1
applies node-level (GAT-style additive, γ-blended with the normalised adjacency)
and type-level attention, layer 2 applies plain graph convolution plus
type-level attention, matching the original code (`gc2 = GraphConvolution`);
transductive training with a few labels per class.

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

A few training hyperparameters also differ from the 2021 run, disclosed here and
in `config.yaml`: hidden 128 (2021: 512), dropout 0.5 (2021: 0.95), γ 0.5
(2021 code: 0.1), 200 epochs with early stopping (2021: 300 fixed). These are
lighter, more standard settings; both feature modes use the same values, so the
comparison stays controlled.

## Improvements over the 2021 code (from the paper's limitations)

- **Multi-seed evaluation** with mean ± std (the 2021 run was single-seed).
- **Sparse, edge-wise node attention** instead of the dense *O(N²)* materialisation
  that caused the original's ~12 GB memory requirement.
- **L2-normalised** embedding features (the 2021 code applied an L1 normaliser
  written for non-negative TF-IDF to signed embeddings).
- Fixed the entity–entity top-k back-fill (the original had a loop-index bug).
- Clean package, config file, fixed seeds, and CPU/MPS/CUDA support — no hard-coded
  Colab paths.

## Results: the paper's central comparison, controlled

`run.py` supports both node-feature modes on the **identical graph, splits, and
seeds** — so the difference isolates the feature initialization, which is the
paper's contribution:

```bash
python run.py --features embedding    # proposed: 300-d pretrained embeddings
python run.py --features tfidf       # baseline: sparse TF-IDF
```

On the 6,000-document AG News subset (160 train / 160 val / 5,680 test — 40
labels per class; graph = 6,000 document + 9,513 entity + 15 topic nodes):

| Node features | Accuracy (5 seeds) | Macro-F1 |
|---------------|--------------------|----------|
| TF-IDF (baseline) | 79.48 ± 1.19 | 79.43 ± 1.13 |
| **Pretrained embeddings (proposed)** | **84.38 ± 0.47** | **84.42 ± 0.47** |

**Gain: +4.90 points** (paired t = 8.33, **p = 0.0011** over 5 matched seeds;
± is the unbiased sample std, and reflects model-initialization/dropout variance
on a fixed split). The 2021 paper reported a single-run gain of +4.20 points on
AG News; the controlled, multi-seed version of the experiment confirms the
effect, and the embedding-initialized model is also more stable across seeds
(± 0.47 vs ± 1.19). Each mode takes ~4–5 minutes on Apple-silicon MPS (CPU also
works); running the second mode automatically prints the paired comparison.

> Absolute numbers differ from the paper's 2021 run because the entity linker
> and word vectors are offline substitutes (spaCy NER + GloVe vs. TAGME +
> Google-News word2vec) and several implementation bugs were fixed. What
> reproduces is the **finding**: dense pretrained node initialization beats
> sparse TF-IDF on the same heterogeneous graph under label scarcity.
