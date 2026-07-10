# Heterogeneous Graph Attention Networks for Short-Text News Classification

Semi-supervised short-text **news classification** with a **Heterogeneous Graph Attention Network (HGAT)**. This repository presents the paper *Pretrained-Embedding Node Initialization for Heterogeneous-Graph Semi-Supervised Short-Text News Classification* (2021) — the IEEE-format manuscript, a project website, a runnable implementation, and the dataset used. The paper refines the author's earlier submitted thesis document (corrected method description, accurate dataset naming, verified results); prior work is compared up to 2020, contemporaneous with the study.

**Authors:** Sujil Devkota, Aman Shakya (Tribhuvan University, Institute of Engineering, Pulchowk Campus)
**Base model:** HGAT — Linmei Hu et al., *“Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification,”* EMNLP 2019 / ACM TOIS 2021.

🌐 **Project site:** https://sujildevkota.github.io/heterogeneous-graph-shorttext/

---

## What this is

News headlines are short (little co-occurrence signal) and labels are scarce. The approach builds a **heterogeneous graph** over the corpus with three node types — **Document**, **Topic** (LDA), and **Entity** (Wikipedia entity linking via TAGME) — and classifies documents transductively with HGAT's dual-level (node- and type-level) attention, so unlabeled documents contribute through the graph.

The specific question studied here is **node-feature initialization**: replacing high-dimensional TF-IDF node features with low-dimensional pretrained embeddings (word2vec for entities, pooled word vectors for documents) while keeping the HGAT architecture fixed. Results on an AG News subset and a HuffPost News Category subset are reported in the paper, with a clear account of the setup and its limits.

## Repository contents

| Path | Contents |
|------|----------|
| [`index.html`](index.html) | Project website (served via GitHub Pages) — overview, method, results, and scope. |
| [`paper/ieee-paper.pdf`](paper/ieee-paper.pdf) | The IEEE-format manuscript. |
| [`code/`](code/) | A modern, runnable Python re-implementation of the pipeline (see [`code/README.md`](code/README.md)). |
| [`data/ag_news/`](data/ag_news/) | The AG News corpus used (see the folder README for details). |

## Running the code

A clean re-implementation (Python 3.13 / PyTorch 2.x) lives in [`code/`](code/). It
builds the document/topic/entity graph and trains HGAT end-to-end:

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python run.py
```

It runs the paper's central comparison **controlled** — identical graph, splits,
and seeds, with only the node features changed:

| Node features | AG News accuracy (5 seeds) |
|---------------|---------------------------|
| TF-IDF (baseline) | 79.48 ± 1.19 |
| **Pretrained embeddings (proposed)** | **84.38 ± 0.47** |

**Gain: +4.90 points** (paired t = 8.33, p = 0.0011) — confirming the paper's
finding under a multi-seed protocol. Each mode takes ~5 minutes on a laptop.
The graph structure and HGAT model are faithful to the paper; TAGME and
Google-News word2vec are swapped for offline equivalents (spaCy NER + GloVe
vectors) so it runs with no API key or large download. See
[`code/README.md`](code/README.md) for details.

## Data

- **AG News** (`data/ag_news/`): the standard 4-class news corpus (120k train / 7.6k test). The paper used a **random 6,000-document balanced subset** (1,500 per class) sampled from this. See [`data/ag_news/README.md`](data/ag_news/README.md).
- **HuffPost News Category** (5-class subset): the second dataset is Misra's *News Category Dataset* (Kaggle). It is **not redistributed here** — download it from the source. *(Note: earlier drafts called this "TagMyNews"; it is the HuffPost corpus, a different dataset from the TagMyNews benchmark of Vitale et al. 2012.)*

## Scope & limitations

The paper is scoped as a study of node-feature initialization on a fixed HGAT graph, and reports its results as single-run measurements. Its limitations are stated in the paper's *Limitations and Future Work* section, chiefly:

- The 2021 results are single-run (one seed, one split) — now addressed by the released implementation's multi-seed controlled comparison (see above).
- The AG News baseline is quoted from the base paper; the HuffPost comparison is a matched re-run.
- The 2021 experiments had no ablation isolating the feature change — now provided by the two feature modes in `code/`.
- The setting is transductive, and efficiency is not measured.

Remaining next steps — contextual embeddings for node features, a neural topic model, and an inductive variant — are noted as future work.

## Acknowledgements

Built on the HGAT model and reference implementation of Linmei Hu, Tianchi Yang, Chuan Shi, Houye Ji, and Xiaoli Li. AG News is from Gulli's news corpus (Zhang et al., 2015). Entity linking uses TAGME (Ferragina & Scaiella, 2010); topic modeling uses gensim/scikit-learn LDA.

## Citation

If you refer to this work, please cite the paper:

```bibtex
@misc{devkota2021hgat,
  author       = {Sujil Devkota and Aman Shakya},
  title        = {Pretrained-Embedding Node Initialization for Heterogeneous-Graph
                  Semi-Supervised Short-Text News Classification},
  year         = {2021},
  howpublished = {\url{https://github.com/SujilDevkota/heterogeneous-graph-shorttext}}
}
```
