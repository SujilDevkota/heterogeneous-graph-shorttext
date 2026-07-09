# Heterogeneous Graph Attention Networks for Short-Text News Classification

Semi-supervised short-text **news classification** with a **Heterogeneous Graph Attention Network (HGAT)**. This repository accompanies an M.Sc. thesis (Tribhuvan University, Institute of Engineering, Pulchowk Campus, 2021), presented as an IEEE-format paper, a project website, and the dataset used.

**Author:** Sujil Devkota · **Thesis supervisor:** Dr. Aman Shakya
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
| [`paper/thesis.pdf`](paper/thesis.pdf) | The final MSc thesis. |
| [`slides/`](slides/) | Presentation slides (`.pptx` and `.pdf`). |
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

On the 6,000-document AG News subset (40 labels/class) it reaches **84.2% ± 0.5**
accuracy over 5 seeds in ~5 minutes. The graph structure and HGAT model are
faithful to the paper; TAGME and Google-News word2vec are swapped for offline
equivalents (spaCy NER + GloVe vectors) so it runs with no API key or large
download. See [`code/README.md`](code/README.md) for details.

## Data

- **AG News** (`data/ag_news/`): the standard 4-class news corpus (120k train / 7.6k test). The thesis used a **random 6,000-document balanced subset** (1,500 per class) sampled from this. See [`data/ag_news/README.md`](data/ag_news/README.md).
- **HuffPost News Category** (5-class subset): the second dataset is Misra's *News Category Dataset* (Kaggle). It is **not redistributed here** — download it from the source. *(Note: earlier drafts called this "TagMyNews"; it is the HuffPost corpus, a different dataset from the TagMyNews benchmark of Vitale et al. 2012.)*

## Scope & limitations

The paper is scoped as a study of node-feature initialization on a fixed HGAT graph, and reports its results as single-run measurements. Its limitations are stated in the paper's *Limitations and Future Work* section, chiefly:

- Results are single-run (one seed, one split); no variance or significance test.
- The AG News baseline is quoted from the base paper; the HuffPost comparison is a matched re-run.
- No ablation isolates the feature change from the graph.
- The setting is transductive, and efficiency is not measured.

Natural next steps — multi-seed evaluation, contextual embeddings for node features, a neural topic model, and an inductive variant — are noted as future work.

## Acknowledgements

Built on the HGAT model and reference implementation of Linmei Hu, Tianchi Yang, Chuan Shi, Houye Ji, and Xiaoli Li. AG News is from Gulli's news corpus (Zhang et al., 2015). Entity linking uses TAGME (Ferragina & Scaiella, 2010); topic modeling uses gensim/scikit-learn LDA.

## Citation

If you refer to this work, please cite the MSc thesis:

```bibtex
@mastersthesis{devkota2021hgat,
  author = {Sujil Devkota},
  title  = {Heterogeneous Graph Attention Network for Semi-Supervised News Classification},
  school = {Institute of Engineering, Pulchowk Campus, Tribhuvan University},
  year   = {2021}
}
```
