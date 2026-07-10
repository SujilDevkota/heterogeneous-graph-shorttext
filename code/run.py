#!/usr/bin/env python
"""End-to-end pipeline: AG News -> heterogeneous graph -> HGAT -> results.

    python run.py [--config config.yaml]

Reproduces the paper's method with fully-offline components (spaCy NER + GloVe
vectors in place of TAGME + Google-News word2vec). Prints per-seed and
mean +/- std accuracy / macro-F1, and writes a JSON summary to artifacts/.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from hgat import data, features, graph, preprocess, topics, train


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--features", choices=["embedding", "tfidf"], default=None,
                    help="override features.mode: 'embedding' (proposed) or 'tfidf' (baseline)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # resolve a relative data path against the config file's directory
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    cfg["data"]["csv"] = os.path.normpath(os.path.join(cfg_dir, cfg["data"]["csv"]))
    work = os.path.join(here, cfg["paths"]["work_dir"])
    os.makedirs(work, exist_ok=True)

    device = get_device()
    t0 = time.time()
    print(f"device: {device}")

    rng = np.random.default_rng(cfg["seed"])
    ds = data.load_agnews(cfg, rng)
    print(f"documents: {ds.n_docs}  classes: {ds.n_classes}  "
          f"train/val/test: {len(ds.idx_train)}/{len(ds.idx_val)}/{len(ds.idx_test)}")

    print("loading spaCy en_core_web_lg ...")
    import spacy
    nlp = spacy.load("en_core_web_lg", disable=["parser"])

    corpus = preprocess.process_corpus(ds.texts, nlp, cfg)
    print(f"unique entities with vectors: {len(corpus.entity_vectors)}")

    tm = topics.fit_topics(corpus.lemmas, cfg, cfg["seed"])
    print(f"topics: {tm.n_topics}")

    mode = args.features or cfg["features"]["mode"]
    used = graph.used_entities(corpus)
    feats = features.build_features(corpus, tm, used, cfg, mode)
    g = graph.build_graph(ds, corpus, tm, cfg, feats, used)
    n_text, n_ent, n_top = g.counts
    n_edges = sum(a._nnz() for row in g.adj for a in row)
    print(f"features: {mode}  |  graph nodes -> text:{n_text} entity:{n_ent} topic:{n_top}  "
          f"feature dims: {g.dims}  directed edges (incl. self-loops): {n_edges}")

    print(f"\ntraining HGAT ({len(cfg['model']['seeds'])} seeds) ...")
    res = train.run_multi(g, cfg, device)
    for r in res["runs"]:
        print(f"  seed {r['seed']:>2}: acc={r['acc']*100:5.2f}  f1={r['f1']*100:5.2f}  "
              f"(val {r['val_acc']*100:5.2f} @ epoch {r['epoch']})")
    print(f"\nRESULT  accuracy = {res['acc_mean']*100:.2f} +/- {res['acc_std']*100:.2f}  "
          f"(best {res['acc_best']*100:.2f})   macro-F1 = {res['f1_mean']*100:.2f} "
          f"+/- {res['f1_std']*100:.2f}")
    print(f"elapsed: {time.time()-t0:.0f}s")

    summary = {
        "features": mode,
        "documents": ds.n_docs, "classes": ds.class_names,
        "graph": {"text": n_text, "entity": n_ent, "topic": n_top},
        "labels_per_class_train": cfg["data"]["labels_per_class_train"],
        **{k: res[k] for k in ["acc_mean", "acc_std", "f1_mean", "f1_std", "acc_best"]},
        "runs": res["runs"],
    }
    out_json = os.path.join(work, f"results_{mode}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_json}")

    # If the other feature mode has also been run, print the paper's central
    # comparison: paired over the matched seeds on the identical graph/splits.
    other = "tfidf" if mode == "embedding" else "embedding"
    other_json = os.path.join(work, f"results_{other}.json")
    if os.path.exists(other_json):
        from scipy import stats
        with open(other_json) as f:
            o = json.load(f)
        a = np.array([r["acc"] for r in summary["runs"]])
        b = np.array([r["acc"] for r in o["runs"]])
        if len(a) == len(b) > 1:
            emb, tfi = (a, b) if mode == "embedding" else (b, a)
            t_stat, p = stats.ttest_rel(emb, tfi)
            print(f"\nCONTROLLED COMPARISON (identical graph/splits/seeds)"
                  f"\n  embedding: {emb.mean()*100:.2f}%   tfidf: {tfi.mean()*100:.2f}%"
                  f"   gain: {(emb-tfi).mean()*100:+.2f} pts"
                  f"   paired t = {t_stat:.2f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
