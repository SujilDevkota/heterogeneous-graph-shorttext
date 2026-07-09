"""Load AG News and build a balanced subset with a few-label semi-supervised split."""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    texts: list[str]            # document text (title + description)
    labels: np.ndarray          # int class id per document, shape (n_docs,)
    class_names: list[str]
    idx_train: np.ndarray       # document indices with labels used for training
    idx_val: np.ndarray
    idx_test: np.ndarray

    @property
    def n_docs(self) -> int:
        return len(self.texts)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def load_agnews(cfg: dict, rng: np.random.Generator) -> Dataset:
    """Read the AG News CSV, draw a balanced subset, and make train/val/test splits.

    The CSV has no header: columns are (class 1-4, title, description).
    A document's text is ``title. description`` (as in the paper's corpus).
    """
    dc = cfg["data"]
    class_names = dc["classes"]
    per_class = dc["docs_per_class"]

    # Collect documents per class.
    buckets: dict[int, list[str]] = {c: [] for c in range(len(class_names))}
    with open(dc["csv"], newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            cls = int(row[0]) - 1                      # 1..4 -> 0..3
            if cls not in buckets:
                continue
            title, desc = row[1].strip(), row[2].strip()
            text = (title + ". " + desc).replace("\\", " ").strip()
            buckets[cls].append(text)

    # Balanced random subset: per_class documents per class.
    texts: list[str] = []
    labels: list[int] = []
    for cls, docs in buckets.items():
        if len(docs) < per_class:
            raise ValueError(f"class {cls} has only {len(docs)} docs (< {per_class})")
        chosen = rng.choice(len(docs), size=per_class, replace=False)
        for i in chosen:
            texts.append(docs[i])
            labels.append(cls)
    labels = np.asarray(labels, dtype=np.int64)

    # Stratified few-label split: labels_per_class_train / _val labelled per class,
    # remainder is the (unlabelled during training) test set.
    n_tr = dc["labels_per_class_train"]
    n_va = dc["labels_per_class_val"]
    idx_train, idx_val, idx_test = [], [], []
    for cls in range(len(class_names)):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        idx_train.extend(cls_idx[:n_tr])
        idx_val.extend(cls_idx[n_tr:n_tr + n_va])
        idx_test.extend(cls_idx[n_tr + n_va:])

    return Dataset(
        texts=texts,
        labels=labels,
        class_names=list(class_names),
        idx_train=np.asarray(sorted(idx_train), dtype=np.int64),
        idx_val=np.asarray(sorted(idx_val), dtype=np.int64),
        idx_test=np.asarray(sorted(idx_test), dtype=np.int64),
    )
