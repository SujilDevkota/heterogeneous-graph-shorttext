"""Transductive training and evaluation, with multi-seed mean +/- std reporting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from .graph import GraphData
from .model import HGAT


def _to_device(graph: GraphData, device) -> GraphData:
    graph.feats = [f.to(device) for f in graph.feats]
    graph.adj = [[a.to(device) for a in row] for row in graph.adj]
    graph.labels = graph.labels.to(device)
    graph.idx_train = graph.idx_train.to(device)
    graph.idx_val = graph.idx_val.to(device)
    graph.idx_test = graph.idx_test.to(device)
    return graph


def _metrics(logits, labels, idx):
    pred = logits[idx].argmax(1).cpu().numpy()
    true = labels[idx].cpu().numpy()
    acc = float((pred == true).mean())
    p = precision_score(true, pred, average="macro", zero_division=0)
    r = recall_score(true, pred, average="macro", zero_division=0)
    f1 = f1_score(true, pred, average="macro", zero_division=0)
    return acc, p, r, f1


def run_once(graph: GraphData, cfg: dict, seed: int, device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mc = cfg["model"]

    model = HGAT(graph.dims, mc["hidden"], int(graph.labels.max()) + 1,
                 mc["dropout"], mc["gamma"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])

    best = {"val_acc": -1.0, "epoch": -1, "test": (0, 0, 0, 0)}
    since = 0
    for epoch in range(mc["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(graph.feats, graph.adj)
        loss = F.nll_loss(out[graph.idx_train], graph.labels[graph.idx_train])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(graph.feats, graph.adj)
            val_acc = _metrics(out, graph.labels, graph.idx_val)[0]
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "epoch": epoch,
                        "test": _metrics(out, graph.labels, graph.idx_test)}
                since = 0
            else:
                since += 1
        if since >= mc["patience"]:
            break

    acc, p, r, f1 = best["test"]
    return {"seed": seed, "val_acc": best["val_acc"], "epoch": best["epoch"],
            "acc": acc, "precision": p, "recall": r, "f1": f1}


def run_multi(graph: GraphData, cfg: dict, device) -> dict:
    graph = _to_device(graph, device)
    runs = [run_once(graph, cfg, s, device) for s in cfg["model"]["seeds"]]
    accs = np.array([r["acc"] for r in runs])
    f1s = np.array([r["f1"] for r in runs])
    return {
        "runs": runs,
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
        "acc_best": float(accs.max()),
    }
