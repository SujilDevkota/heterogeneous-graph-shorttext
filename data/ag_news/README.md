# AG News data

The standard **AG News** topic-classification corpus (4 balanced classes).

| File | Rows | Notes |
|------|------|-------|
| `train.csv` | 120,000 | 30,000 per class |
| `test.csv`  | 7,600   | 1,900 per class |

**Format:** CSV with three columns and no header — `"class","title","description"`.
Class index: `1` = World, `2` = Sports, `3` = Business, `4` = Sci/Tech.

## How the paper used it

The paper did **not** train on the full corpus. It sampled a **random, balanced 6,000-document subset** (1,500 documents per class) and, within that, used a small labeled set (40 labeled documents per class for training, 40 per class for validation, remainder as unlabeled/test) for the semi-supervised setting. The subset was drawn from `train.csv` above.

> Reproducibility note: the released implementation in [`code/`](../../code/) regenerates a balanced subset and stratified few-label split deterministically from a fixed seed (see `code/config.yaml`), so the controlled comparison is fully reproducible from these CSVs.

## Source

AG News is a public benchmark (Zhang, Zhao & LeCun, 2015; original corpus by Gulli). These CSVs mirror the widely used version distributed with the CharCNN dataset release.
