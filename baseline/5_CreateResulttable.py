#!/usr/bin/env python3
"""
Recomputes metrics directly from prediction and ground-truth CSV files,
following the same logic as the official evaluation script (main.py).
Produces the same markdown result table as script 5.

Prediction files (written by script 4):
  {results_dir}/test_prediction_files/{dataset}/{tag}/{session}/{role}.engagement.pred.csv
  {results_dir}/test_prediction_files/{dataset}/{tag}/{session}/{role}.{type}.pred.csv

Ground-truth annotation files:
  {gt_root}/noxi/test-base/{session}/{role}.engagement.annotation.csv
  {gt_root}/noxi/test-additional/{session}/{role}.engagement.annotation.csv
  {gt_root}/noxi-j/test/{session}/{role}.engagement.annotation.csv
  {gt_root}/mpiigroupinteraction/test/{session}/{role}.engagement.annotation.csv
  {gt_root}/pinsoro/test-cc/{session}/{role}.{type}.annotation.csv
  {gt_root}/pinsoro/test-cr/{session}/{role}.{type}.annotation.csv
"""

from __future__ import annotations

import math
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
#                         DATASET / MODALITY CONSTANTS
# ---------------------------------------------------------------------------

REGRESSION_DATASETS: list[tuple[str, str]] = [
    ("noxi-base",            "NOXI"),
    ("test-additional",      "NOXI (Add.)"),
    ("noxi-j",               "NOXI-J"),
    ("mpiigroupinteraction", "MPIIGI"),
]

CLASSIFICATION_DATASETS: list[tuple[str, str]] = [
    ("pinsoro-cc", "Pinsoro-CC"),
    ("pinsoro-cr", "Pinsoro-CR"),
]

# (modality_key, category_label, display_name)
MODALITY_ORDER: list[tuple[str, str, str]] = [
    ("openface2.stream",                 "Video", "OpenFace 2.0"),
    ("openface3.stream",                 "Video", "OpenFace 3.0"),
    ("openpose.stream",                  "Video", "OpenPose"),
    ("clip.stream",                      "Video", "CLIP"),
    ("dino.stream",                      "Video", "DINO"),
    ("swin.stream",                      "Video", "SwinTransformer"),
    ("videomae.stream",                  "Video", "VideoMAE"),
    ("audio.egemapsv2.stream",           "Voice", "eGeMAPS v2"),
    ("audio.w2vbert2_embeddings.stream", "Voice", "w2vBERT2"),
    ("xlm_roberta_embeddings.stream",    "Text",  "XLM RoBERTa"),
]

# Maps dataset key -> subdirectory within gt_root
GT_SUBDIR: dict[str, str] = {
    "noxi-base":            "noxi/test-base",
    "test-additional":      "noxi/test-additional",
    "noxi-j":               "noxi-j/test",
    "mpiigroupinteraction": "mpiigroupinteraction/test",
    "pinsoro-cc":           "pinsoro/test-cc",
    "pinsoro-cr":           "pinsoro/test-cr",
}

# Maps dataset key -> folder name in test_prediction_files/
# (only entries that differ from the key)
PRED_FOLDER: dict[str, str] = {
    "test-additional": "noxi-additional",
}


def _pred_folder(ds_key: str) -> str:
    return PRED_FOLDER.get(ds_key, ds_key)


def _mod_tag(mod_key: str) -> str:
    return mod_key.strip(".~")


# ---------------------------------------------------------------------------
#                              METRIC FUNCTIONS
# ---------------------------------------------------------------------------

def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    """CCC matching main.py's formula exactly (no epsilon)."""
    vx = np.var(x)
    vy = np.var(y)
    s_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    denom = vx + vy + (np.mean(x) - np.mean(y)) ** 2
    if denom == 0.0:
        return float("nan")
    return float((2 * s_xy) / denom)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def _cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    n = len(y_true)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    po = float(np.trace(cm)) / n
    pe = float(np.dot(cm.sum(axis=1), cm.sum(axis=0))) / (n * n)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
#                              CSV READERS
# ---------------------------------------------------------------------------

def _read_raw_lines(path: Path) -> list[str]:
    """Read every line of a file, stripping whitespace but keeping empty lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_bytes().decode("latin-1", errors="ignore")
    return [ln.strip() for ln in text.splitlines()]


def _read_floats(path: Path) -> np.ndarray:
    """Read all lines, skipping empty / nan labels; invalid → nan."""
    vals = []
    for ln in _read_raw_lines(path):
        if not ln or ln.lower() in ("nan", "-nan(ind)"):
            continue
        try:
            vals.append(float(ln))
        except ValueError:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
#                         REGRESSION METRICS
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    gt_root: Path, pred_root: Path, ds_key: str, tag: str
) -> dict | None:
    """
    For every *.engagement.pred.csv in pred_root/{dataset}/{tag}/,
    find the matching *.engagement.annotation.csv in gt_root/{gt_subdir}/,
    concatenate all frames, and compute CCC / MSE / RMSE / MAE / R².
    """
    gt_ds   = gt_root  / GT_SUBDIR[ds_key]
    pred_ds = pred_root / _pred_folder(ds_key) / tag

    if not pred_ds.exists() or not gt_ds.exists():
        return None

    yt_all: list[np.ndarray] = []
    yp_all: list[np.ndarray] = []

    for sess_dir in sorted(pred_ds.iterdir()):
        if not sess_dir.is_dir():
            continue
        gt_sess = gt_ds / sess_dir.name
        if not gt_sess.is_dir():
            continue

        for pred_file in sorted(sess_dir.glob("*.engagement.pred.csv")):
            gt_name = pred_file.name.replace(".pred.csv", ".annotation.csv")
            gt_file = gt_sess / gt_name
            if not gt_file.exists():
                continue

            yp = _read_floats(pred_file)
            yt = _read_floats(gt_file)
            n  = min(len(yp), len(yt))
            yp, yt = yp[:n], yt[:n]
            mask = np.isfinite(yp) & np.isfinite(yt)
            if mask.any():
                yt_all.append(yt[mask])
                yp_all.append(yp[mask])

    if not yt_all:
        return None

    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)

    mse = float(np.mean((yt - yp) ** 2))
    return {
        "mse":  mse,
        "rmse": float(np.sqrt(mse)),
        "mae":  float(np.mean(np.abs(yt - yp))),
        "ccc":  _ccc(yt, yp),
        "r2":   _r2(yt, yp),
    }


# ---------------------------------------------------------------------------
#                       CLASSIFICATION METRICS
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    gt_root: Path, pred_root: Path, ds_key: str, tag: str
) -> dict | None:
    """
    For every *.pred.csv in pred_root/{dataset}/{tag}/ (excluding regression
    .engagement.pred.csv files), find the matching *.annotation.csv in gt_root,
    accumulate labels, and return {eng_type: {"accuracy": float, "kappa": float}}.
    """
    gt_ds   = gt_root  / GT_SUBDIR[ds_key]
    pred_ds = pred_root / _pred_folder(ds_key) / tag

    if not pred_ds.exists() or not gt_ds.exists():
        return None

    # buckets[eng_type] = ([gt_labels], [pred_labels])
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

    for sess_dir in sorted(pred_ds.iterdir()):
        if not sess_dir.is_dir():
            continue
        gt_sess = gt_ds / sess_dir.name
        if not gt_sess.is_dir():
            continue

        for pred_file in sorted(sess_dir.glob("*.pred.csv")):
            # skip regression files
            if pred_file.name.endswith(".engagement.pred.csv"):
                continue

            gt_name = pred_file.name.replace(".pred.csv", ".annotation.csv")
            gt_file = gt_sess / gt_name
            if not gt_file.exists():
                continue

            # derive engagement type from filename, e.g.:
            #   "purple.social_engagement.pred.csv" -> "social_engagement"
            base  = pred_file.name.removesuffix(".pred.csv")
            parts = base.split(".", 1)
            eng_type = parts[1] if len(parts) == 2 else base

            # Read entire file; strip empty and nan lines from GT upfront so
            # indices align with pred (which was generated for valid GT frames only)
            gt_lines = [
                ln for ln in _read_raw_lines(gt_file)
                if ln and ln.lower() not in ("nan", "-nan(ind)")
            ]
            pred_lines = _read_raw_lines(pred_file)
            n = min(len(gt_lines), len(pred_lines))

            for gt_l, pred_l in zip(gt_lines[:n], pred_lines[:n]):
                buckets[eng_type][0].append(gt_l)
                buckets[eng_type][1].append(pred_l)

    if not buckets:
        return None

    result: dict = {}
    for eng_type, (gt, pred) in sorted(buckets.items()):
        if not gt:
            continue
        acc = float(np.mean([g == p for g, p in zip(gt, pred)]))
        try:
            kappa = _cohen_kappa(gt, pred)
        except Exception as e:
            print(f"    [Warning] kappa failed for {eng_type}: {e}")
            kappa = float("nan")
        result[eng_type] = {"accuracy": acc, "kappa": kappa}

    return result or None


def _discover_clf_heads(gt_root: Path) -> list[str]:
    """Collect all engagement-type names from GT annotation filenames."""
    heads: list[str] = []
    seen:  set[str]  = set()
    for ds_key, _ in CLASSIFICATION_DATASETS:
        gt_ds = gt_root / GT_SUBDIR[ds_key]
        if not gt_ds.is_dir():
            continue
        for sess_dir in sorted(gt_ds.iterdir()):
            if not sess_dir.is_dir():
                continue
            for f in sorted(sess_dir.glob("*.annotation.csv")):
                # skip non-engagement files (gender, age, language, …)
                if f.name.endswith(".engagement.annotation.csv"):
                    continue
                if not any(f.name.endswith(f"{e}.annotation.csv")
                           for e in ("social_engagement", "task_engagement")):
                    # generalise: accept any *_engagement.annotation.csv
                    if "_engagement.annotation.csv" not in f.name:
                        continue
                # strip ".annotation.csv" before splitting, not just .csv
                base  = f.name.removesuffix(".annotation.csv")
                parts = base.split(".", 1)
                eng_type = parts[1] if len(parts) == 2 else base
                if eng_type not in seen:
                    heads.append(eng_type)
                    seen.add(eng_type)
            if seen:
                break  # one session is enough per dataset

    return sorted(heads)


# ---------------------------------------------------------------------------
#                           TABLE RENDERING
# ---------------------------------------------------------------------------

def _mean_finite(values: list) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "-"
    if isinstance(val, float) and math.isnan(val):
        return "nan"
    return f"{val:.{decimals}f}"


def _apply_bold_per_col(rows: list[tuple]) -> list[tuple]:
    if not rows:
        return rows
    n_cols = len(rows[0][2])
    best_val: list = [None] * n_cols
    best_row: list = [-1]  * n_cols
    for i, (_, _, formatted) in enumerate(rows):
        for j, s in enumerate(formatted):
            if s in ("-", "nan"):
                continue
            try:
                v = float(s.strip("*"))
            except ValueError:
                continue
            if best_val[j] is None or v > best_val[j]:
                best_val[j] = v
                best_row[j] = i
    new_rows = []
    for i, (cat, name, formatted) in enumerate(rows):
        new_fmt = list(formatted)
        for j in range(n_cols):
            if best_row[j] == i and new_fmt[j] not in ("-", "nan"):
                new_fmt[j] = f"**{new_fmt[j]}**"
        new_rows.append((cat, name, new_fmt))
    return new_rows


def _render_md_table(col_headers: list[str], rows: list[tuple]) -> str:
    indent = "&nbsp;&nbsp;"
    feat_header = "Feature set"
    feat_w = max(
        len(feat_header),
        max(len(f"*{cat}*") for cat, _, _ in rows),
        max(len(indent + name) for _, name, _ in rows),
    )
    val_ws = [
        max(len(h), max(len(r[2][j]) for r in rows))
        for j, h in enumerate(col_headers)
    ]

    def pad(s: str, w: int) -> str:
        return s.ljust(w)

    header = (
        "| " + pad(feat_header, feat_w)
        + " | " + " | ".join(pad(h, val_ws[j]) for j, h in enumerate(col_headers))
        + " |"
    )
    sep = (
        "| " + "-" * feat_w
        + " | " + " | ".join("-" * w for w in val_ws)
        + " |"
    )
    lines = [header, sep]
    current_cat: str | None = None
    for category, display_name, formatted in rows:
        if category != current_cat:
            current_cat = category
            empty_cells = " | ".join(" " * w for w in val_ws)
            lines.append(
                "| " + pad(f"*{category}*", feat_w) + " | " + empty_cells + " |"
            )
        vals = " | ".join(pad(v, val_ws[j]) for j, v in enumerate(formatted))
        lines.append("| " + pad(indent + display_name, feat_w) + " | " + vals + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#                           TABLE BUILDERS
# ---------------------------------------------------------------------------

def build_regression_table(gt_root: Path, pred_root: Path) -> str:
    col_headers = [label for _, label in REGRESSION_DATASETS] + ["Combined"]
    rows = []
    for mod_key, category, display_name in MODALITY_ORDER:
        tag  = _mod_tag(mod_key)
        vals = []
        for ds_key, _ in REGRESSION_DATASETS:
            m = compute_regression_metrics(gt_root, pred_root, ds_key, tag)
            vals.append(m["ccc"] if m else None)
        combined  = _mean_finite(vals)
        formatted = [_fmt(v) for v in vals + [combined]]
        rows.append((category, display_name, formatted))
    rows = _apply_bold_per_col(rows)
    return _render_md_table(col_headers, rows)


def build_classification_table(
    gt_root: Path, pred_root: Path, metric: str
) -> str:
    head_names = _discover_clf_heads(gt_root)

    col_headers = [
        f"{ds_label} {head.split('_')[0].title()}"
        for _, ds_label in CLASSIFICATION_DATASETS
        for head in head_names
    ] + ["Combined"]

    # Cache per (ds_key, tag) so each pair is computed only once
    clf_cache: dict[tuple[str, str], dict | None] = {}

    rows = []
    for mod_key, category, display_name in MODALITY_ORDER:
        tag  = _mod_tag(mod_key)
        vals = []
        for ds_key, _ in CLASSIFICATION_DATASETS:
            cache_key = (ds_key, tag)
            if cache_key not in clf_cache:
                clf_cache[cache_key] = compute_classification_metrics(
                    gt_root, pred_root, ds_key, tag
                )
            clf = clf_cache[cache_key]
            for head in head_names:
                if clf is None or head not in clf:
                    vals.append(None)
                else:
                    vals.append(clf[head][metric])
        combined  = _mean_finite(vals)
        formatted = [_fmt(v) for v in vals + [combined]]
        rows.append((category, display_name, formatted))
    rows = _apply_bold_per_col(rows)
    return _render_md_table(col_headers, rows)


# ---------------------------------------------------------------------------
#                                  MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute engagement metrics from prediction/GT CSV files "
            "and produce a markdown result table."
        )
    )
    parser.add_argument(
        "--results_dir", default="results/2026",
        metavar="DIR",
        help="Results root; prediction files are read from {results_dir}/test_prediction_files/",
    )
    parser.add_argument(
        "--gt_root",
        default=r"./engagement-mm26-test",
        metavar="DIR",
        help="Root of ground-truth annotation files",
    )
    parser.add_argument(
        "--output", default="../results.md",
        metavar="FILE",
    )
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    gt_root   = Path(args.gt_root)
    pred_root = Path(args.results_dir) / "test_prediction_files"

    print(f"GT root   : {gt_root}")
    print(f"Pred root : {pred_root}")

    for ds_key, ds_label in REGRESSION_DATASETS + CLASSIFICATION_DATASETS:
        gt_path   = gt_root   / GT_SUBDIR[ds_key]
        pred_path = pred_root / _pred_folder(ds_key)
        gt_ok   = "OK"   if gt_path.exists()   else "MISSING"
        pred_ok = "OK"   if pred_path.exists()  else "MISSING"
        print(f"  {ds_label:20s}  GT={gt_ok:7s}  Pred={pred_ok}")

    reg_table   = build_regression_table(gt_root, pred_root)
    kappa_table = build_classification_table(gt_root, pred_root, "kappa")
    acc_table   = build_classification_table(gt_root, pred_root, "accuracy")

    content = "\n".join([
        "# Baseline Results 2026",
        "",
        "## Regression Datasets (CCC)",
        "",
        reg_table,
        "",
        "## Classification Datasets (Cohen's Kappa)",
        "",
        kappa_table,
        "",
        "## Classification Datasets (Accuracy)",
        "",
        acc_table,
        "",
        "---",
        "",
    ])

    output_path = Path(args.output)
    output_path.write_text(content, encoding="utf-8")
    print(f"\nWritten to '{output_path}'")


if __name__ == "__main__":
    main()
