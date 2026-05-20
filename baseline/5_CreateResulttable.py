#!/usr/bin/env python3
"""
Reads per-dataset test summary files and prints a formatted result table
covering all modalities for both regression (CCC) and classification (Kappa)
datasets.
"""

import re
import math
import argparse
from pathlib import Path


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


def find_latest_summary(dataset_dir: Path) -> Path | None:
    """Return the lexicographically last test_summary_*.txt in full_test_fairness/."""
    fairness_dir = dataset_dir / "full_test_fairness"
    if not fairness_dir.exists():
        return None
    files = sorted(fairness_dir.glob("test_summary_*.txt"))
    return files[-1] if files else None


def _to_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def parse_summary(path: Path) -> dict:
    """
    Parse a test_summary_*.txt file.

    Returns:
      {
        "task": "regression" | "classification",
        "data": {
            modality: {"mse", "rmse", "mae", "ccc", "r2"}      # regression
          | modality: {head_name: {"accuracy", "f1"}}           # classification
        }
      }
    """
    text = path.read_text(encoding="utf-8")
    # Split on modality block headers: === modality.name (N-D) ===
    parts = re.split(r"===\s+(.+?)\s+\(\d+-D\)\s+===", text)
    # parts[0] = preamble, then alternating: modality_name, block_content

    task: str | None = None
    data: dict = {}

    for i in range(1, len(parts), 2):
        modality = parts[i].strip()
        content  = parts[i + 1] if i + 1 < len(parts) else ""

        # Detect task type from first block
        if task is None:
            task = "classification" if re.search(r"Accuracy:", content) else "regression"

        if task == "regression":
            m = re.search(
                r"MSE\s+(\S+)\s+RMSE\s+(\S+)\s+MAE\s+(\S+)\s+CCC\s+(\S+)\s+R2\s+(\S+)",
                content,
            )
            if m:
                data[modality] = {
                    "mse":  _to_float(m.group(1)),
                    "rmse": _to_float(m.group(2)),
                    "mae":  _to_float(m.group(3)),
                    "ccc":  _to_float(m.group(4)),
                    "r2":   _to_float(m.group(5)),
                }
        else:
            # [head_name] Accuracy: X  Cohen's Kappa: Y
            heads = re.findall(
                r"\[(\w+)\]\s+Accuracy:\s+(\S+)\s+Cohen's Kappa:\s+(\S+)", content
            )
            if heads:
                data[modality] = {
                    head: {"accuracy": float(acc), "kappa": float(kappa)}
                    for head, acc, kappa in heads
                }

    return {"task": task or "regression", "data": data}


def _mean_finite(values: list) -> float:
    """Mean over values that are not None and not nan."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "-"
    if isinstance(val, float) and math.isnan(val):
        return "nan"
    return f"{val:.{decimals}f}"


def _apply_bold_per_col(rows: list[tuple]) -> list[tuple]:
    """
    Bold the maximum finite value in each column across all rows.
    Returns a new list of rows with bold markers applied.
    """
    if not rows:
        return rows
    n_cols = len(rows[0][2])
    best_val: list = [None] * n_cols
    best_row: list = [-1] * n_cols

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
    """
    Render a GitHub-flavored markdown table.
    """
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
            lines.append("| " + pad(f"*{category}*", feat_w) + " | " + empty_cells + " |")
        vals = " | ".join(pad(v, val_ws[j]) for j, v in enumerate(formatted))
        lines.append("| " + pad(indent + display_name, feat_w) + " | " + vals + " |")

    return "\n".join(lines)

def build_regression_table(parsed: dict[str, dict]) -> str:
    """CCC table across regression datasets."""
    col_headers = [label for _, label in REGRESSION_DATASETS] + ["Combined"]

    rows = []
    for mod_key, category, display_name in MODALITY_ORDER:
        vals = []
        for ds_key, _ in REGRESSION_DATASETS:
            result = parsed.get(ds_key)
            if result is None:
                vals.append(None)
            else:
                entry = result["data"].get(mod_key)
                vals.append(entry["ccc"] if entry else None)

        combined = _mean_finite(vals)
        formatted = [_fmt(v) for v in vals + [combined]]
        rows.append((category, display_name, formatted))

    rows = _apply_bold_per_col(rows)
    return _render_md_table(col_headers, rows)


def _discover_heads(parsed: dict[str, dict]) -> list[str]:
    """Return classification head names in order of first appearance."""
    heads: list[str] = []
    seen: set[str] = set()
    for ds_key, _ in CLASSIFICATION_DATASETS:
        result = parsed.get(ds_key)
        if not result:
            continue
        for mod_data in result["data"].values():
            for head in mod_data:
                if head not in seen:
                    heads.append(head)
                    seen.add(head)
    return heads


def build_classification_table(parsed: dict[str, dict], metric: str) -> str:
    """Cohen's Kappa or Accuracy table across classification datasets."""
    head_names = _discover_heads(parsed)

    col_headers = [
        f"{ds_label} {head.split('_')[0].title()}"
        for ds_key, ds_label in CLASSIFICATION_DATASETS
        for head in head_names
    ] + ["Combined"]

    rows = []
    for mod_key, category, display_name in MODALITY_ORDER:
        vals = []
        for ds_key, _ in CLASSIFICATION_DATASETS:
            result = parsed.get(ds_key)
            for head in head_names:
                if result is None:
                    vals.append(None)
                else:
                    entry = result["data"].get(mod_key)
                    vals.append(entry[head][metric] if (entry and head in entry) else None)

        combined = _mean_finite(vals)
        formatted = [_fmt(v) for v in vals + [combined]]
        rows.append((category, display_name, formatted))

    rows = _apply_bold_per_col(rows)
    return _render_md_table(col_headers, rows)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", default="results/2026",
        metavar="DIR",
    )
    parser.add_argument(
        "--output", default="../results.md",
        metavar="FILE",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_dir)

    # --- Load summaries ---
    reg_parsed: dict[str, dict] = {}
    clf_parsed: dict[str, dict] = {}

    print(f"Reading summaries from '{results_root}' ...")
    for ds_key, _ in REGRESSION_DATASETS:
        path = find_latest_summary(results_root / ds_key)
        if path is None:
            print(f"  Warning: no test_summary found for '{ds_key}', skipping")
            continue
        print(f"  {path.relative_to(results_root)}")
        reg_parsed[ds_key] = parse_summary(path)

    for ds_key, _ in CLASSIFICATION_DATASETS:
        path = find_latest_summary(results_root / ds_key)
        if path is None:
            print(f"  Warning: no test_summary found for '{ds_key}', skipping")
            continue
        print(f"  {path.relative_to(results_root)}")
        clf_parsed[ds_key] = parse_summary(path)

    # --- Build tables ---
    reg_table   = build_regression_table(reg_parsed)
    kappa_table = build_classification_table(clf_parsed, "kappa")
    acc_table   = build_classification_table(clf_parsed, "accuracy")

    # --- Write output ---
    output_path = Path(args.output)
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
    output_path.write_text(content, encoding="utf-8")
    print(f"\nWritten to '{output_path}'")


if __name__ == "__main__":
    main()
