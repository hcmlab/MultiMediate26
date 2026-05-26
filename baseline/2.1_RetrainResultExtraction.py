#!/usr/bin/env python3
"""
Parses FINAL_RESULT lines from logs written by 2_RetrainNN_full.py and
summarises per-feature retrain results into a single JSON file and a
markdown table file (results_val.md) using the same format as 5_combine_results.py.
"""

import json
import math
import re
from pathlib import Path
import argparse


# ---------------------------------------------------------------------------
# Markdown table configuration (mirrors 5_combine_results.py)
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


# ---------------------------------------------------------------------------
# Markdown helpers (mirrors 5_combine_results.py)
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
    head_names = _discover_heads(parsed)

    col_headers = [
        f"{ds_label} {head.split('_')[0].title()}"
        for _, ds_label in CLASSIFICATION_DATASETS
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


# ---------------------------------------------------------------------------
# Convert retrain_results dict → parsed format for table builders
# ---------------------------------------------------------------------------

def _normalize_modality(key: str) -> str:
    """Strip leading '.' and trailing '~' from raw retrain modality keys."""
    return key.strip(".~")


def results_to_parsed(results: dict[str, dict]) -> tuple[dict, dict]:
    """
    Convert {dataset: {raw_modality: metrics}} from retrain_results.json into
    the parsed dicts expected by build_regression_table / build_classification_table.

    Returns (reg_parsed, clf_parsed).
    """
    reg_parsed: dict[str, dict] = {}
    clf_parsed: dict[str, dict] = {}

    for dataset, modalities in results.items():
        data: dict = {}
        task: str | None = None

        for raw_mod, metrics in modalities.items():
            mod_key = _normalize_modality(raw_mod)

            if "kappa_per_head" in metrics:
                task = "classification"
                heads = {}
                for head, kappa in metrics.get("kappa_per_head", {}).items():
                    acc = metrics.get("accuracy_per_head", {}).get(head)
                    heads[head] = {"accuracy": acc, "kappa": kappa}
                data[mod_key] = heads
            else:
                task = "regression"
                data[mod_key] = {k: metrics[k] for k in ("ccc", "mse", "mae", "rmse", "r2") if k in metrics}

        parsed = {"task": task or "regression", "data": data}
        if task == "classification":
            clf_parsed[dataset] = parsed
        else:
            reg_parsed[dataset] = parsed

    return reg_parsed, clf_parsed


def write_markdown(results: dict[str, dict], output_path: str) -> None:
    reg_parsed, clf_parsed = results_to_parsed(results)

    reg_table    = build_regression_table(reg_parsed)
    kappa_table  = build_classification_table(clf_parsed, "kappa")
    acc_table    = build_classification_table(clf_parsed, "accuracy")

    content = "\n".join([
        "# Validation Results (Retrain)",
        "",
        "## Regression Datasets (CCC)",
        "",
        "Combined is the mean over available (non-nan) datasets per feature set.",
        "",
        reg_table,
        "",
        "## Classification Datasets (Cohen's Kappa)",
        "",
        "Combined is the mean over all columns.",
        "",
        kappa_table,
        "",
        "## Classification Datasets (Accuracy)",
        "",
        "Combined is the mean over all columns.",
        "",
        acc_table,
        "",
    ])

    Path(output_path).write_text(content, encoding="utf-8")
    print(f"Written to '{output_path}'")


_LOG_TO_CONFIG_DATASET: dict[str, str] = {
    "noxi": "noxi-base",
}


def _config_dataset(dataset: str) -> str:
    """Translate a log-directory dataset name to the configs/ directory name."""
    return _LOG_TO_CONFIG_DATASET.get(dataset, dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise per-feature retrain results from logs written by 2_RetrainNN_full.py"
    )
    parser.add_argument(
        "--configs_dir", type=str, default="configs",
        metavar="DIR",
        help="Root configs directory containing {dataset}/retrain/ subdirs (default: configs)"
    )
    parser.add_argument(
        "--slurm_dir", type=str, default=".",
        metavar="DIR",
        help="Directory used to resolve modality order via full_train/ configs (default: .)"
    )
    parser.add_argument(
        "--nn_logs_dir", type=str, default="./nn/logs",
        metavar="DIR",
        help="Root log directory written by 2_RetrainNN_full.py (default: ./nn/logs)"
    )
    parser.add_argument(
        "--output", type=str, default="retrain_results.json",
        metavar="FILE",
        help="Output JSON file (default: retrain_results.json)"
    )
    parser.add_argument(
        "--output_md", type=str, default="results_val.md",
        metavar="FILE",
        help="Output markdown file (default: results_val.md)"
    )
    return parser.parse_args()


def get_modality_order(slurm_dir: str, dataset: str) -> list[str]:
    full_train_dir = Path(slurm_dir) / "configs" / _config_dataset(dataset) / "full_train"
    if not full_train_dir.exists():
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    def config_num(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    for cfg_file in sorted(full_train_dir.glob("config*.json"), key=config_num):
        try:
            with open(cfg_file, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  Warning: cannot read '{cfg_file}': {exc}")
            continue
        for mod in cfg.get("modalities", []):
            if mod not in seen:
                ordered.append(mod)
                seen.add(mod)
    return ordered



def evaluate_all(configs_dir: str, slurm_dir: str, nn_logs_dir: str) -> dict[str, dict]:
    """
    For each dataset/retrain/*.json config, find the most recent log written by
    2_RetrainNN_full.py and parse the FINAL_RESULT line.
    Returns {dataset: {modality: {metric: value, ...}}}.
    """
    results: dict[str, dict] = {}
    configs_path = Path(configs_dir)
    logs_path = Path(nn_logs_dir)

    for dataset_dir in sorted(configs_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        retrain_dir = dataset_dir / "retrain"
        if not retrain_dir.exists():
            continue

        dataset = dataset_dir.name
        print(f"\nDataset: {dataset}")

        config_order = get_modality_order(slurm_dir, dataset)
        feature_results: dict[str, dict] = {}

        for cfg_file in sorted(retrain_dir.glob("*.json")):
            try:
                with open(cfg_file, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  Warning: cannot read '{cfg_file}': {exc}")
                continue

            modality = cfg["modalities"][0]
            modality_clean = modality.strip(".~")

            # Find log files written by 2_RetrainNN_full.py 
            log_dir = logs_path / dataset / modality_clean / "retrain"
            log_files = sorted(log_dir.glob("*/retrain_log_*.txt")) if log_dir.exists() else []

            if not log_files:
                print(f"  Skipping {modality}: no log files found at {log_dir}")
                continue

            # Use the most recent log file
            log_file = log_files[-1]

            # Parse the last FINAL_RESULT line written by 2_RetrainNN_full.py
            entry: dict | None = None
            try:
                with open(log_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("FINAL_RESULT:"):
                            entry = json.loads(line[len("FINAL_RESULT:"):].strip())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  Warning: cannot parse log '{log_file}': {exc}")
                continue

            if entry is None:
                print(f"  Skipping {modality}: no FINAL_RESULT found in {log_file}")
                continue

            feature_results[modality] = entry
            ccc = entry.get("ccc")
            mse = entry.get("mse")
            kappa = entry.get("kappa")
            ccc_str = f"  CCC={ccc:.4f}" if ccc is not None else ""
            mse_str = f"  MSE={mse:.4f}" if mse is not None else ""
            kappa_str = f"  Kappa={kappa:.4f}" if kappa is not None else ""
            print(f"  {modality}{ccc_str}{mse_str}{kappa_str}")

        sorted_mods = [m for m in config_order if m in feature_results]
        sorted_mods += sorted(m for m in feature_results if m not in set(config_order))
        results[dataset] = {m: feature_results[m] for m in sorted_mods}

    return results


def main():
    args = parse_args()

    results = evaluate_all(args.configs_dir, args.slurm_dir, args.nn_logs_dir)

    print(f"\nWriting '{args.output}' ...")
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nDone. Results for {len(results)} dataset(s):")
    for dataset, modalities in results.items():
        print(f"  {dataset}  ({len(modalities)} modality/ies)")
        for mod, metrics in modalities.items():
            ccc = metrics.get("ccc")
            mse = metrics.get("mse")
            kappa = metrics.get("kappa")
            ccc_str = f"  CCC={ccc:.4f}" if ccc is not None else ""
            mse_str = f"  MSE={mse:.4f}" if mse is not None else ""
            kappa_str = f"  Kappa={kappa:.4f}" if kappa is not None else ""
            print(f"    {mod}{ccc_str}{mse_str}{kappa_str}")

    write_markdown(results, args.output_md)


if __name__ == "__main__":
    main()
