#!/usr/bin/env python3
"""
Parses FINAL_RESULT lines from logs written by 2_RetrainNN_full.py and
summarises per-feature retrain results into a single JSON file.
"""

import json
import re
from pathlib import Path
import argparse


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


if __name__ == "__main__":
    main()
