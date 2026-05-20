#!/usr/bin/env python3
"""
Extracts best hyperparameters from completed tuning runs and writes them to a
combined JSON file. Also generates per-feature retrain configs for use with
2_RetrainNN_full.py. Expects log directories with structure
{dataset}/{modality}/{job_id}/best_hyperparameters.json.
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict


HYPERPARAMS_KEYS = {"units1", "units2", "units3", "dropout", "learning_rate", "batch_size"}

_LOG_TO_CONFIG_DATASET: dict[str, str] = {
    "noxi": "noxi-base",
}


def _config_dataset(dataset: str) -> str:
    """Translate a log-directory dataset name to the configs/ directory name."""
    return _LOG_TO_CONFIG_DATASET.get(dataset, dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine best_hyperparameters.json files from all tuning runs"
    )
    parser.add_argument(
        "--logs_dirs", nargs="+", default=["nn/logs"],
        metavar="DIR",
        help="One or more root log directories with structure "
             "{dataset}/{modality}/{job_id}/ (default: nn/logs)"
    )
    parser.add_argument(
        "--slurm_dir", type=str, default=".",
        metavar="DIR",
        help="Directory containing *.slurm scripts used to determine job_name→dataset mapping "
             "(default: current directory)"
    )
    parser.add_argument(
        "--output", type=str, default="combined_hyperparameters.json",
        metavar="FILE",
        help="Output JSON file (default: combined_hyperparameters.json)"
    )
    parser.add_argument(
        "--configs_dir", type=str, default="configs",
        metavar="DIR",
        help="Root configs directory for writing per-feature retrain configs (default: configs)"
    )
    return parser.parse_args()


def get_job_name_to_dataset(slurm_dir: str) -> dict[str, str]:
    """
    Build a job_name -> dataset_name mapping by combining information from two sources:

    1. *.slurm scripts: provide #SBATCH --job-name and, when present, an inline
       configs/<dataset>/ path (e.g. in usage comments).
    2. run_all*.sh scripts: provide sbatch lines of the form
         sbatch <slurm_file> configs/<dataset>/...
       which link a slurm file (and therefore its job name) to a dataset.

    Returns {job_name: dataset_name}.
    """
    slurm_path = Path(slurm_dir)

    slurm_info: dict[str, tuple[str, str | None]] = {}
    for slurm_file in sorted(slurm_path.glob("*.slurm")):
        try:
            content = slurm_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  Warning: cannot read '{slurm_file}': {exc}")
            continue

        job_match = re.search(r"#SBATCH\s+--job-name=(\S+)", content)
        if not job_match:
            continue
        job_name = job_match.group(1)

        cfg_match = re.search(r"\bconfigs/([^/\s]+)/", content)
        dataset = cfg_match.group(1) if cfg_match else None
        slurm_info[slurm_file.name] = (job_name, dataset)

    sbatch_re = re.compile(r"\bsbatch\s+(\S+\.slurm)\s+[^\n]*\bconfigs/([^/\s]+)/")
    for sh_file in sorted(slurm_path.glob("run_all*.sh")):
        try:
            content = sh_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  Warning: cannot read '{sh_file}': {exc}")
            continue

        for m in sbatch_re.finditer(content):
            slurm_fname = Path(m.group(1)).name
            dataset = m.group(2)
            if slurm_fname in slurm_info:
                job_name, _ = slurm_info[slurm_fname]
                slurm_info[slurm_fname] = (job_name, dataset)

    mapping: dict[str, str] = {}
    for job_name, dataset in slurm_info.values():
        if dataset:
            mapping[job_name] = dataset

    return mapping


def get_job_id_to_job_name(logs_dir: Path) -> dict[str, str]:
    """
    Parse SLURM log filenames in logs_dir.

    Filename pattern:  {job_name}-{job_id}-{array_id}.log
    Returns {job_id: job_name}.
    """
    mapping: dict[str, str] = {}
    pattern = re.compile(r"^(.+)-(\d+)-\d+\.log$")

    try:
        entries = list(logs_dir.iterdir())
    except OSError as exc:
        print(f"  Warning: cannot list '{logs_dir}': {exc}")
        return mapping

    for entry in entries:
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if m:
            job_name, job_id = m.group(1), m.group(2)
            mapping[job_id] = job_name

    return mapping


def get_modality_order(slurm_dir: str, dataset: str) -> list[str]:
    """
    Read full_train configs (config1.json, config2.json, ...) for a dataset in
    numeric order and return the modalities in that order.

    Configs are expected at: {slurm_dir}/configs/{dataset}/full_train/config*.json
    """
    full_train_dir = Path(slurm_dir) / "configs" / _config_dataset(dataset) / "full_train"
    if not full_train_dir.exists():
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    # Sort config files numerically by the number in their name
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

        for modality in cfg.get("modalities", []):
            if modality not in seen:
                ordered.append(modality)
                seen.add(modality)

    return ordered


def apply_modality_order(
    combined: dict[str, dict],
    slurm_dir: str,
) -> dict[str, dict]:
    """
    Return a new combined dict where each dataset's modalities are ordered
    according to the tune configs (config1, config2, config3, ...).
    Modalities not found in any config are appended at the end.
    """
    ordered: dict[str, dict] = {}
    for dataset, modalities in combined.items():
        config_order = get_modality_order(slurm_dir, dataset)
        # Start with modalities that appear in the config order
        sorted_mods = [m for m in config_order if m in modalities]
        # Append any extras not covered by configs
        extras = [m for m in modalities if m not in set(config_order)]
        sorted_mods.extend(sorted(extras))
        ordered[dataset] = {m: modalities[m] for m in sorted_mods}
    return ordered


def get_modality_metadata(slurm_dir: str, dataset: str) -> dict[str, dict]:
    """
    Read full_train configs for a dataset and return per-modality metadata.

    Returns {modality: {'dim': int, 'train_dir': str, 'val_dir': str, 'full_base_dir': str}}
    """
    full_train_dir = Path(slurm_dir) / "configs" / _config_dataset(dataset) / "full_train"
    if not full_train_dir.exists():
        return {}

    metadata: dict[str, dict] = {}

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

        modalities = cfg.get("modalities", [])
        dims = cfg.get("modalities_dim", [])
        train_dir = cfg.get("train_dir", "")
        val_dir = cfg.get("val_dir", "")
        full_base_dir = cfg.get("full_base_dir", "./results/full_models")

        for i, modality in enumerate(modalities):
            if modality not in metadata:
                metadata[modality] = {
                    "dim": dims[i] if i < len(dims) else None,
                    "train_dir": train_dir,
                    "val_dir": val_dir,
                    "full_base_dir": full_base_dir,
                }

    return metadata


def write_feature_configs(
    combined: dict[str, dict],
    slurm_dir: str,
    configs_base_dir: str,
) -> None:
    """
    For each dataset+modality in combined, write a single-modality JSON config to
    {configs_base_dir}/{config_dataset}/retrain/{modality_clean}.json

    These per-feature configs can be passed directly to 2_RetrainNN_full.py.
    """
    print(f"\nWriting per-feature retrain configs to '{configs_base_dir}/' ...")
    for dataset, modalities in combined.items():
        cfg_dataset = _config_dataset(dataset)
        metadata = get_modality_metadata(slurm_dir, dataset)
        retrain_dir = Path(configs_base_dir) / cfg_dataset / "retrain"
        retrain_dir.mkdir(parents=True, exist_ok=True)

        for modality, hps in modalities.items():
            meta = metadata.get(modality, {})
            modality_clean = modality.strip(".~")
            cfg = {
                "modalities": [modality],
                "modalities_dim": [meta.get("dim")],
                "train_dir": meta.get("train_dir", ""),
                "val_dir": meta.get("val_dir", ""),
                "full_base_dir": meta.get("full_base_dir", "./results/full_models"),
                "best_hyperparameters": {modality: hps},
            }
            out_path = retrain_dir / f"{modality_clean}.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"  {out_path}")


def modality_from_dirname(dirname: str) -> str:
    """
    Convert a result directory name to a config-style modality key.

    'audio.egemapsv2.stream_20260507_063316'  →  '.audio.egemapsv2.stream~'
    """
    base = re.sub(r"_\d{8}_\d{6}$", "", dirname)
    return f".{base}~"


def timestamp_from_dirname(dirname: str) -> str:
    """Return the 'YYYYMMDD_HHMMSS' timestamp from a result directory name, or ''."""
    m = re.search(r"_(\d{8}_\d{6})$", dirname)
    return m.group(1) if m else ""


def filter_params(raw: dict) -> dict:
    """Keep only the model hyperparameters (drop tuner/* bookkeeping fields)."""
    return {k: v for k, v in raw.items() if k in HYPERPARAMS_KEYS}


def collect_all(
    logs_dirs: list[str],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    """
    Walk each logs_dir which has structure {dataset}/{modality}/{job_id}/.

    Dataset and modality are taken directly from directory names.
    When multiple job_ids exist for the same (dataset, modality), the highest
    (newest) job_id wins.

    Returns:
        combined     -- {dataset: {modality: {param: value, ...}}}
        ccc_results  -- {dataset: {modality: float}}
        mse_results  -- {dataset: {modality: float}}
        acc_results  -- {dataset: {modality: float}}  (mean across heads)
    """
    result: dict[str, dict] = defaultdict(dict)
    ccc_results: dict[str, dict] = defaultdict(dict)
    mse_results: dict[str, dict] = defaultdict(dict)
    acc_results: dict[str, dict] = defaultdict(dict)
    # best_job_id[dataset][modality] = highest job_id seen so far (int)
    best_job_id: dict[str, dict] = defaultdict(dict)

    for logs_dir_str in logs_dirs:
        logs_dir = Path(logs_dir_str)
        print(f"\nScanning '{logs_dir}' ...")

        if not logs_dir.exists():
            print(f"  Warning: directory does not exist, skipping")
            continue

        # Iterate dataset dirs
        try:
            dataset_dirs = sorted(logs_dir.iterdir())
        except OSError as exc:
            print(f"  Warning: cannot list directory: {exc}")
            continue

        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            # Iterate modality dirs
            try:
                modality_dirs = sorted(dataset_dir.iterdir())
            except OSError as exc:
                print(f"  Warning: cannot list '{dataset_dir}': {exc}")
                continue

            for mod_dir in modality_dirs:
                if not mod_dir.is_dir():
                    continue
                modality = modality_from_dirname(mod_dir.name)

                # Iterate job_id dirs, pick the highest numeric job_id with results
                try:
                    job_dirs = sorted(mod_dir.iterdir())
                except OSError as exc:
                    print(f"  Warning: cannot list '{mod_dir}': {exc}")
                    continue

                for job_dir in job_dirs:
                    if not job_dir.is_dir():
                        continue
                    if not job_dir.name.isdigit():
                        continue
                    job_id = int(job_dir.name)

                    hp_file = job_dir / "best_hyperparameters.json"
                    if not hp_file.exists():
                        continue

                    # Keep only the highest job_id for each (dataset, modality)
                    if job_id <= best_job_id[dataset].get(modality, -1):
                        print(f"  Skipping older job {job_dir.name}: "
                              f"{dataset} / {modality}")
                        continue

                    try:
                        with open(hp_file, "r", encoding="utf-8") as fh:
                            raw = json.load(fh)
                    except (OSError, json.JSONDecodeError) as exc:
                        print(f"  Warning: cannot read '{hp_file}': {exc}")
                        continue

                    result[dataset][modality] = filter_params(raw)
                    best_job_id[dataset][modality] = job_id

                    ccc_file = job_dir / "final_ccc.txt"
                    if ccc_file.exists():
                        try:
                            ccc_results[dataset][modality] = float(
                                ccc_file.read_text(encoding="utf-8").strip()
                            )
                        except (OSError, ValueError) as exc:
                            print(f"  Warning: cannot read CCC from '{ccc_file}': {exc}")

                    mse_file = job_dir / "final_mse.txt"
                    if mse_file.exists():
                        try:
                            mse_results[dataset][modality] = float(
                                mse_file.read_text(encoding="utf-8").strip()
                            )
                        except (OSError, ValueError) as exc:
                            print(f"  Warning: cannot read MSE from '{mse_file}': {exc}")

                    acc_file = job_dir / "final_accuracy.txt"
                    if acc_file.exists():
                        try:
                            vals = [
                                float(v)
                                for v in acc_file.read_text(encoding="utf-8").splitlines()
                                if v.strip()
                            ]
                            if vals:
                                acc_results[dataset][modality] = sum(vals) / len(vals)
                        except (OSError, ValueError) as exc:
                            print(f"  Warning: cannot read accuracy from '{acc_file}': {exc}")

                    print(f"  Loaded  {dataset} / {modality}  (job {job_dir.name})")

    return dict(result), dict(ccc_results), dict(mse_results), dict(acc_results)


def main():
    args = parse_args()

    combined, ccc_results, mse_results, acc_results = collect_all(args.logs_dirs)
    combined = apply_modality_order(combined, args.slurm_dir)

    print(f"\nWriting '{args.output}' ...")
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)

    write_feature_configs(combined, args.slurm_dir, args.configs_dir)

    print(f"\nDone. Results for {len(combined)} dataset(s):")
    for dataset, modalities in combined.items():
        print(f"  {dataset}  ({len(modalities)} modality/ies)")
        for mod in modalities:
            ccc = ccc_results.get(dataset, {}).get(mod)
            mse = mse_results.get(dataset, {}).get(mod)
            acc = acc_results.get(dataset, {}).get(mod)
            ccc_str = f"  CCC={ccc:.4f}" if ccc is not None else ""
            mse_str = f"  MSE={mse:.4f}" if mse is not None else ""
            acc_str = f"  Acc={acc:.4f}" if acc is not None else ""
            print(f"    {mod}{ccc_str}{mse_str}{acc_str}")


if __name__ == "__main__":
    main()
