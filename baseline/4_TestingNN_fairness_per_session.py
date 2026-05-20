#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, cohen_kappa_score)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#                               FAIRNESS METRICS
# ---------------------------------------------------------------------------

def fairness_gender(y: np.ndarray, y_pred: np.ndarray, gender: np.ndarray, n_bins: int = 10) -> float:
    """CDD_G – mean difference M(pred|male) − M(pred|female) per y‑bin."""
    y, y_pred, gender = map(np.asarray, (y, y_pred, gender))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    y_binned = np.digitize(y, bin_edges, right=False) - 1
    diffs: list[float] = []
    for b in range(n_bins):
        mask_m = (y_binned == b) & (gender == 1)
        mask_f = (y_binned == b) & (gender == 0)
        if mask_m.sum() == 0 or mask_f.sum() == 0:
            continue
        diffs.append(y_pred[mask_m].mean() - y_pred[mask_f].mean())
    return float(np.mean(diffs)) if diffs else 0.0


def fairness_language(
    y: np.ndarray, y_pred: np.ndarray, language: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """CDD_L – per‑language mean difference between language bin mean and overall bin mean."""
    y, y_pred, language = map(np.asarray, (y, y_pred, language))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    y_binned = np.digitize(y, bin_edges, right=False) - 1
    cdd_sum: dict[str, list[float]] = {}
    for b in range(n_bins):
        bin_mask = y_binned == b
        if not bin_mask.any():
            continue
        overall_mean = y_pred[bin_mask].mean()
        for lang in np.unique(language[bin_mask]):
            lang_mask = bin_mask & (language == lang)
            if lang_mask.sum() == 0:
                continue
            diff = y_pred[lang_mask].mean() - overall_mean
            cdd_sum.setdefault(lang, []).append(diff)
    return {lang: float(np.mean(vals)) for lang, vals in cdd_sum.items()}


# ---------------------------------------------------------------------------
#                           UTILITY / METRIC HELPERS
# ---------------------------------------------------------------------------

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same implementation as the *old* script – no extra safeguards."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if df.empty:
        return float("nan")
    yt, yp = df["y_true"].values, df["y_pred"].values
    cor = np.corrcoef(yt, yp)[0, 1]
    mt, mp = np.mean(yt), np.mean(yp)
    vt, vp = np.var(yt), np.var(yp)
    sdt, sdp = np.std(yt), np.std(yp)
    return float((2 * cor * sdt * sdp) / (vt + vp + (mt - mp) ** 2 + 1e-12))


def _ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


# ---------------------------------------------------------------------------
#                              DATA  LOADERS
# ---------------------------------------------------------------------------

def _parse_language(path: str | None) -> str | None:
    if path is None or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().strip().split(";")
    return first[2].strip() if len(first) >= 3 else None


def _parse_gender(path: str | None, n: int) -> np.ndarray:
    arr = np.full(n, np.nan, dtype=np.float32)
    if path is None or not os.path.exists(path):
        return arr
    vals: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.strip().split(";")
            if len(parts) < 3:
                continue
            try:
                g = int(parts[2])
                vals.append(0 if g == 2 else 1)  # 2→female(0), 1→male(1)
            except ValueError:
                vals.append(np.nan)
    if len(vals) < n:
        vals += [np.nan] * (n - len(vals))
    return np.array(vals[:n], dtype=np.float32)


def load_data_with_sessions(root_dirs: List[str], modality: str, feat_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate samples across *root_dirs* (train/val) – identical to old script."""
    X, y, sess = [], [], []
    for root in _ensure_list(root_dirs):
        for dirpath, _, files in os.walk(root):
            session_id = os.path.basename(dirpath)
            stream_map, anno_map = {}, {}
            for fn in files:
                key = f"{fn.split('.')[0]};{session_id}"
                fpath = os.path.join(dirpath, fn)
                if modality in fn:
                    stream_map[key] = fpath
                elif fn.endswith(".engagement.annotation.csv"):
                    anno_map[key] = fpath
            for key, anno_path in anno_map.items():
                feat_path = stream_map.get(key)
                if not feat_path:
                    continue
                try:
                    a = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)
                    with open(anno_path, encoding="utf-8") as fh:
                        annos = [ln.strip() for ln in fh]
                except UnicodeDecodeError:
                    with open(anno_path, encoding="latin1", errors="ignore") as fh:
                        annos = [ln.strip() for ln in fh]
                n = min(len(a), len(annos))
                for i in range(n):
                    v = annos[i]
                    if v in ("", "nan", "-nan(ind)"):
                        continue
                    try:
                        yval = float(v)
                    except ValueError:
                        continue
                    X.append(a[i])
                    y.append(yval)
                    sess.append(session_id)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(sess)


def load_test_data_with_gt_multi(
    feat_root: str | List[str],
    gt_root: str | List[str],
    modality: str,
    feat_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load test samples plus language & gender annotations."""
    feat_roots, gt_roots = map(_ensure_list, (feat_root, gt_root))
    X, y, sess, lang, gender = [], [], [], [], []
    for froot in feat_roots:
        for groot in gt_roots:
            sessions = [s for s in os.listdir(froot) if os.path.isdir(os.path.join(froot, s))]
            for sess_id in sessions:
                feat_dir, gt_dir = os.path.join(froot, sess_id), os.path.join(groot, sess_id)
                if not os.path.isdir(gt_dir):
                    continue
                language = _parse_language(os.path.join(gt_dir, "language.annotation.csv")) or "Unknown"
                for fn in os.listdir(feat_dir):
                    if modality not in fn:
                        continue
                    role = fn.split(".")[0]
                    feat_path = os.path.join(feat_dir, fn)
                    eng_path = os.path.join(gt_dir, f"{role}.engagement.annotation.csv")
                    gen_path = os.path.join(gt_dir, f"{role}.gender.annotation.csv")
                    if not os.path.exists(eng_path):
                        continue
                    try:
                        a = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)
                        with open(eng_path, encoding="utf-8") as fh:
                            annos = [ln.strip() for ln in fh]
                    except UnicodeDecodeError:
                        with open(eng_path, encoding="latin1", errors="ignore") as fh:
                            annos = [ln.strip() for ln in fh]
                    n = min(len(a), len(annos))
                    g_arr = _parse_gender(gen_path, n)
                    for i in range(n):
                        v = annos[i]
                        if v in ("", "nan", "-nan(ind)"):
                            continue
                        try:
                            yval = float(v)
                        except ValueError:
                            continue
                        X.append(a[i])
                        y.append(yval)
                        sess.append(sess_id)
                        lang.append(language)
                        gender.append(g_arr[i])
    return (
        np.asarray(X, np.float32),
        np.asarray(y, np.float32),
        np.asarray(sess),
        np.asarray(lang),
        np.asarray(gender, np.float32),
    )

def save_sessionwise_predictions_testgt(
    feat_roots: List[str],
    gt_roots: List[str],
    modality: str,
    feat_dim: int,
    scaler: MinMaxScaler,
    model: tf.keras.Model,
    results_dir: str,
):
    """
    For each session/role, predicts and saves to:
    results_dir/session/role.engagement.annotation.csv, using test feature folders and ground truth annotation.
    """
    from collections import defaultdict
    feat_roots, gt_roots = map(_ensure_list, (feat_roots, gt_roots))

    for froot in feat_roots:
        for groot in gt_roots:
            sessions = [s for s in os.listdir(froot) if os.path.isdir(os.path.join(froot, s))]
            for sess_id in sessions:
                feat_dir = os.path.join(froot, sess_id)
                gt_dir = os.path.join(groot, sess_id)
                if not os.path.isdir(gt_dir):
                    continue
                # Find all feature files for this modality in feat_dir
                for fn in os.listdir(feat_dir):
                    if modality not in fn:
                        continue
                    role = fn.split(".")[0]
                    feat_path = os.path.join(feat_dir, fn)
                    anno_path = os.path.join(gt_dir, f"{role}.engagement.annotation.csv")
                    if not os.path.exists(anno_path):
                        continue
                    try:
                        a = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)
                        with open(anno_path, encoding="utf-8") as fh:
                            annos = [ln.strip() for ln in fh]
                    except Exception as e:
                        print(f"  Error loading {feat_path} or {anno_path}: {e}")
                        continue
                    n = min(len(a), len(annos))
                    valid_idxs = [
                        i for i, v in enumerate(annos[:n]) if v not in ("", "nan", "-nan(ind)")
                    ]
                    if not valid_idxs:
                        continue
                    X_valid = a[valid_idxs]
                    X_valid_scaled = scaler.transform(X_valid)
                    y_pred = model.predict(X_valid_scaled, batch_size=128, verbose=0).squeeze()
                    out_dir = os.path.join(results_dir, sess_id)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{role}.engagement.annotation.csv")
                    np.savetxt(out_path, y_pred, fmt="%.6f")

# ---------------------------------------------------------------------------
#              TEST PREDICTION FILE SAVERS (annotation-mirrored)
# ---------------------------------------------------------------------------

def save_test_predictions_regression(feat_roots, gt_roots, modality, feat_dim, scaler, model, out_dir,
                                     role_filter=None):
    """Save per-session regression predictions mirroring annotation file structure.

    Output: out_dir/{session}/{role}.engagement.prediction.csv  (one float per line)
    If role_filter is set, only predictions for that role are saved.
    """
    feat_roots, gt_roots = map(_ensure_list, (feat_roots, gt_roots))
    for froot in feat_roots:
        for groot in gt_roots:
            sessions = [s for s in os.listdir(froot) if os.path.isdir(os.path.join(froot, s))]
            for sess_id in sessions:
                feat_dir = os.path.join(froot, sess_id)
                gt_dir   = os.path.join(groot, sess_id)
                if not os.path.isdir(gt_dir):
                    continue
                for fn in os.listdir(feat_dir):
                    if modality not in fn:
                        continue
                    role      = fn.split('.')[0]
                    if role_filter is not None and role != role_filter:
                        continue
                    feat_path = os.path.join(feat_dir, fn)
                    anno_path = os.path.join(gt_dir, f"{role}.engagement.annotation.csv")
                    if not os.path.exists(anno_path):
                        continue
                    try:
                        a = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)
                        with open(anno_path, encoding='utf-8') as fh:
                            annos = [ln.strip() for ln in fh]
                    except Exception as e:
                        print(f"  [Warning] {feat_path}: {e}")
                        continue
                    n = min(len(a), len(annos))
                    valid_idxs = [i for i, v in enumerate(annos[:n])
                                  if v not in ('', 'nan', '-nan(ind)')]
                    if not valid_idxs:
                        continue
                    X_scaled = scaler.transform(a[valid_idxs])
                    y_pred = model.predict(X_scaled, batch_size=128, verbose=0).squeeze()
                    sess_out = os.path.join(out_dir, sess_id)
                    os.makedirs(sess_out, exist_ok=True)
                    np.savetxt(os.path.join(sess_out, f"{role}.engagement.prediction.csv"),
                               np.atleast_1d(y_pred), fmt="%.6f")


def save_test_predictions_classification(feat_roots, gt_roots, modality, feat_dim, scaler, model,
                                         dataset_config, out_dir, role_filter=None):
    """Save per-session classification predictions mirroring annotation file structure.

    Output: out_dir/{session}/{role}.{head}.engagement.prediction.csv  (one string label per line)
    If role_filter is set, only predictions for that role are saved.
    """
    feat_roots, gt_roots = map(_ensure_list, (feat_roots, gt_roots))
    head_names = list(dataset_config['label_heads'].keys())
    heads      = list(dataset_config['label_heads'].values())
    inv_maps   = [{v: k for k, v in h['label_map'].items()} for h in heads]

    for froot in feat_roots:
        for groot in gt_roots:
            sessions = [s for s in os.listdir(froot) if os.path.isdir(os.path.join(froot, s))]
            for sess_id in sessions:
                feat_dir = os.path.join(froot, sess_id)
                gt_dir   = os.path.join(groot, sess_id)
                if not os.path.isdir(gt_dir):
                    continue
                for fn in os.listdir(feat_dir):
                    if modality not in fn:
                        continue
                    role      = fn.split('.')[0]
                    if role_filter is not None and role != role_filter:
                        continue
                    feat_path = os.path.join(feat_dir, fn)
                    anno_paths, skip = {}, False
                    for head_name, head in zip(head_names, heads):
                        ap = os.path.join(gt_dir, f"{role}{head['annotation_suffix']}")
                        if not os.path.exists(ap):
                            skip = True
                            break
                        anno_paths[head_name] = ap
                    if skip:
                        continue
                    try:
                        a = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)
                    except Exception as e:
                        print(f"  [Warning] {feat_path}: {e}")
                        continue
                    # use first head's annotations to determine valid frame indices
                    try:
                        with open(anno_paths[head_names[0]], encoding='utf-8') as fh:
                            ref_annos = [ln.strip() for ln in fh]
                    except UnicodeDecodeError:
                        with open(anno_paths[head_names[0]], encoding='latin1', errors='ignore') as fh:
                            ref_annos = [ln.strip() for ln in fh]
                    n = min(len(a), len(ref_annos))
                    valid_idxs = [i for i, v in enumerate(ref_annos[:n])
                                  if v not in ('', 'nan', '-nan(ind)')]
                    if not valid_idxs:
                        continue
                    X_scaled = scaler.transform(a[valid_idxs])
                    preds = model.predict(X_scaled, batch_size=128, verbose=0)
                    if not isinstance(preds, list):
                        preds = [preds]
                    sess_out = os.path.join(out_dir, sess_id)
                    os.makedirs(sess_out, exist_ok=True)
                    for head_name, head, pred_probs, inv_map in zip(head_names, heads, preds, inv_maps):
                        pred_classes = np.argmax(pred_probs, axis=1)
                        pred_labels  = [inv_map.get(int(c), str(c)) for c in pred_classes]
                        suffix = head['annotation_suffix'].replace('.annotation.csv', '.engagement.prediction.csv')
                        out_path = os.path.join(sess_out, f"{role}{suffix}")
                        with open(out_path, 'w', encoding='utf-8') as fh:
                            fh.write('\n'.join(pred_labels) + '\n')


# ---------------------------------------------------------------------------
#                                 MAIN
# ---------------------------------------------------------------------------
def canonical(mod_name: str) -> str:
    return mod_name.lstrip(".").rstrip("~")


_DATASET_FOLDER_MAP = {"test-additional": "noxi-additional"}

def _dataset_name_from_config(config_path: str) -> str:
    """Derive a dataset label from the config file path.

    Expects a path containing a 'configs' directory, e.g.
    baseline/configs/noxi-base/test/config1.json  ->  'noxi-base'
    """
    parts = pathlib.Path(config_path).parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "configs")
        folder = parts[idx + 1]
    except (StopIteration, IndexError):
        folder = pathlib.Path(config_path).parent.parent.name
    return _DATASET_FOLDER_MAP.get(folder, folder)

# ---------------------------------------------------------------------------
#                     CLASSIFICATION DATA LOADER
# ---------------------------------------------------------------------------

def load_classification_data_with_sessions(root_dirs, modality, feat_dim, dataset_config):
    X, y, sessions = [], [], []
    for root in (root_dirs if isinstance(root_dirs, list) else [root_dirs]):
        for dirpath, _, files in os.walk(root):
            session_id = os.path.basename(dirpath)
            stream_map, anno_map = {}, {}
            for fname in files:
                key = f"{fname.split('.')[0]};{session_id}"
                fpath = os.path.join(dirpath, fname)
                if modality in fname:
                    stream_map[key] = fpath
                for eng_type, head in dataset_config['label_heads'].items():
                    if fname.endswith(head['annotation_suffix']):
                        if key not in anno_map:
                            anno_map[key] = {}
                        anno_map[key][eng_type] = fpath
            for base_key, anno_paths in anno_map.items():
                if len(anno_paths) < len(dataset_config['label_heads']):
                    continue
                stream_path = stream_map.get(base_key)
                if not stream_path:
                    continue
                a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)
                all_annos = {}
                for eng_type, path in anno_paths.items():
                    try:
                        with open(path, encoding='utf-8') as fh:
                            all_annos[eng_type] = [ln.strip() for ln in fh]
                    except UnicodeDecodeError:
                        with open(path, encoding='latin1', errors='ignore') as fh:
                            all_annos[eng_type] = [ln.strip() for ln in fh]
                n = min(len(a), *[len(v) for v in all_annos.values()])
                for i in range(n):
                    labels = []
                    for eng_type, head in dataset_config['label_heads'].items():
                        val = all_annos[eng_type][i]
                        if val in ('', 'nan', '-nan(ind)'):
                            labels.append(-1)
                            continue
                        idx = head['label_map'].get(val)
                        labels.append(idx if idx is not None else -1)
                    if all(l == -1 for l in labels):
                        continue
                    X.append(a[i])
                    y.append(labels)
                    sessions.append(session_id)
    return (np.asarray(X, np.float32),
            np.asarray(y, np.int32),
            np.asarray(sessions))


def load_classification_test_data_with_gt(feat_roots, gt_roots, modality, feat_dim, dataset_config):
    """Like load_classification_data_with_sessions but with separate feature/GT roots for test."""
    feat_roots = feat_roots if isinstance(feat_roots, list) else [feat_roots]
    gt_roots = gt_roots if isinstance(gt_roots, list) else [gt_roots]
    X, y, sessions = [], [], []
    for froot in feat_roots:
        for groot in gt_roots:
            sess_ids = [s for s in os.listdir(froot) if os.path.isdir(os.path.join(froot, s))]
            for sess_id in sess_ids:
                feat_dir = os.path.join(froot, sess_id)
                gt_dir = os.path.join(groot, sess_id)
                if not os.path.isdir(gt_dir):
                    continue
                for fname in os.listdir(feat_dir):
                    if modality not in fname:
                        continue
                    role = fname.split('.')[0]
                    feat_path = os.path.join(feat_dir, fname)
                    anno_paths = {}
                    skip = False
                    for eng_type, head in dataset_config['label_heads'].items():
                        anno_file = os.path.join(gt_dir, f"{role}{head['annotation_suffix']}")
                        if not os.path.exists(anno_file):
                            skip = True
                            break
                        anno_paths[eng_type] = anno_file
                    if skip:
                        continue
                    try:
                        a = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)
                    except Exception as e:
                        print(f"[Warning] Could not load {feat_path}: {e}")
                        continue
                    all_annos = {}
                    for eng_type, path in anno_paths.items():
                        try:
                            with open(path, encoding='utf-8') as fh:
                                all_annos[eng_type] = [ln.strip() for ln in fh]
                        except UnicodeDecodeError:
                            with open(path, encoding='latin1', errors='ignore') as fh:
                                all_annos[eng_type] = [ln.strip() for ln in fh]
                    n = min(len(a), *[len(v) for v in all_annos.values()])
                    for i in range(n):
                        labels = []
                        for eng_type, head in dataset_config['label_heads'].items():
                            val = all_annos[eng_type][i]
                            if val in ('', 'nan', '-nan(ind)'):
                                labels.append(-1)
                                continue
                            idx = head['label_map'].get(val)
                            labels.append(idx if idx is not None else -1)
                        if all(l == -1 for l in labels):
                            continue
                        X.append(a[i])
                        y.append(labels)
                        sessions.append(sess_id)
    return (np.asarray(X, np.float32),
            np.asarray(y, np.int32),
            np.asarray(sessions))


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate engagement models with fairness metrics"
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to JSON config file (same as used by script 3)")
    p.add_argument("--results_dir", type=str, default="./results/full_test_fairness",
                   help="Where to save results")
    p.add_argument("--batch_size", type=int, default=0)
    return p.parse_args()



def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    # Auto-detect dataset-config.json (same logic as script 3)
    dataset_configpath = os.path.join(
        os.path.dirname(os.path.dirname(args.config)), 'dataset-config.json'
    )
    dataset_config = {}
    if os.path.exists(dataset_configpath):
        with open(dataset_configpath) as fh:
            dataset_config = json.load(fh)
    is_classification = dataset_config.get('task_type', 'regression').lower() == 'classification'

    modalities    = config['modalities']
    dims          = config['modalities_dim']
    train_dirs    = [config['train_dir']]
    val_dirs      = [config['val_dir']]
    test_dirs     = [config['test_dir']]
    gt_dirs       = [config.get('engagement_gt', config['test_dir'])]
    full_base_dir = os.path.expanduser(config.get('full_base_dir', './results/full_models'))

    dataset_name = _dataset_name_from_config(args.config)
    role_filter  = "purple" if dataset_name == "pinsoro-cr" else None

    out_dir = os.path.join(args.results_dir, dataset_name, "full_test_fairness")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(
        out_dir, f"test_summary_{datetime.now():%Y%m%d_%H%M%S}.txt"
    )
    summary_log = open(summary_path, "w")
    summary_log.write("# Evaluation summary\n")
    summary_log.write(json.dumps(vars(args), indent=2) + "\n\n")
    summary_log.flush()

    for modality, feat_dim in zip(modalities, dims):
        tag          = modality.strip(".~")
        modality_can = canonical(modality)

        print(f"\n=== {tag} ({feat_dim}-D) ===")
        summary_log.write(f"\n=== {tag} ({feat_dim}-D) ===\n")

        # ---- Model loading (flat file, same convention as script 3) ----
        model_file = os.path.join(full_base_dir, f"retrained_full_nn_model_{modality_can}.keras")
        if not os.path.exists(model_file):
            msg = f"[ERR] model not found: {model_file}"
            print(msg); summary_log.write(msg + "\n"); summary_log.flush(); continue

        # ---- Classification path ----
        if is_classification:
            Xtr, _, _ = load_classification_data_with_sessions(train_dirs, modality, feat_dim, dataset_config)
            Xva, _, _ = load_classification_data_with_sessions(val_dirs,   modality, feat_dim, dataset_config)
            Xte, yte, test_sess = load_classification_test_data_with_gt(test_dirs, gt_dirs, modality, feat_dim, dataset_config)

            if Xte.size == 0:
                msg = f"[Warning] No test data for {tag}, skipping."
                print(msg); summary_log.write(msg + "\n"); continue

            def _clean(a): a = np.nan_to_num(a); a[np.isinf(a)] = 0; return a
            Xtr, Xva, Xte = map(_clean, (Xtr, Xva, Xte))
            scaler = MinMaxScaler().fit(np.concatenate([Xtr, Xva], 0) if len(Xtr) else Xva)
            Xte_norm = scaler.transform(Xte)

            model = tf.keras.models.load_model(model_file)
            batch = args.batch_size or min(2048, max(128, len(Xte_norm) // 64))
            preds = model.predict(Xte_norm, batch_size=batch, verbose=0)
            if not isinstance(preds, list):
                preds = [preds]

            head_names = list(dataset_config['label_heads'].keys())
            head_accs, head_kappas = [], []
            for h, (head_name, pred_probs) in enumerate(zip(head_names, preds)):
                y_true = yte[:, h]
                y_pred = np.argmax(pred_probs, axis=1)
                valid = y_true >= 0
                acc   = accuracy_score(y_true[valid], y_pred[valid])
                kappa = cohen_kappa_score(y_true[valid], y_pred[valid])
                head_accs.append(acc)
                head_kappas.append(kappa)
                line = f"[{head_name}] Accuracy: {acc:.4f}  Cohen's Kappa: {kappa:.4f}"
                print(line); summary_log.write(line + "\n")
                np.save(os.path.join(out_dir, f"ypred_{tag}_{head_name}.npy"), pred_probs)

            np.save(os.path.join(out_dir, f"yte_{tag}.npy"), yte)
            np.save(os.path.join(out_dir, f"test_sessions_{tag}.npy"), test_sess)
            with open(os.path.join(out_dir, f"test_accuracy_{tag}.txt"), "w") as f:
                for a in head_accs:
                    f.write(str(a) + "\n")
            with open(os.path.join(out_dir, f"test_kappa_{tag}.txt"), "w") as f:
                for v in head_kappas:
                    f.write(str(v) + "\n")
            summary_log.flush()

            pred_out_dir = os.path.join("results", "test_prediction_files", dataset_name, tag)
            print(f"Saving test prediction files for {tag} → {pred_out_dir}")
            save_test_predictions_classification(
                test_dirs, gt_dirs, modality, feat_dim, scaler, model, dataset_config, pred_out_dir,
                role_filter=role_filter,
            )
            continue

        # ---- Regression data loading ----
        Xtr, ytr, _ = load_data_with_sessions(train_dirs, modality, feat_dim)
        Xva, yva, _ = load_data_with_sessions(val_dirs,   modality, feat_dim)
        Xte, yte, test_sess, lang_arr, gender_arr = load_test_data_with_gt_multi(
            test_dirs, gt_dirs, modality, feat_dim
        )

        def _clean(a):
            a = np.nan_to_num(a)
            a[np.isinf(a)] = 0
            return a

        Xtr, ytr, Xva, yva, Xte, yte = map(_clean, (Xtr, ytr, Xva, yva, Xte, yte))
        scaler = MinMaxScaler().fit(
            np.concatenate([Xtr, Xva], 0) if len(Xtr) else Xva
        )
        Xte_norm = scaler.transform(Xte)

        model = tf.keras.models.load_model(model_file)

        # ---- Prediction and metrics ----
        batch = args.batch_size or min(2048, max(128, len(Xte_norm) // 64))
        ypred = model.predict(Xte_norm, batch_size=batch, verbose=0).squeeze()

        mse  = mean_squared_error(yte, ypred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(yte, ypred)
        ccc  = concordance_correlation_coefficient(yte, ypred)
        try:
            r2 = r2_score(yte, ypred)
        except Exception:
            r2 = float("nan")

        mask_g = np.isfinite(gender_arr)
        cdd_g  = (fairness_gender(yte[mask_g], ypred[mask_g], gender_arr[mask_g])
                  if mask_g.any() else None)

        mask_l = lang_arr != "Unknown"
        cdd_l  = (fairness_language(yte[mask_l], ypred[mask_l], lang_arr[mask_l])
                  if mask_l.any() else None)

        line = f"MSE {mse:.4f}  RMSE {rmse:.4f}  MAE {mae:.4f}  CCC {ccc:.4f}  R2 {r2:.4f}"
        print(line)
        summary_log.write(line + "\n")
        summary_log.write(f"CDD_G {cdd_g:.6f}\n" if cdd_g is not None else "CDD_G NA\n")
        summary_log.write(f"CDD_L {json.dumps(cdd_l)}\n" if cdd_l is not None else "CDD_L NA\n")
        summary_log.flush()

        np.save(os.path.join(out_dir, f"yte_{tag}.npy"), yte)
        np.save(os.path.join(out_dir, f"ypred_{tag}.npy"), ypred)
        np.save(os.path.join(out_dir, f"test_sessions_{tag}.npy"),  test_sess)
        np.save(os.path.join(out_dir, f"test_language_{tag}.npy"),  lang_arr)
        np.save(os.path.join(out_dir, f"test_gender_{tag}.npy"),    gender_arr)
        with open(os.path.join(out_dir, f"test_metrics_{tag}.txt"), "w") as fp:
            fp.write("\n".join([
                f"MSE   {mse:.6f}",
                f"RMSE  {rmse:.6f}",
                f"MAE   {mae:.6f}",
                f"CCC   {ccc:.6f}",
                f"R2    {r2:.6f}",
                f"CDD_G {cdd_g:.6f}" if cdd_g is not None else "CDD_G NA",
                f"CDD_L {json.dumps(cdd_l)}" if cdd_l is not None else "CDD_L NA",
                f"Model {model_file}",
            ]))

        # Individual metric files (mirrors 1 / 2.1 pattern – one value per line)
        with open(os.path.join(out_dir, f"test_ccc_{tag}.txt"), "w") as f:
            f.write(str(ccc) + "\n")
        with open(os.path.join(out_dir, f"test_mse_{tag}.txt"), "w") as f:
            f.write(str(mse) + "\n")
        with open(os.path.join(out_dir, f"test_mae_{tag}.txt"), "w") as f:
            f.write(str(mae) + "\n")
        with open(os.path.join(out_dir, f"test_r2_{tag}.txt"), "w") as f:
            f.write(str(r2) + "\n")
        if cdd_g is not None:
            with open(os.path.join(out_dir, f"test_cdd_g_{tag}.txt"), "w") as f:
                f.write(str(cdd_g) + "\n")
        if cdd_l is not None:
            with open(os.path.join(out_dir, f"test_cdd_l_{tag}.txt"), "w") as f:
                f.write(json.dumps(cdd_l) + "\n")

        # ---- Test prediction files (annotation-mirrored) ----
        pred_out_dir = os.path.join("results", "test_prediction_files", dataset_name, tag)
        print(f"Saving test prediction files for {tag} → {pred_out_dir}")
        save_test_predictions_regression(
            test_dirs, gt_dirs, modality, feat_dim, scaler, model, pred_out_dir,
            role_filter=role_filter,
        )

        # ---- Sessionwise predictions ----
        print(f"Saving sessionwise predictions for {tag}...")
        sessionwise_outdir = os.path.join(out_dir, f"sessionwise_pred_{tag}")
        os.makedirs(sessionwise_outdir, exist_ok=True)
        save_sessionwise_predictions_testgt(
            feat_roots=test_dirs,
            gt_roots=gt_dirs,
            modality=modality,
            feat_dim=feat_dim,
            scaler=scaler,
            model=model,
            results_dir=sessionwise_outdir,
        )
        print(f"Sessionwise predictions saved in: {sessionwise_outdir}")

    summary_log.close()
    print(f"\nSummary written to {summary_path}")




if __name__ == "__main__":
    main()
