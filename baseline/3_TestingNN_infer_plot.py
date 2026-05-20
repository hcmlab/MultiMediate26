#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# ---- CCC Metric ----
def concordance_correlation_coefficient(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12
    return numerator / denominator

# ---- Data Loader with Session Tracking ----
def load_data_with_sessions(root_dir, modality, feat_dim):
    stream_map, anno_map = {}, {}
    for dirpath, _, files in os.walk(root_dir):
        session_id = os.path.basename(dirpath)
        for fname in files:
            key = f"{fname.split('.')[0]};{session_id}"
            full_path = os.path.join(dirpath, fname)
            if modality in fname:
                stream_map[key] = full_path
            if fname.endswith('.engagement.annotation.csv'):
                anno_map[key] = full_path
    X, y, session_list = [], [], []
    for key, anno_path in anno_map.items():
        stream_path = stream_map.get(key)
        session_id = key.split(';')[1]
        if not stream_path:
            continue
        try:
            a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)
            with open(anno_path, 'r', encoding='utf-8') as f:
                annos = [line.strip() for line in f]
        except UnicodeDecodeError:
            with open(anno_path, 'r', encoding='latin1', errors='ignore') as f:
                annos = [line.strip() for line in f]
        except Exception as e:
            print(f"[Warning] Could not load {stream_path} or {anno_path}: {e}")
            continue
        n = min(len(a), len(annos))
        for i in range(n):
            val = annos[i]
            if val in ('', 'nan', '-nan(ind)'):
                continue
            try:
                y_val = float(val)
                X.append(a[i])
                y.append(y_val)
                session_list.append(session_id)
            except ValueError:
                continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(session_list)

# ---- Test Loader (Separate roots, session tracked) ----
def load_test_data_with_gt(test_feat_root, gt_root, modality, feat_dim):
    X_list, y_list, session_list = [], [], []
    sessions = [s for s in os.listdir(test_feat_root)
                if os.path.isdir(os.path.join(test_feat_root, s))
                and os.path.isdir(os.path.join(gt_root, s))]
    for session in sessions:
        session_feat_dir = os.path.join(test_feat_root, session)
        session_gt_dir = os.path.join(gt_root, session)
        feat_files = [f for f in os.listdir(session_feat_dir) if modality in f]
        for fname in feat_files:
            stream_path = os.path.join(session_feat_dir, fname)
            role = fname.split('.')[0]
            anno_files = [f for f in os.listdir(session_gt_dir)
                          if f.startswith(role) and f.endswith('.engagement.annotation.csv')]
            if not anno_files:
                continue
            anno_path = os.path.join(session_gt_dir, anno_files[0])
            try:
                a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)
                with open(anno_path, 'r', encoding='utf-8') as f:
                    annos = [line.strip() for line in f]
            except UnicodeDecodeError:
                with open(anno_path, 'r', encoding='latin1', errors='ignore') as f:
                    annos = [line.strip() for line in f]
            except Exception as e:
                print(f"[Warning] Could not load {stream_path} or {anno_path}: {e}")
                continue
            n = min(len(a), len(annos))
            for i in range(n):
                val = annos[i]
                if val in ('', 'nan', '-nan(ind)'):
                    continue
                try:
                    y_val = float(val)
                    X_list.append(a[i])
                    y_list.append(y_val)
                    session_list.append(session)
                except ValueError:
                    continue
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(session_list)

# ---- Data Reporting ----
def report_data_stats(name, X, y, session_list, logfile, n_bins=20):
    log = []
    log.append(f"\n--- {name} Stats ---")
    log.append(f"Samples: {len(y)}")
    log.append(f"Sessions: {len(np.unique(session_list)) if session_list is not None else 'N/A'}")
    # NaN/Inf reporting
    n_feat_nan = np.isnan(X).sum()
    n_feat_inf = np.isinf(X).sum()
    n_y_nan = np.isnan(y).sum()
    n_y_inf = np.isinf(y).sum()
    log.append(f"Feature NaNs: {n_feat_nan}, Infs: {n_feat_inf}")
    log.append(f"Target NaNs: {n_y_nan}, Infs: {n_y_inf}")
    # Statistics
    if len(y) > 0:
        log.append(f"Target Min: {np.min(y):.4f}, Max: {np.max(y):.4f}, Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}")
        percentiles = np.percentile(y, [0, 1, 25, 50, 75, 99, 100])
        log.append("Target percentiles: " + ", ".join([f"{p:.2f}" for p in percentiles]))
        # Histogram (value distribution)
        hist, bin_edges = np.histogram(y, bins=n_bins)
        log.append(f"Histogram counts (bins={n_bins}): {hist.tolist()}")
        log.append(f"Histogram bin edges: {bin_edges.tolist()}")
    # Write out
    logfile.write("\n".join(log) + "\n")
    logfile.flush()
    print("\n".join(log))

# ---- Classification Data Loader ----
def load_classification_data_with_sessions(root_dir, modality, feat_dim, dataset_config):
    stream_map, anno_map = {}, {}
    for dirpath, _, files in os.walk(root_dir):
        session_id = os.path.basename(dirpath)
        for fname in files:
            key = f"{fname.split('.')[0]};{session_id}"
            full_path = os.path.join(dirpath, fname)
            if modality in fname:
                stream_map[key] = full_path
            for eng_type, head in dataset_config['label_heads'].items():
                if fname.endswith(head['annotation_suffix']):
                    if key not in anno_map:
                        anno_map[key] = {}
                    anno_map[key][eng_type] = full_path
    X, y, sessions = [], [], []
    for base_key, anno_paths in anno_map.items():
        if len(anno_paths) < len(dataset_config['label_heads']):
            continue
        stream_path = stream_map.get(base_key)
        session_id = base_key.split(';')[1]
        if not stream_path:
            continue
        a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)
        all_annos = {}
        for eng_type, path in anno_paths.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    all_annos[eng_type] = [line.strip() for line in f]
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin1', errors='ignore') as f:
                    all_annos[eng_type] = [line.strip() for line in f]
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
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(sessions))


# ---- Classification Test Loader (Separate feat/gt roots, multi-head) ----
def load_test_classification_data_with_gt(test_feat_root, gt_root, modality, feat_dim, dataset_config):
    X_list, y_list, session_list = [], [], []
    sessions = [s for s in os.listdir(test_feat_root)
                if os.path.isdir(os.path.join(test_feat_root, s))
                and os.path.isdir(os.path.join(gt_root, s))]
    for session in sessions:
        session_feat_dir = os.path.join(test_feat_root, session)
        session_gt_dir = os.path.join(gt_root, session)
        feat_files = [f for f in os.listdir(session_feat_dir) if modality in f]
        for fname in feat_files:
            stream_path = os.path.join(session_feat_dir, fname)
            role = fname.split('.')[0]
            all_annos = {}
            skip = False
            for eng_type, head in dataset_config['label_heads'].items():
                anno_files = [f for f in os.listdir(session_gt_dir)
                              if f.startswith(role) and f.endswith(head['annotation_suffix'])]
                if not anno_files:
                    skip = True
                    break
                anno_path = os.path.join(session_gt_dir, anno_files[0])
                try:
                    with open(anno_path, 'r', encoding='utf-8') as f:
                        all_annos[eng_type] = [line.strip() for line in f]
                except UnicodeDecodeError:
                    with open(anno_path, 'r', encoding='latin1', errors='ignore') as f:
                        all_annos[eng_type] = [line.strip() for line in f]
            if skip:
                continue
            try:
                a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)
            except Exception as e:
                print(f"[Warning] Could not load {stream_path}: {e}")
                continue
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
                X_list.append(a[i])
                y_list.append(labels)
                session_list.append(session)
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(session_list))


# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    parser.add_argument('--results_dir', type=str, default="./results/full_test_predictions_plots", help="Where to save test predictions/metrics")
    parser.add_argument('--batch_size', type=int, default=0, help="Override batch size (0=auto)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    dataset_configpath = os.path.join(os.path.dirname(os.path.dirname(args.config)), 'dataset-config.json')
    dataset_config = {}
    if os.path.exists(dataset_configpath):
        with open(dataset_configpath, 'r') as f:
            dataset_config = json.load(f)
    is_classification = dataset_config.get('task_type', 'regression').lower() == 'classification'

    modalities = config['modalities']
    modalities_dim = config['modalities_dim']
    train_dir = config['train_dir']
    val_dir = config['val_dir']
    test_feat_root = config['test_dir']
    gt_root = config.get('engagement_gt', test_feat_root)
    full_base_dir = os.path.expanduser(config.get('full_base_dir', './results/full_models'))

    os.makedirs(args.results_dir, exist_ok=True)
    summary_logfile_path = os.path.join(args.results_dir, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    summary_logfile = open(summary_logfile_path, 'w')
    summary_logfile.write(f"Testing Summary Log: {datetime.now()}\n")
    summary_logfile.write(f"Config: {json.dumps(config, indent=2)}\n\n")

    for idx, modality in enumerate(modalities):
        feat_dim = modalities_dim[idx]
        print(f"\n=== Testing modality {modality} ({feat_dim}-D) ===")
        summary_logfile.write(f"\n\n=== Modality: {modality} ({feat_dim}-D) ===\n")

        # --- Classification path ---
        if is_classification:
            Xtr, _, _ = load_classification_data_with_sessions(train_dir, modality, feat_dim, dataset_config)
            Xva, _, _ = load_classification_data_with_sessions(val_dir, modality, feat_dim, dataset_config)
            if gt_root != test_feat_root:
                Xte, yte, test_sessions = load_test_classification_data_with_gt(test_feat_root, gt_root, modality, feat_dim, dataset_config)
            else:
                Xte, yte, test_sessions = load_classification_data_with_sessions(test_feat_root, modality, feat_dim, dataset_config)

            if Xte.size == 0:
                print(f"[Warning] No test data for {modality}, skipping.")
                continue

            combined = np.concatenate([Xtr, Xva], axis=0) if Xtr.size > 0 and Xva.size > 0 else (Xtr if Xtr.size > 0 else Xva)
            scaler = MinMaxScaler().fit(np.nan_to_num(combined))
            Xte_norm = np.nan_to_num(scaler.transform(Xte))

            batch_size = args.batch_size or min(2048, max(128, len(Xte_norm) // 64))

            model_name = f"retrained_full_nn_model_{modality.strip('.~')}.keras"
            model_file = os.path.join(full_base_dir, model_name)
            if not os.path.exists(model_file):
                print(f"[Error] Model file not found: {model_file}")
                summary_logfile.write(f"[Error] Model file not found: {model_file}\n")
                continue
            model = tf.keras.models.load_model(model_file)

            preds = model.predict(Xte_norm, batch_size=batch_size, verbose=1)
            if not isinstance(preds, list):
                preds = [preds]

            head_names = list(dataset_config['label_heads'].keys())
            modality_name = modality.strip('.~')
            for h, (head_name, pred_probs) in enumerate(zip(head_names, preds)):
                y_true = yte[:, h]
                y_pred = np.argmax(pred_probs, axis=1)
                valid = y_true >= 0
                acc = accuracy_score(y_true[valid], y_pred[valid])
                f1  = f1_score(y_true[valid], y_pred[valid], average='macro', zero_division=0)
                summary_logfile.write(f"[{head_name}] Accuracy: {acc:.4f}  F1-macro: {f1:.4f}\n")
                print(f"  [{head_name}] Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
                with open(os.path.join(args.results_dir, f'test_metrics_{modality_name}_{head_name}.txt'), 'w') as mf:
                    mf.write(f"Accuracy: {acc:.6f}\nF1-macro: {f1:.6f}\nModel: {model_file}\n")

                label_map = dataset_config['label_heads'][head_name]['label_map']
                class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
                cm = confusion_matrix(y_true[valid], y_pred[valid])
                fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right'); ax.set_yticklabels(class_names)
                ax.set_xlabel('Predicted'); ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix: {modality_name} / {head_name}')
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                                color='white' if cm[i, j] > cm.max() / 2 else 'black')
                plt.tight_layout()
                plt.savefig(os.path.join(args.results_dir, f'confusion_{modality_name}_{head_name}.png'), dpi=180)
                plt.close()

            np.save(os.path.join(args.results_dir, f'yte_{modality_name}.npy'), yte)
            np.save(os.path.join(args.results_dir, f'test_sessions_{modality_name}.npy'), test_sessions)
            for h, (head_name, pred_probs) in enumerate(zip(head_names, preds)):
                np.save(os.path.join(args.results_dir, f'ypred_{modality_name}_{head_name}.npy'), pred_probs)
            continue

        # --- Load data with session tracking ---
        Xtr, ytr, train_sessions = load_data_with_sessions(train_dir, modality, feat_dim)
        Xva, yva, val_sessions = load_data_with_sessions(val_dir, modality, feat_dim)
        Xte, yte, test_sessions = load_test_data_with_gt(test_feat_root, gt_root, modality, feat_dim)

        # --- Report stats and NaNs/Infs ---
        report_data_stats("Train", Xtr, ytr, train_sessions, summary_logfile)
        report_data_stats("Val", Xva, yva, val_sessions, summary_logfile)
        report_data_stats("Test", Xte, yte, test_sessions, summary_logfile)

        # --- Remove or fill NaNs/Infs for scaling ---
        def clean(X, y):
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            X[np.isinf(X)] = 0
            y[np.isinf(y)] = 0
            return X, y
        Xtr, ytr = clean(Xtr, ytr)
        Xva, yva = clean(Xva, yva)
        Xte, yte = clean(Xte, yte)

        # --- Adaptive batch size based on available memory ---
        batch_size = args.batch_size
        if batch_size == 0:
            # 48 GPU, 64GB CPU RAM: batch size could be large, but best not to overdo it
            batch_size = min(2048, max(128, int(len(Xte) // 64)))
            print(f"[Info] Adaptive batch size for {modality}: {batch_size}")
            summary_logfile.write(f"Adaptive batch size: {batch_size}\n")
        else:
            print(f"[Info] Using user-specified batch size for {modality}: {batch_size}")
            summary_logfile.write(f"Batch size (user): {batch_size}\n")

        # --- Fit scaler on train+val ---
        combined = np.concatenate([Xtr, Xva], axis=0) if Xtr.size > 0 and Xva.size > 0 else Xtr if Xtr.size > 0 else Xva
        scaler = MinMaxScaler().fit(combined)

        # --- Scale test data ---
        Xte_norm = scaler.transform(Xte)
        Xte_norm = np.nan_to_num(Xte_norm)

        # --- Load model ---
        model_name = f"retrained_full_nn_model_{modality.strip('.~')}.keras"
        model_file = os.path.join(full_base_dir, model_name)
        if not os.path.exists(model_file):
            print(f"[Error] Model file not found for {modality} in {full_base_dir}")
            summary_logfile.write(f"[Error] Model file not found: {model_file}\n")
            continue
        print(f"Loading model: {model_file}")
        model = tf.keras.models.load_model(model_file)

        # --- Predict ---
        ypred = model.predict(Xte_norm, batch_size=batch_size, verbose=1)
        ypred = np.squeeze(ypred)

        # --- Metrics ---
        mse = mean_squared_error(yte, ypred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(yte, ypred)
        ccc = concordance_correlation_coefficient(yte, ypred)
        r2 = r2_score(yte, ypred)
        summary_logfile.write(f"Test MSE: {mse:.6f}\n")
        summary_logfile.write(f"Test RMSE: {rmse:.6f}\n")
        summary_logfile.write(f"Test MAE: {mae:.6f}\n")
        summary_logfile.write(f"Test CCC: {ccc:.6f}\n")
        summary_logfile.write(f"Test R2: {r2:.6f}\n")
        summary_logfile.flush()
        print(f"Test MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  CCC: {ccc:.4f}  R2: {r2:.4f}")

        # --- Save predictions and metrics ---
        modality_name = modality.strip('.~')
        np.save(os.path.join(args.results_dir, f'yte_{modality_name}.npy'), yte)
        np.save(os.path.join(args.results_dir, f'ypred_{modality_name}.npy'), ypred)
        np.save(os.path.join(args.results_dir, f'test_sessions_{modality_name}.npy'), test_sessions)
        with open(os.path.join(args.results_dir, f'test_metrics_{modality_name}.txt'), 'w') as f:
            f.write(f"Test MSE: {mse:.6f}\n")
            f.write(f"Test RMSE: {rmse:.6f}\n")
            f.write(f"Test MAE: {mae:.6f}\n")
            f.write(f"Test CCC: {ccc:.6f}\n")
            f.write(f"Test R2: {r2:.6f}\n")
            f.write(f"Model used: {model_file}\n")

        # --- Diagnostic Plots ---
        plt.figure()
        plt.scatter(yte, ypred, alpha=0.4, s=5)
        plt.xlabel('True Engagement')
        plt.ylabel('Predicted Engagement')
        plt.title(f'Prediction Scatter ({modality_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, f'scatter_{modality_name}.png'), dpi=180)
        plt.close()

        plt.figure()
        plt.hist(yte - ypred, bins=100)
        plt.xlabel('Residual (True - Predicted)')
        plt.ylabel('Count')
        plt.title(f'Residuals Histogram ({modality_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, f'residuals_{modality_name}.png'), dpi=180)
        plt.close()

        # --- Per-session analysis ---
        session_metrics = {}
        unique_sessions = np.unique(test_sessions)
        with open(os.path.join(args.results_dir, f'per_session_metrics_{modality_name}.txt'), 'w') as f:
            f.write(f"Session\tSamples\tMSE\tRMSE\tMAE\tCCC\tR2\n")
            for session in unique_sessions:
                idx = np.where(test_sessions == session)[0]
                if len(idx) < 5:
                    continue  # skip tiny sessions (optionally tune)
                ys_true = yte[idx]
                ys_pred = ypred[idx]
                sess_mse = mean_squared_error(ys_true, ys_pred)
                sess_rmse = np.sqrt(sess_mse)
                sess_mae = mean_absolute_error(ys_true, ys_pred)
                sess_ccc = concordance_correlation_coefficient(ys_true, ys_pred)
                try:
                    sess_r2 = r2_score(ys_true, ys_pred)
                except Exception:
                    sess_r2 = float('nan')
                f.write(f"{session}\t{len(idx)}\t{sess_mse:.4f}\t{sess_rmse:.4f}\t{sess_mae:.4f}\t{sess_ccc:.4f}\t{sess_r2:.4f}\n")
                session_metrics[session] = dict(mse=sess_mse, rmse=sess_rmse, mae=sess_mae, ccc=sess_ccc, r2=sess_r2)
        summary_logfile.write(f"Per-session metrics written to per_session_metrics_{modality_name}.txt\n")
        summary_logfile.flush()

    summary_logfile.close()

if __name__ == "__main__":
    main()
