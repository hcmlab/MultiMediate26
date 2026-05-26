"""
Retrains a single-modality NN on the combined train+val set using the best
hyperparameters from tuning. Supports both regression (CCC/MSE) and
classification tasks. Saves the trained model and logs a FINAL_RESULT line
for downstream result extraction.
"""

#!/usr/bin/env python3
import os
import gc
import json
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle, class_weight as sklearn_cw
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from keras import layers, models

# ---------------- Argument Parsing --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to per-feature JSON config file")
    return parser.parse_args()

# --------------- Concordance Correlation Coefficient ---------------
def concordance_correlation_coefficient(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2 + 1e-12
    return numerator / denominator

# ----------------- Config Loading ---------------------
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

# ------------------ Data Loader -----------------------
def load_data_for_modality(train_dir, val_dir, modality, feat_dim):
    def walk_and_load(root_dir):
        stream_map, anno_map = {}, {}
        for dirpath, dirnames, files in os.walk(root_dir):
            dirnames.sort()
            session_id = os.path.basename(dirpath)
            for fname in sorted(files):
                key = f"{fname.split('.')[0]};{session_id}"
                full_path = os.path.join(dirpath, fname)
                if modality in fname:
                    stream_map[key] = full_path
                if fname.endswith('.engagement.annotation.csv'):
                    anno_map[key] = full_path
        X, y = [], []
        for key, anno_path in anno_map.items():
            stream_path = stream_map.get(key)
            if not stream_path:
                continue
            a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)
            annos = []
            try:
                with open(anno_path, 'r', encoding='utf-8') as f:
                    annos = [line.strip() for line in f]
            except UnicodeDecodeError:
                with open(anno_path, 'r', encoding='latin1', errors='ignore') as f:
                    annos = [line.strip() for line in f]
            n = min(len(a), len(annos))
            for i in range(n):
                val = annos[i]
                if val in ('', 'nan', '-nan(ind)'):
                    continue
                try:
                    y_val = float(val)
                except ValueError:
                    continue
                X.append(a[i])
                y.append(y_val)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    Xtr, ytr = walk_and_load(train_dir)
    Xva, yva = walk_and_load(val_dir)
    Xfull = np.concatenate([Xtr, Xva], axis=0)
    yfull = np.concatenate([ytr, yva], axis=0)
    return Xfull, yfull, Xva, yva

# --------------- Model Construction -------------------
def build_model(input_shape, hps):
    model = models.Sequential()
    model.add(layers.Dense(hps["units1"], activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(hps["units2"], activation='relu'))
    model.add(layers.Dropout(hps["dropout"]))
    model.add(layers.Dense(hps["units3"], activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hps["learning_rate"]),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

# --------------- Classification Support ---------------
def load_classification_data(train_dir, val_dir, modality, feat_dim, dataset_config):
    def walk_and_load(root_dir):
        stream_map, anno_map = {}, {}
        for dirpath, dirnames, files in os.walk(root_dir):
            dirnames.sort()
            session_id = os.path.basename(dirpath)
            for fname in sorted(files):
                key = f"{fname.split('.')[0]};{session_id}"
                full_path = os.path.join(dirpath, fname)
                if modality in fname:
                    stream_map[key] = full_path
                for eng_type, head in dataset_config['label_heads'].items():
                    if fname.endswith(head['annotation_suffix']):
                        if key not in anno_map:
                            anno_map[key] = {}
                        anno_map[key][eng_type] = full_path
        X, y = [], []
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
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    Xtr, ytr = walk_and_load(train_dir)
    Xva, yva = walk_and_load(val_dir)
    Xfull = np.concatenate([Xtr, Xva], axis=0)
    yfull = np.concatenate([ytr, yva], axis=0)
    return Xfull, yfull, Xva, yva


def build_classification_model(input_shape, hps, num_classes):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(hps["units1"], activation='relu')(inputs)
    x = layers.Dense(hps["units2"], activation='relu')(x)
    x = layers.Dropout(hps["dropout"])(x)
    x = layers.Dense(hps["units3"], activation='relu')(x)
    outputs = [
        layers.Dense(n, activation='softmax', name=head)(x)
        for head, n in num_classes.items()
    ]
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hps["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] * len(outputs)
    )
    return model

_TRAIN_DIR_TO_DATASET = {
    "/mnt/data/noxi/train/":       "noxi-base",
    "/mnt/data/noxi-j/train/":     "noxi-j",
    "/mnt/data/pinsoro/train-cc/": "pinsoro-cc",
    "/mnt/data/pinsoro/train-cr/": "pinsoro-cr",
}

# ----------------- Main Routine -----------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    # SLURM ENV
    slurm_job_id   = os.getenv("SLURM_JOB_ID", "local")
    slurm_job_name = os.getenv("SLURM_JOB_NAME", "local")
    slurm_nodelist = os.getenv("SLURM_NODELIST", "local")

    # Reproducibility
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    dataset_configpath = os.path.join(os.path.dirname(os.path.dirname(args.config)), 'dataset-config.json')
    dataset_config = {}
    if os.path.exists(dataset_configpath):
        dataset_config = load_config(dataset_configpath)
    is_classification = dataset_config.get('task_type', 'regression').lower() == 'classification'

    modalities      = config['modalities']
    modalities_dim  = config['modalities_dim']
    train_dir       = config['train_dir']
    val_dir         = config['val_dir']
    full_base_dir   = os.path.expanduser(config.get('full_base_dir', './results/full_models'))
    best_hps_dict   = config['best_hyperparameters']

    dataset_name = _TRAIN_DIR_TO_DATASET.get(train_dir, slurm_job_name)

    os.makedirs(full_base_dir, exist_ok=True)

    print(f"SLURM Job ID: {slurm_job_id}, Name: {slurm_job_name}, Nodes: {slurm_nodelist}")
    print("Loaded config:")
    print(json.dumps(config, indent=2))

    for idx, modality in enumerate(modalities):
        feat_dim = modalities_dim[idx]
        modality_clean = modality.strip('.~')

        LOG_DIR = f"./nn/logs/{dataset_name}/{modality_clean}/retrain/{slurm_job_id}"
        os.makedirs(LOG_DIR, exist_ok=True)
        logfile = os.path.join(LOG_DIR, f"retrain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        def logmsg(msg, _lf=logfile):
            print(msg)
            with open(_lf, 'a') as f:
                f.write(msg + '\n')

        logmsg(f"SLURM Job ID: {slurm_job_id}, Name: {slurm_job_name}, Nodes: {slurm_nodelist}")
        logmsg("Loaded config:")
        logmsg(json.dumps(config, indent=2))
        logmsg(f"\n--- Retraining modality {modality} ({feat_dim}-D) ---")

        hps = best_hps_dict[modality]

        # --- Classification path ---
        if is_classification:
            Xfull, yfull, Xva, yva = load_classification_data(train_dir, val_dir, modality, feat_dim, dataset_config)
            logmsg(f"Loaded shapes: Xfull={Xfull.shape}, yfull={yfull.shape}")
            if Xfull.size == 0 or yfull.size == 0:
                logmsg(f"[Warning] No data for modality {modality}, skipping.")
                continue

            Xfull = np.nan_to_num(Xfull)
            scaler = MinMaxScaler().fit(Xfull)
            Xfull_norm = np.nan_to_num(scaler.transform(Xfull))
            Xfull_norm, yfull = shuffle(Xfull_norm, yfull, random_state=SEED)
            n_heads = len(dataset_config['label_heads'])
            # Filter rows where any label is -1 (missing sentinel)
            mask = np.all(yfull != -1, axis=1)
            Xfull_norm = Xfull_norm[mask]
            yfull = yfull[mask]
            num_classes = {k: v['num_classes'] for k, v in dataset_config['label_heads'].items()}
            y_split = [yfull[:, i] for i in range(n_heads)]

            head_names = list(dataset_config['label_heads'].keys())
            sample_weights = []
            for i, name in enumerate(head_names):
                classes = np.unique(y_split[i])
                w = sklearn_cw.compute_class_weight(
                    'balanced', classes=classes, y=y_split[i])
                cw = dict(zip(classes.tolist(), w.tolist()))
                sample_weights.append(np.array([cw[int(l)] for l in y_split[i]], dtype=np.float32))

            model = build_classification_model(feat_dim, hps, num_classes)
            model_filepath = os.path.join(full_base_dir, f"retrained_full_nn_model_{modality_clean}.keras")
            logmsg(f"Training model with best hyperparameters: {hps}")
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(Xfull_norm, y_split, epochs=100, batch_size=hps["batch_size"],
                      callbacks=[early_stop], sample_weight=sample_weights,
                      shuffle=True, verbose=2)
            model.save(model_filepath)
            logmsg(f"Model saved to {model_filepath}")
            # Evaluate on val set and write result for 2.1_RetrainResultExtraction.py
            Xva_cls = np.nan_to_num(Xva)
            Xva_norm_eval = np.nan_to_num(scaler.transform(Xva_cls))
            mask_va = np.all(yva != -1, axis=1)
            Xva_eval = Xva_norm_eval[mask_va]
            yva_eval = yva[mask_va]
            if Xva_eval.size > 0:
                yva_split_va = [yva_eval[:, i] for i in range(n_heads)]
                ypred_va = model.predict(Xva_eval, verbose=0)
                ypred_cls_va = [np.argmax(ypred_va[i], axis=1) for i in range(n_heads)]
                accs_va = [accuracy_score(yva_split_va[i], ypred_cls_va[i]) for i in range(n_heads)]
                f1s_va = [f1_score(yva_split_va[i], ypred_cls_va[i], average='weighted') for i in range(n_heads)]
                kappas_va = [cohen_kappa_score(yva_split_va[i], ypred_cls_va[i]) for i in range(n_heads)]
                result = {
                    "accuracy": float(np.mean(accs_va)),
                    "accuracy_per_head": {head_names[i]: float(accs_va[i]) for i in range(n_heads)},
                    "f1_per_head": {head_names[i]: float(f1s_va[i]) for i in range(n_heads)},
                    "kappa": float(np.mean(kappas_va)),
                    "kappa_per_head": {head_names[i]: float(kappas_va[i]) for i in range(n_heads)},
                }
                logmsg(f"FINAL_RESULT: {json.dumps(result)}")
            tf.keras.backend.clear_session()
            gc.collect()
            continue

        Xfull, yfull, Xva, yva = load_data_for_modality(train_dir, val_dir, modality, feat_dim)
        logmsg(f"Loaded shapes: Xfull={Xfull.shape}, yfull={yfull.shape}")

        if Xfull.size == 0 or yfull.size == 0:
            logmsg(f"[Warning] No data for modality {modality}, skipping.")
            continue

        Xfull = np.nan_to_num(Xfull)
        scaler = MinMaxScaler().fit(Xfull)
        Xfull_norm = scaler.transform(Xfull)
        Xfull_norm, yfull = shuffle(Xfull_norm, yfull, random_state=SEED)
        Xfull_norm = np.nan_to_num(Xfull_norm)
        yfull = np.nan_to_num(yfull)

        model = build_model(feat_dim, hps)
        model_filepath = os.path.join(full_base_dir, f"retrained_full_nn_model_{modality_clean}.keras")
        logmsg(f"Training model with best hyperparameters: {hps}")

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(
            Xfull_norm, yfull,
            epochs=100,
            batch_size=hps["batch_size"],
            callbacks=[early_stop],
            shuffle=True,
            verbose=2
        )

        model.save(model_filepath)
        logmsg(f"Model saved to {model_filepath}")
        # Evaluate on val set and write result for 2.1_RetrainResultExtraction.py
        Xva_norm_eval = np.nan_to_num(scaler.transform(np.nan_to_num(Xva)))
        yva_clean = np.nan_to_num(yva)
        ypred_val = np.squeeze(model.predict(Xva_norm_eval, verbose=0))
        ccc_val = concordance_correlation_coefficient(ypred_val, yva_clean)
        mse_val = float(np.mean((ypred_val - yva_clean) ** 2))
        logmsg(f"FINAL_RESULT: {json.dumps({'ccc': float(ccc_val), 'mse': mse_val})}")

        tf.keras.backend.clear_session()
        gc.collect()

    print("Retraining complete.")

if __name__ == "__main__":
    main()
