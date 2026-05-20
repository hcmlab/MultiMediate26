#!/usr/bin/env python3
import os
import gc
import math
import random
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle, class_weight as sklearn_cw
from sklearn.metrics import accuracy_score, f1_score
import keras_tuner as kt
from keras import layers, models
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband
from tensorboard.plugins.hparams import api as hp

# ----------------------- Argument Parsing --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to JSON config file")
    return parser.parse_args()

# ----------------------- Config Loading --------------------------
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

# --------------- HParams plugin setup ---------------
HP_UNITS1 = hp.HParam('units1', hp.Discrete([8, 16, 32, 64, 128, 256, 512]))
HP_UNITS2 = hp.HParam('units2', hp.Discrete([8, 16, 32, 64, 128, 256, 512]))
HP_UNITS3 = hp.HParam('units3', hp.Discrete([8, 16, 32, 64, 128, 256, 512]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.5))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128, 256, 512, 1024, 2048]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4]))

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

# --------------- KerasTuner + batch size ---------------
class MyTuner(Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        batch_size = hp.get('batch_size')
        kwargs['batch_size'] = batch_size
        return super().run_trial(trial, *args, **kwargs)

# --------------- HyperModel ---------------
class MultimediateHyperModelBase(HyperModel):
    def __init__(self, input_shape, num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def build(self, hp):
        #model = models.Sequential()
        u1 = hp.Int('units1', 8, 512, step=8)
        u2 = hp.Int('units2', 8, 512, step=8)
        u3 = hp.Int('units3', 8, 512, step=8)
        #model.add(layers.Dense(u1, activation='relu', input_shape=(self.input_shape,)))
        #model.add(layers.Dense(u2, activation='relu'))
        #model.add(layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.05, default=0.25)))
        #model.add(layers.Dense(u3, activation='relu'))

        dropout = hp.Float('dropout', 0.0, 0.5, step=0.05, default=0.25)

        inputs = layers.Input(shape=(self.input_shape,))
        x = layers.Dense(u1, activation='relu')(inputs)
        x = layers.Dense(u2, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(u3, activation='relu')(x)

    
        return x, inputs

    
class MultimediateHyperModelNoxi(MultimediateHyperModelBase):
    def build(self, hp):        
        x, inputs = super().build(hp)
        out = layers.Dense(self.num_classes, activation='linear')(x)
        model = models.Model(inputs=inputs, outputs=out)
        lr = hp.Choice('learning_rate', [1e-3, 1e-4])
        hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024, 2048], default=128)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        return model
    
class MultimediateHyperModelPinsoro(MultimediateHyperModelBase):
    def build(self, hp):
        x, inputs = super().build(hp)

        outputs = [
            layers.Dense(n, activation='softmax', name=head)(x)
            for head, n in self.num_classes.items()
        ]

        lr = hp.Choice('learning_rate', [1e-3, 1e-4])
        hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024, 2048], default=128)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'] * len(outputs)
        )
        return model

# --------------- Data Loader ---------------
def load_data_for_modality(train_dir, val_dir, modality, feat_dim, is_classification, dataset_config):
    def walk_and_load(root_dir):
        stream_map, anno_map = {}, {}
        for dirpath, _, files in os.walk(root_dir):
            session_id = os.path.basename(dirpath)
            for fname in files:
                key = f"{fname.split('.')[0]};{session_id}"
                full_path = os.path.join(dirpath, fname)
                if modality in fname:
                    stream_map[key] = full_path
                if is_classification:
                    for eng_type, head in dataset_config['label_heads'].items():
                        if fname.endswith(head['annotation_suffix']):
                            if key not in anno_map:
                                anno_map[key] = {}
                            anno_map[key][eng_type] = full_path
                else:
                    if fname.endswith('.engagement.annotation.csv'):
                        anno_map[key] = full_path
        X, y = [], []
        
        # Iterate over anno paths for both versions
        if is_classification:
            load_classification_annos(anno_map, stream_map, X, y, dataset_config, feat_dim)
        else:
            load_all_continuous_annos(anno_map, stream_map, X, y, feat_dim)
            
        #return the different versions of the data as numpy arrays
        if is_classification:
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
        else:
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)       
        
    Xtr, ytr = walk_and_load(train_dir)
    Xva, yva = walk_and_load(val_dir)
    return Xtr, ytr, Xva, yva

def load_classification_annos(anno_map,stream_map,X,y,dataset_config,feat_dim):
    for base_key, anno_paths in anno_map.items():


        if len(anno_paths) < len(dataset_config['label_heads']):
            continue  # skip if not all 3 annotation files found
        stream_path = stream_map.get(base_key)
        if not stream_path:
            continue

        a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)

        # load all 3 annotation files
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
                    labels.append(-1)  # sentinel for missing, need to use partial labels sometimes
                    continue
                idx = head['label_map'].get(val)
                labels.append(idx if idx is not None else -1)
                
            if all(l == -1 for l in labels):
                continue
            X.append(a[i])
            y.append(labels)   # [te_label, se_label, sa_label]
    
def load_all_continuous_annos(anno_map,stream_map,X,y,feat_dim):
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

# --------------- MAIN -------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_configpath = os.path.join(os.path.dirname(os.path.dirname(args.config)), 'dataset-config.json')
    dataset_config = load_config(dataset_configpath)
    is_classification = dataset_config['task_type'].lower() == 'classification'

    # SLURM ENV
    slurm_job_id   = os.getenv("SLURM_JOB_ID", "local")
    slurm_job_name = os.getenv("SLURM_JOB_NAME", "local")
    slurm_nodelist = os.getenv("SLURM_NODELIST", "local")

    # Set seed for reproducibility
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Unpack config vars
    modalities        = config['modalities']
    modalities_dim    = config['modalities_dim']
    DEBUG_MODE        = config.get('debug_mode', False)
    TUNING_SUBSET_FRAC = config.get('tuning_subset_frac', 0.15)
    TRAINING_SUBSET_FRAC = config.get('training_subset_frac', 1.0)
    MAX_DIVERGENCE_RATIO = config.get('max_divergence_ratio', 5.0)
    MAX_LR            = 0.001
    train_dir         = config['train_dir']
    val_dir           = config['val_dir']

    _TRAIN_DIR_TO_DATASET = {
        "/mnt/data/noxi/train/":       "noxi",
        "/mnt/data/noxi-j/train/":     "noxi-j",
        "/mnt/data/pinsoro/train-cc/": "pinsoro-cc",
        "/mnt/data/pinsoro/train-cr/": "pinsoro-cr",
    }
    dataset_name = _TRAIN_DIR_TO_DATASET.get(train_dir, slurm_job_name)

    tuner_base_dir    = os.path.expanduser(config.get('tuner_base_dir', './nn/tuner'))
    final_base_dir    = os.path.expanduser(config.get('final_base_dir', f"./final_models/nn/single_modality"))
    os.makedirs(final_base_dir, exist_ok=True)

    _logfile = [None]

    def logmsg(msg):
        print(msg)
        if _logfile[0] is not None:
            with open(_logfile[0], 'a') as f:
                f.write(msg + '\n')

    # Print job header to stdout before any modality logfile is opened
    print(f"SLURM Job ID: {slurm_job_id}, Name: {slurm_job_name}, Nodes: {slurm_nodelist}")
    print("Loaded config:")
    print(json.dumps(config, indent=2))

    for idx, modality in enumerate(modalities):
        feat_dim = modalities_dim[idx]
        modality_clean = modality.strip('.~')
        LOG_DIR = f"./nn/logs/{dataset_name}/{modality_clean}/{slurm_job_id}"
        os.makedirs(LOG_DIR, exist_ok=True)
        _logfile[0] = os.path.join(LOG_DIR, f"runlog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        logmsg(f"SLURM Job ID: {slurm_job_id}, Name: {slurm_job_name}, Nodes: {slurm_nodelist}")
        logmsg("Loaded config:")
        logmsg(json.dumps(config, indent=2))
        logmsg(f"\n=== Processing modality {modality} ({feat_dim}-D) ===")
        Xtr, ytr, Xva, yva = load_data_for_modality(train_dir, val_dir, modality, feat_dim, is_classification, dataset_config)
        logmsg(f"Loaded shapes: Xtr={Xtr.shape}, Xva={Xva.shape}, ytr={ytr.shape}, yva={yva.shape}")

        if Xtr.size == 0 or Xva.size == 0:
            logmsg(f"[Warning] No data for modality {modality}, skipping.")
            continue

        n_tune = max(10, int(len(Xtr) * TUNING_SUBSET_FRAC))
        n_train = max(10, int(len(Xtr) * TRAINING_SUBSET_FRAC))
        Xtr_sub = Xtr[:n_train]
        Xtr_tune = Xtr[:n_tune]
        
        ytr_sub = ytr[:n_train]
        ytr_tune = ytr[:n_tune]

        Xtr_sub = np.nan_to_num(Xtr_sub)
        Xva     = np.nan_to_num(Xva)
        Xtr_tune = np.nan_to_num(Xtr_tune)

        combined = np.concatenate([Xtr_sub, Xva], axis=0)
        scaler = MinMaxScaler().fit(combined)
        Xtr_norm = scaler.transform(Xtr_sub)
        Xva_norm = scaler.transform(Xva)
        Xtr_tune_norm = scaler.transform(Xtr_tune)
        Xtr_norm, ytr_sub = shuffle(Xtr_norm, ytr_sub, random_state=SEED)

        Xtr_norm = np.nan_to_num(Xtr_norm)
        Xtr_tune_norm = np.nan_to_num(Xtr_tune_norm)
        Xva_norm = np.nan_to_num(Xva_norm)
        if is_classification:
            n_heads = len(dataset_config['label_heads'])
            # Filter rows where any label is -1 (missing sentinel)
            mask_tr = np.all(ytr_sub != -1, axis=1)
            Xtr_norm = Xtr_norm[mask_tr]
            ytr_sub = ytr_sub[mask_tr]
            mask_tune = np.all(ytr_tune != -1, axis=1)
            Xtr_tune_norm = Xtr_tune_norm[mask_tune]
            ytr_tune = ytr_tune[mask_tune]
            mask_va = np.all(yva != -1, axis=1)
            Xva_norm = Xva_norm[mask_va]
            yva = yva[mask_va]
            ytr_sub = [ytr_sub[:, i] for i in range(n_heads)]
            ytr_tune = [ytr_tune[:, i] for i in range(n_heads)]
            yva = [yva[:, i] for i in range(n_heads)]

            # Compute per-sample weights for each head (class_weight unsupported
            # for multi-output models; sample_weight dict is the workaround)
            def make_sample_weights(y_list, names):
                sw = []
                for i, name in enumerate(names):
                    classes = np.unique(y_list[i])
                    w = sklearn_cw.compute_class_weight(
                        'balanced', classes=classes, y=y_list[i])
                    cw = dict(zip(classes.tolist(), w.tolist()))
                    sw.append(np.array([cw[int(l)] for l in y_list[i]], dtype=np.float32))
                return sw

            head_names = list(dataset_config['label_heads'].keys())
            sample_weights_tune = make_sample_weights(ytr_tune, head_names)
            sample_weights      = make_sample_weights(ytr_sub,  head_names)
        else:
            ytr_sub = np.nan_to_num(ytr_sub)
            ytr_tune = np.nan_to_num(ytr_tune)
            yva = np.nan_to_num(yva)

        tb_logdir = LOG_DIR
        tuner_tb_logdir = os.path.join(LOG_DIR, "tuner")
        final_tb_logdir = os.path.join(LOG_DIR, "final_train")
        best_hparams_path = os.path.join(LOG_DIR, 'best_hyperparameters.json')

        if is_classification:
            num_classes = {k: v['num_classes'] for k, v in dataset_config['label_heads'].items()}

            model = MultimediateHyperModelPinsoro(feat_dim,num_classes=num_classes)
        else:
            model = MultimediateHyperModelNoxi(feat_dim,num_classes=1)

        tuner = MyTuner(
            model,
            objective='val_loss',
            max_epochs=10 if DEBUG_MODE else 20,
            factor=3,
            directory=os.path.join(tuner_base_dir, dataset_name),
            project_name=modality_clean,
            overwrite=False,
            seed=SEED)

        early_stop_tuner = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2 if DEBUG_MODE else 3, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1 if DEBUG_MODE else 2)
        tb_callback_tuner = tf.keras.callbacks.TensorBoard(log_dir=tuner_tb_logdir)

        # ----------- LOAD OR TUNE HYPERPARAMETERS -------------
        if os.path.exists(best_hparams_path):
            logmsg(f"Loading best hyperparameters for {modality} from {best_hparams_path}")
            with open(best_hparams_path, 'r') as f:
                best_hps_dict = json.load(f)
            best_hps = kt.engine.hyperparameters.HyperParameters()
            for k, v in best_hps_dict.items():
                best_hps.values[k] = v
        else:
            tuner.search(
                Xtr_tune_norm, ytr_tune,
                epochs=10 if DEBUG_MODE else 20,
                validation_data=(Xva_norm, yva),
                callbacks=[early_stop_tuner, reduce_lr], #, tb_callback_tuner
                sample_weight=sample_weights_tune if is_classification else None,
                verbose=2
            )
            def _last_metric(trial, name):
                # Try in-memory structure first
                try:
                    obs = trial.metrics.metrics[name].observations
                    last = obs[-1]
                    val = last['value'] if isinstance(last, dict) else last.value
                    return val[0] if hasattr(val, '__getitem__') else float(val)
                except Exception:
                    pass
                # Fallback: read from trial.json on disk (handles overwrite=False)
                try:
                    trial_json = os.path.join(
                        tuner_base_dir, dataset_name, modality_clean,
                        f"trial_{trial.trial_id}", "trial.json"
                    )
                    with open(trial_json) as f:
                        t = json.load(f)
                    obs = t["metrics"]["metrics"].get(name, {}).get("observations", [])
                    return obs[-1]["value"][0] if obs else float('nan')
                except Exception:
                    return float('nan')

            completed = sorted(
                [t for t in tuner.oracle.trials.values()
                 if t.status == 'COMPLETED' and t.score is not None],
                key=lambda t: t.score
            )
            safe = []
            for t in completed:
                lr = t.hyperparameters.get('learning_rate')
                if lr is not None and lr > MAX_LR:
                    continue
                tl = _last_metric(t, 'loss')
                vl = _last_metric(t, 'val_loss')
                if not (math.isnan(tl) or math.isnan(vl) or vl <= 0):
                    if tl / vl <= MAX_DIVERGENCE_RATIO:
                        safe.append(t)
            if safe:
                logmsg(f"Divergence guard: {len(completed)-len(safe)}/{len(completed)} trials "
                       f"filtered (lr>{MAX_LR} or train/val ratio>{MAX_DIVERGENCE_RATIO}), "
                       f"selecting best of {len(safe)} safe trials.")
                best_hps = safe[0].hyperparameters
            else:
                logmsg(f"[Warning] Divergence guard: all {len(completed)} trials exceeded "
                       f"ratio > {MAX_DIVERGENCE_RATIO}, falling back to best val_loss.")
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            logmsg(f"Best hyperparameters for {modality}: {best_hps.values}")
            with open(best_hparams_path, 'w') as f:
                json.dump(best_hps.values, f, indent=2)

        best_batch_size = best_hps.get('batch_size')

        # ------------ HParams logging for TensorBoard ------------
        with tf.summary.create_file_writer(final_tb_logdir).as_default():
            hp.hparams({
                HP_UNITS1: best_hps.get('units1'),
                HP_UNITS2: best_hps.get('units2'),
                HP_UNITS3: best_hps.get('units3'),
                HP_DROPOUT: best_hps.get('dropout'),
                HP_BATCH_SIZE: best_batch_size,
                HP_LR: best_hps.get('learning_rate'),
            })

        # Final training
        model = tuner.hypermodel.build(best_hps)
        model_filepath = os.path.join(
            final_base_dir,
            f"tuner_nn_model_{slurm_job_id}.keras"
        )
        early_stop_final = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4 if DEBUG_MODE else 10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
        )
        tb_callback_final = tf.keras.callbacks.TensorBoard(log_dir=final_tb_logdir)

        model.fit(
            Xtr_norm, ytr_sub,
            validation_data=(Xva_norm, yva),
            epochs=15 if DEBUG_MODE else 100,
            batch_size=best_batch_size,
            callbacks=[early_stop_final, model_checkpoint, tb_callback_final],
            sample_weight=sample_weights if is_classification else None,
            shuffle=True,
            verbose=2
        )
        head_names = list(dataset_config['label_heads'].keys()) if is_classification else None
        evaluate_model(model, Xva_norm, yva, modality, tb_logdir, best_batch_size, logmsg, is_classification, head_names)

        tf.keras.backend.clear_session()
        gc.collect()

    # Final config log (summary)
    logmsg("\n===== RUN SUMMARY =====")
    logmsg(f"Config: {json.dumps(config, indent=2)}")
    logmsg(f"DEBUG_MODE: {DEBUG_MODE}")
    logmsg(f"TUNING_SUBSET_FRAC: {TUNING_SUBSET_FRAC}")
    logmsg(f"TRAINING_SUBSET_FRAC: {TRAINING_SUBSET_FRAC}")
    logmsg(f"MAX_DIVERGENCE_RATIO: {MAX_DIVERGENCE_RATIO}")
    logmsg(f"Train Dir: {train_dir}")
    logmsg(f"Val Dir: {val_dir}")
    logmsg("\nAll done!")

def evaluate_model(model, Xva_norm, yva, modality, tb_logdir, best_batch_size, logmsg, is_classification, head_names=None):
    # Evaluation and result saving
    ypred = model.predict(Xva_norm, batch_size=best_batch_size)
    if is_classification:
        n_heads = len(yva)
        ypred_classes = [np.argmax(ypred[i], axis=1) for i in range(n_heads)]
        accs = [accuracy_score(yva[i], ypred_classes[i]) for i in range(n_heads)]
        f1s = [f1_score(yva[i], ypred_classes[i], average='weighted') for i in range(n_heads)]
        names = head_names if head_names is not None else [f"Head {i}" for i in range(n_heads)]
        logmsg(f"Results for {modality}:")
        for i in range(n_heads):
            logmsg(f"  {names[i]} - Accuracy: {accs[i]:.4f}, F1-Score: {f1s[i]:.4f}")
        np.save(os.path.join(tb_logdir, 'yva.npy'), np.column_stack(yva))
        np.save(os.path.join(tb_logdir, 'ypred.npy'), np.column_stack(ypred))
        with open(os.path.join(tb_logdir, 'final_f1.txt'), 'w') as f:
            for f1 in f1s:
                f.write(str(f1) + '\n')
        with open(os.path.join(tb_logdir, 'final_accuracy.txt'), 'w') as f:
            for acc in accs:
                f.write(str(acc) + '\n')

    else:
        ypred_sq = np.squeeze(ypred)
        ccc = concordance_correlation_coefficient(ypred_sq, yva)
        mse = float(np.mean((ypred_sq - yva) ** 2))
        logmsg(f"CCC for {modality}: {ccc:.4f}  MSE: {mse:.4f}")
        np.save(os.path.join(tb_logdir, 'yva.npy'), yva)
        np.save(os.path.join(tb_logdir, 'ypred.npy'), ypred)
        with open(os.path.join(tb_logdir, 'final_ccc.txt'), 'w') as f:
            f.write(str(ccc) + '\n')
        with open(os.path.join(tb_logdir, 'final_mse.txt'), 'w') as f:
            f.write(str(mse) + '\n')

if __name__ == "__main__":
    main()
