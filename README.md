# Baseline Repository for the MultiMediate'26 Cross-cultural Multi-domain Engagement Estimation Task

In this repository, we share the baseline implementation, baseline results, as well as code for feature extraction.

## Baseline Results

We evaluated all our feature sets on all available test datasets. Models tested on NoXI, NoXI (Additional Languages),
and MPIIGroupInteraction were trained on NoXI. Models tested on NoXI-J were trained on NoXI-J.

## Test Set Results (CCC)

We use the best performing model on the test set for our baseline performance.

| Feature set                 | NOXI       | NOXI (Add.) | NOXI-J     | MPIIGI     | Combined   |
| --------------------------- | ---------- | ----------- | ---------- | ---------- | ---------- |
| *Video*                     |            |             |            |            |            |
| &nbsp;&nbsp;OpenFace 2.0    | 0.1070     | 0.1948      | 0.1556     | 0.1159     | 0.1433     |
| &nbsp;&nbsp;OpenFace 3.0    | 0.2192     | 0.1081      | 0.2701     | 0.0880     | 0.1714     |
| &nbsp;&nbsp;OpenPose        | 0.4801     | 0.4272      | 0.2687     | 0.0854     | 0.3153     |
| &nbsp;&nbsp;CLIP            | 0.4350     | 0.3652      | 0.2206     | -0.0099    | 0.2527     |
| &nbsp;&nbsp;DINO            | 0.5290     | 0.4592      | 0.1071     | 0.0068     | 0.2755     |
| &nbsp;&nbsp;SwinTransformer | 0.5406     | 0.4649      | 0.2627     | -0.0482    | 0.3050     |
| &nbsp;&nbsp;VideoMAE        | 0.6065     | 0.5602      | 0.2168     | 0.0173     | 0.3502     |
| *Voice*                     |            |             |            |            |            |
| &nbsp;&nbsp;eGeMAPS v2      | **0.5535** | **0.4909**  | **0.3132** | **0.4539** | **0.4529** |
| &nbsp;&nbsp;w2vBERT2        | 0.5986     | 0.4710      | 0.3178     | 0.0540     | 0.3604     |
| *Text*                      |            |             |            |            |            |
| &nbsp;&nbsp;XLM RoBERTa     | 0.3916     | 0.2698      | 0.2553     | 0.0090     | 0.2314     |

As eGeMAPs v2 achieves the best combined performance, please at least always compare your approach to this baseline
method.


## Test Set Results (Cohen's Kappa)

We use the best performing model on each pinsoro subset for our classification task baseline.


| Feature set                 | Pinsoro-CC Social | Pinsoro-CC Task | Pinsoro-CR Social | Pinsoro-CR Task | Combined   |
| --------------------------- | ----------------- | --------------- | ----------------- | --------------- | ---------- |
| *Video*                     |                   |                 |                   |                 |            |
| &nbsp;&nbsp;OpenFace 2.0    | 0.0776            | 0.0583          | -0.0118           | 0.2463      | 0.0926     |
| &nbsp;&nbsp;OpenFace 3.0    | 0.1250            | 0.0980          | 0.0164            | 0.1715          | 0.1027     |
| &nbsp;&nbsp;OpenPose        | 0.0615            | 0.0332          | -0.0261           | 0.2090          | 0.0694     |
| &nbsp;&nbsp;CLIP            | 0.0933            | 0.1245          | -0.0216           | 0.1601          | 0.0891     |
| &nbsp;&nbsp;DINO            | 0.0752            | 0.0729          | 0.0121            | 0.0826          | 0.0607     |
| &nbsp;&nbsp;SwinTransformer | 0.0905            | 0.0844          | **0.0901**        | **0.2426**          | **0.1269** |
| &nbsp;&nbsp;VideoMAE        | 0.1436            | 0.1347          | -0.0191           | 0.0465          | 0.0764     |
| *Voice*                     |                   |                 |                   |                 |            |
| &nbsp;&nbsp;eGeMAPS v2      | 0.1320            | 0.0207          | -0.0188           | -0.0262         | 0.0269     |
| &nbsp;&nbsp;w2vBERT2        | 0.1089            | 0.0549          | 0.0032            | 0.0262          | 0.0483     |
| *Text*                      |                   |                 |                   |                 |            |
| &nbsp;&nbsp;XLM RoBERTa     | **0.1438**        | **0.1609**      | -0.0265           | 0.1044          | 0.0956     |



## Test Set Results (CDD_G)

Global Conditional Demographic Disparity (CDD) for Gender (CDD_G) on the pinsoro subsets, reported per engagement target.
Values closer to 0 indicate less disparity; the sign indicates the direction of the disparity. While it shows higher values than for the regression task (as seen in MultiMediate'25), the CDD_G is calculated using categorical label predictions and should be interpreted with caution.

| Feature set                 | Pinsoro-CC Social | Pinsoro-CC Task | Pinsoro-CR Social | Pinsoro-CR Task |
| --------------------------- | ----------------- | --------------- | ----------------- | --------------- |
| *Video*                     |                   |                 |                   |                 |
| &nbsp;&nbsp;OpenFace 2.0    | 0.507752          | -0.670318       | -0.263529         | 0.464052        |
| &nbsp;&nbsp;OpenFace 3.0    | 0.228690          | -0.524587       | -0.756035         | 0.507905        |
| &nbsp;&nbsp;OpenPose        | 0.369051          | -0.175057       | -1.135412         | -0.532956       |
| &nbsp;&nbsp;CLIP            | -0.642862         | 0.793529        | 0.225779          | 0.590486        |
| &nbsp;&nbsp;DINO            | -1.221760         | 0.441415        | -0.457306         | 0.745108        |
| &nbsp;&nbsp;SwinTransformer | -0.694572         | 0.357152        | -0.590268         | 0.567694        |
| &nbsp;&nbsp;VideoMAE        | -0.844042         | 0.195072        | 0.532718          | 0.483792        |
| *Voice*                     |                   |                 |                   |                 |
| &nbsp;&nbsp;eGeMAPS v2      | 0.040014          | -0.130496       | 0.251175          | 0.039863        |
| &nbsp;&nbsp;w2vBERT2        | 0.337442          | -0.330527       | -0.224748         | 0.006976        |
| *Text*                      |                   |                 |                   |                 |
| &nbsp;&nbsp;XLM RoBERTa     | 0.013098          | -0.160921       | -0.080272         | 0.235656        |

CDD for Language was 0.0 for the only language group present (English) across all feature sets and both subsets.


## Feature Extraction

Details on the feature extraction methods can be found under the feature_extraction directory in this repository.



## Baseline Implementation

A SLURM-ready deep learning pipeline for hyperparameter tuning and training of neural regressors on multimodal data.  
This repository enables flexible experimentation through external config files and supports scalable GPU-based training on HPC clusters.

---

## Directory Structure

````
baseline/
├── env_gpu.yml                            # Conda environment file
├── results.md                             # Summary of results
│
├── 1_TrainingNN_tuner.py                  # Step 1: Hyperparameter tuning
├── 1.1_TuneResultExtraction.py            # Step 1.1: Extract tuning results
├── 2_RetrainNN_full.py                    # Step 2: Full retrain with best hyperparameters
├── 2.1_RetrainResultExtraction.py         # Step 2.1: Extract retrain results
├── 3_TestingNN_infer_plot.py              # Step 3: Inference and prediction plots
├── 4_TestingNN_fairness_per_session.py    # Step 4: Per-session fairness evaluation
├── 5_CreateResulttable.py                 # Step 5: Recompute metrics from predictions/GT and generate markdown result table
│
├── run_1_tuner.slurm                      # SLURM script for step 1 (all datasets)
├── run_2_full_trainer.slurm               # SLURM script for step 2 (all datasets)
├── run_3_test_plots.slurm                 # SLURM script for step 3
├── run_4_test_fairness_allmod_psess.slurm # SLURM script for step 4
├── run_all_1.sh ... run_all_4.sh          # Convenience scripts to submit all jobs
│
├── configs/                               # Experiment configuration files (JSON)
│   ├── noxi-base/
│   │   ├── dataset-config.json
│   │   ├── tune/                          # Per-modality tuning configs
│   │   ├── retrain/                       # Per-modality retrain configs
│   │   ├── full_train/                    # Full training configs (config1–4.json)
│   │   └── test/
│   ├── noxi-j/                            # Same structure as noxi-base
│   ├── pinsoro-cc/                        # Same structure as noxi-base
│   ├── pinsoro-cr/                        # Same structure as noxi-base
│   ├── mpiigroupinteraction/
│   │   ├── dataset-config.json
│   │   └── test/
│   └── test-additional/
│       ├── dataset-config.json
│       └── test/
│
├── final_models/                          # Tuned model checkpoints (.keras) per dataset
│
└── results/                               # Output folder for results

````


---

## 1. Setup

### A. Prerequisites

- Python 3.8+
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- A Slurm-managed HPC cluster with GPU nodes (A40 in this example)
- Required Python packages (handled by `env_gpu.yml`)
- Your data directory structure (see config files)

### B. Conda Environment

Each SLURM script automatically creates or updates the conda environment from `env_gpu.yml` on first run.
Edit the `env_name` variable at the top of any `.slurm` file to change the environment name.

### C. Cluster paths

All four `.slurm` files share the same three configuration variables at the top. Set these to match your HPC environment before submitting jobs:

```bash
CONDA_ROOT=/mnt/data/miniconda3            # Root of your Miniconda/Anaconda installation
DATASET_ROOT=/mnt/datasets/MultiMediate/2026/public  # Where the raw datasets are stored
LOCAL_DATA=/mnt/data                       # Fast local scratch space for rsync'd data
```

The dataset is inferred automatically from the config path (e.g. `configs/noxi-base/...` → syncs the `noxi` directory). Override SLURM resource requests directly on the `sbatch` command line if needed:

```sh
sbatch --mem=64G run_1_tuner.slurm configs/noxi-j/tune/config_clip.json
```

---

## 2. Configuration

All experiment settings are defined in JSON files under `configs/<dataset>/`.
There are three types of config, used by different pipeline steps:

### Tune config (`configs/<dataset>/tune/config_<modality>.json`)

Used by step 1. One file per modality per dataset.

```json
{
  "modalities": [".audio.egemapsv2.stream~"],
  "modalities_dim": [88],
  "debug_mode": false,
  "tuning_subset_frac": 1.0,
  "training_subset_frac": 1.0,
  "train_dir": "/mnt/data/noxi/train/",
  "val_dir": "/mnt/data/noxi/val/",
  "tuner_base_dir": "./nn/tuner/tuner_models_regression",
  "final_base_dir": "./final_models/nn/single_modality"
}
```

### Retrain config (`configs/<dataset>/retrain/<modality>.json`)

Auto-generated by step 1.1 (`1.1_TuneResultExtraction.py`). Contains the best hyperparameters found during tuning.

```json
{
  "modalities": [".audio.egemapsv2.stream~"],
  "modalities_dim": [88],
  "train_dir": "/mnt/data/noxi/train/",
  "val_dir": "/mnt/data/noxi/val/",
  "full_base_dir": "./results/full_models",
  "best_hyperparameters": {
    ".audio.egemapsv2.stream~": {
      "units1": 360, "units2": 384, "units3": 128,
      "dropout": 0.35, "learning_rate": 0.001, "batch_size": 32
    }
  }
}
```

### Test config (`configs/<dataset>/test/config1.json`)

Used by steps 3 and 4. Lists all modalities and points to the test split and ground-truth labels.

```json
{
  "modalities": [".audio.egemapsv2.stream~", ".clip.stream~", "..."],
  "modalities_dim": [88, 512, "..."],
  "train_dir": "/mnt/data/noxi/train/",
  "val_dir": "/mnt/data/noxi/val/",
  "test_dir": "/mnt/data/noxi/test-base/",
  "full_base_dir": "./results/full_models",
  "engagement_gt": "./engagement-mm26-test/noxi/test-base"
}
```

### Config parameters

| Parameter | Description |
|---|---|
| `modalities` | List of feature modalities to use |
| `modalities_dim` | Feature dimensions matching each modality |
| `debug_mode` | Set `true` for fast test runs (reduced epochs) |
| `tuning_subset_frac` | Fraction of training data used for hyperparameter search |
| `training_subset_frac` | Fraction of training data used during tuning’s final fit |
| `train_dir` / `val_dir` / `test_dir` | Data directories for each split |
| `tuner_base_dir` | Where KerasTuner checkpoints and logs are stored |
| `final_base_dir` | Where retrained full models are saved |
| `best_hyperparameters` | Best hyperparameters per modality (auto-filled by step 1.1) |
| `engagement_gt` | Path to ground-truth engagement labels for testing |

---

## 3. Running the Pipeline

The pipeline runs in 5 steps. Use the `run_all_*.sh` scripts to submit all jobs for all datasets at once, or submit individual SLURM jobs manually. Without test set annotations, only the first 2 steps can be run.

### Step 1 — Hyperparameter tuning

```sh
# All datasets at once:
bash run_all_1.sh

# Or a single job:
sbatch run_1_tuner.slurm configs/noxi-base/tune/config_audio_egemapsv2.json
```

### Step 1.1 — Extract tuning results

Reads tuner output and writes retrain configs to `configs/<dataset>/retrain/`.

```sh
python 1.1_TuneResultExtraction.py
```

### Step 2 — Full retrain with best hyperparameters

```sh
# All datasets at once:
bash run_all_2.sh

# Or a single job:
sbatch run_2_full_trainer.slurm configs/noxi-base/retrain/audio.egemapsv2.stream.json
```

### Step 2.1 — Extract retrain results

```sh
python 2.1_RetrainResultExtraction.py
```

### Step 3 — Inference and prediction plots

```sh
bash run_all_3.sh

# Or a single job:
sbatch run_3_test_plots.slurm configs/noxi-base/test/config1.json
```

### Step 4 — Per-session fairness evaluation

```sh
bash run_all_4.sh

# Or a single job:
sbatch run_4_test_fairness_allmod_psess.slurm configs/noxi-base/test/config1.json
```

### Step 5 — Create result table

Recomputes CCC (regression) and Cohen's Kappa / Accuracy (classification) metrics directly from prediction and ground-truth CSV files, and writes a markdown result table.

```sh
python 5_CreateResulttable.py \
    --results_dir results/2026 \
    --gt_root ./engagement-mm26-test \
    --output ../results.md
```

All arguments are optional and default to the values shown above.

---

## 4. Output and Results

* **SLURM logs:** `logs/<jobname>-<jobid>.log`

* **Tuned model checkpoints:** `final_models/<dataset>/nn/single_modality/*.keras`
  (path set by `final_base_dir` in tune configs)

* **Retrained full models:** `results/full_models/retrained_full_nn_model_<modality>.keras`
  (path set by `full_base_dir` in retrain/test configs)

* **CCC / MSE scores per modality:** `results/full_models/final_ccc_<modality>.txt` and `final_mse_<modality>.txt`

* **Test predictions and plots:** `results/<dataset>/` — scatter plots, residuals, per-session metrics

---

## 5. Monitoring Progress

```sh
# Check job status
squeue -u $USER

# Live log output
tail -f logs/<jobname>-<jobid>.log
```

---

## 6. Loading Saved Models

```python
import tensorflow as tf
model = tf.keras.models.load_model(
    ‘results/full_models/retrained_full_nn_model_audio.egemapsv2.stream.keras’
)
```

---

## 7. Troubleshooting

* If `module: command not found`, check your cluster’s module system documentation.
* If jobs fail, check resource requests and verify data/config paths in the JSON files.
* Check `logs/` for Python errors or data shape mismatches.

---

## 8. Customization

* Add tune configs in `configs/<dataset>/tune/` for new modalities or datasets.
* Adjust GPU/CPU/memory requests at the top of any `.slurm` file, or override per-job with e.g. `sbatch --mem=64G run_1_tuner.slurm ...`.
* To add a new dataset, add a `case` entry in `run_1_tuner.slurm` and `run_2_full_trainer.slurm` that maps the new dataset directory name to its source path:
  ```bash
  my-new-dataset) path_to_data="$DATASET_ROOT/my-new-dataset" ;;
  ```
* Modify model architecture or training logic in `1_TrainingNN_tuner.py` or `2_RetrainNN_full.py`.

---

## 9. Contact

For questions, consult your cluster admin or the script author.

---

**Happy training and experimentation!**

