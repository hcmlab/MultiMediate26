# Baseline Repository for the MultiMediate'26 Cross-cultural Multi-domain Engagement Estimation Task

In this repository, we share the baseline implementation, baseline results, as well as code for feature extraction.

## Baseline Results

We evaluated all our feature sets on all available test datasets. Models tested on NoXI, NoXI (Additional Languages),
and MPIIGroupInteraction were trained on NoXI. Models tested on NoXI-J were trained on NoXI-J.



## 2026 Regression Datasets (CCC)


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


## Classification Datasets (Cohen's Kappa)

| Feature set                 | Pinsoro-CC Task | Pinsoro-CC Social | Pinsoro-CR Task | Pinsoro-CR Social | Combined   |
| --------------------------- | --------------- | ----------------- | --------------- | ----------------- | ---------- |
| *Video*                     |                 |                   |                 |                   |            |
| &nbsp;&nbsp;OpenFace 2.0    | 0.0583          | 0.0836            | 0.5309          | 0.0944            | 0.1918     |
| &nbsp;&nbsp;OpenFace 3.0    | 0.0964          | 0.1379            | 0.5322          | 0.0863            | 0.2132     |
| &nbsp;&nbsp;OpenPose        | 0.0332          | 0.0568            | **0.7106**      | 0.0007            | 0.2003     |
| &nbsp;&nbsp;CLIP            | 0.1245          | 0.0668            | 0.6752          | 0.0540            | **0.2301** |
| &nbsp;&nbsp;DINO            | 0.0729          | 0.0579            | 0.5499          | 0.1398            | 0.2051     |
| &nbsp;&nbsp;SwinTransformer | 0.0844          | 0.0411            | 0.6449          | 0.0916            | 0.2155     |
| &nbsp;&nbsp;VideoMAE        | 0.1347          | 0.1196            | 0.4226          | **0.1724**        | 0.2123     |
| *Voice*                     |                 |                   |                 |                   |            |
| &nbsp;&nbsp;eGeMAPS v2      | 0.0247          | 0.2126            | 0.0403          | 0.0078            | 0.0713     |
| &nbsp;&nbsp;w2vBERT2        | 0.0549          | 0.0920            | 0.1199          | -0.0055           | 0.0653     |
| *Text*                      |                 |                   |                 |                   |            |
| &nbsp;&nbsp;XLM RoBERTa     | **0.1609**      | **0.2620**        | 0.0191          | 0.0009            | 0.1107     |



## Feature Extraction

Details on the feature extraction methods can be found under the feature_extraction directory in this repository.



## Baseline Implementation

A SLURM-ready deep learning pipeline for hyperparameter tuning and training of neural regressors on multimodal data.  
This repository enables flexible experimentation through external config files and supports scalable GPU-based training on HPC clusters.

---

## 📁 Directory Structure

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
├── 5_combine_results.py                   # Step 5: Combine results across datasets
│
├── run_1_tuner_noxi.slurm                 # SLURM submission scripts for step 1
├── run_1_tuner_noxi_j.slurm
├── run_1_tuner_pinsoro_cc.slurm
├── run_1_tuner_pinsoro_cr.slurm
├── run_2_full_trainer_noxi.slurm          # SLURM submission scripts for step 2
├── run_2_full_trainer_noxij.slurm
├── run_2_full_trainer_pinsoro_cc.slurm
├── run_2_full_trainer_pinsoro_cr.slurm
├── run_2.1_retrain_result_extraction.slurm
├── run_3_test_plots.slurm
├── run_4_test_fairness_allmod_psess.slurm
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
sbatch run_1_tuner_noxi.slurm configs/noxi-base/tune/config_audio_egemapsv2.json
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
sbatch run_2_full_trainer_noxi.slurm configs/noxi-base/retrain/audio.egemapsv2.stream.json
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

### Step 5 — Combine results

```sh
python 5_combine_results.py
```

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
* Adjust GPU/CPU/memory requests at the top of any `.slurm` file.
* Modify model architecture or training logic in `1_TrainingNN_tuner.py` or `2_RetrainNN_full.py`.

---

## 9. Contact

For questions, consult your cluster admin or the script author.

---

**Happy training and experimentation!**

