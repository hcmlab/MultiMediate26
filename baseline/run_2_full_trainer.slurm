#!/bin/bash

#SBATCH --job-name=mm-retrain
#SBATCH --time=2-0
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1

# Usage: sbatch run_2_full_trainer.slurm configs/<dataset>/retrain/<config>.json
# Override resources on the command line if needed, e.g.:
#   sbatch --mem=64G run_2_full_trainer.slurm configs/noxi-j/retrain/clip.stream.json

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <config_file.json>"
    exit 1
fi

CONFIG_FILE=$1

# Adjust these three variables to match your HPC environment
CONDA_ROOT=/mnt/data/miniconda3
DATASET_ROOT=/mnt/datasets/MultiMediate/2026/public
LOCAL_DATA=/mnt/data

env_name="mmslurm63"
path_to_project="$(pwd)"
path_to_env_yml=$path_to_project/env_gpu.yml
path_to_env_dst=$CONDA_ROOT/envs/$env_name

# Derive dataset name from config path: configs/<dataset>/retrain/<config>.json
DATASET=$(basename $(dirname $(dirname $CONFIG_FILE)))

# Map dataset name to its source directory under DATASET_ROOT
case "$DATASET" in
    noxi-base)             path_to_data="$DATASET_ROOT/noxi" ;;
    noxi-j)                path_to_data="$DATASET_ROOT/noxi-j" ;;
    pinsoro-cc|pinsoro-cr) path_to_data="$DATASET_ROOT/pinsoro" ;;
    mpiigroupinteraction)  path_to_data="$DATASET_ROOT/mpiigroupinteraction" ;;
    *) echo "Unknown dataset '$DATASET' in config path. Set path_to_data manually."; exit 1 ;;
esac

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Dataset           = $DATASET  ->  $path_to_data"
echo "Environment       = $path_to_env_dst"
echo ""

source $CONDA_ROOT/bin/activate

conda env list
echo ""
echo "Checking and setting up conda environment $env_name ..."

if [ ! -d $path_to_env_dst ]; then
    echo "Environment does not exist, creating..."
    conda env create -p $path_to_env_dst --file=$path_to_env_yml
else
    echo "Environment exists, updating..."
    conda env update -p $path_to_env_dst --file $path_to_env_yml --prune
fi

conda activate $env_name

module load cuda
module load cudnn
echo nvcc --version
echo $LD_LIBRARY_PATH

pip install tensorflow[and-cuda]

# Sync dataset to local fast storage
rsync -havzP --stats --delete $path_to_data $LOCAL_DATA/

# Run full retraining with best hyperparameters
python 2_RetrainNN_full.py --config $CONFIG_FILE
