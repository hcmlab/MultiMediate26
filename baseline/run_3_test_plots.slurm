#!/bin/bash

#SBATCH --job-name=testnoxibase
#SBATCH --time=2-0
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1

# Usage: sbatch run_3_test_plots.slurm configs/<dataset>/test/config1.json

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
path_to_data=$DATASET_ROOT/noxi
path_to_data_noxij=$DATASET_ROOT/noxi-j
path_to_data_pinsoro=$DATASET_ROOT/pinsoro
path_to_data_mpii=$DATASET_ROOT/mpiigroupinteraction

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Environment Target Direcory = $path_to_env_dst"
echo ""
echo "Setting up conda environment named $env_name using $path_to_env_yml ..."
echo ""

source $CONDA_ROOT/bin/activate

conda env list
echo ""
echo "Checking and setting up conda environment named $env_name using $path_to_env_yml ..."
echo ""

if [ ! -d $path_to_env_dst ]
then
    echo "Specified environment does not exist, so it is created..."
    conda env create -p $path_to_env_dst --file=$path_to_env_yml
else
    echo "Specified environment already exists."
    echo "Updating environment to newest yml-configuration..."
    conda env update -p $path_to_env_dst --file $path_to_env_yml --prune
fi

conda activate $env_name

module load cuda
module load cudnn
echo nvcc --version
echo $LD_LIBRARY_PATH
pip install matplotlib

pip install tensorflow[and-cuda]

# Syncing the datasets
rsync -havzP --stats --delete $path_to_data $LOCAL_DATA/
rsync -havzP --stats --delete $path_to_data_noxij $LOCAL_DATA/
rsync -havzP --stats --delete $path_to_data_pinsoro $LOCAL_DATA/
rsync -havzP --stats --delete $path_to_data_mpii $LOCAL_DATA/

# Derive results dir from dataset name in config path
# e.g. configs/noxi-base/test/config1.json -> results/noxi-base/full_test_predictions_plots
DATASET=$(basename $(dirname $(dirname $CONFIG_FILE)))
RESULTS_DIR="results/${DATASET}/full_test_predictions_plots"

# Run inference and plot script
python 3_TestingNN_infer_plot.py --config $CONFIG_FILE --results_dir $RESULTS_DIR
