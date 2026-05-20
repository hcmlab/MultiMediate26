#!/bin/bash

#SBATCH --job-name=mmpincctun
#SBATCH --time=2-0
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
####SBATCH --gres=gpu:a40:1

# Usage: sbatch run_1_tuner_pinsoro_cc.slurm configs/pinsoro-cc/tune/config1.json

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <config_file.json>"
    exit 1
fi

CONFIG_FILE=$1

env_name="mmslurm63"
path_to_project="$(pwd)"
path_to_env_yml=$path_to_project/env_gpu.yml
path_to_env_dst=/mnt/data/miniconda3/envs/$env_name
path_to_data=/mnt/datasets/MultiMediate/2026/public/pinsoro

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "Environment Target Direcory = $path_to_env_dst"
echo ""
echo "Setting up conda environment named $env_name using $path_to_env_yml ..."
echo ""

source /mnt/data/miniconda3/bin/activate

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

pip install tensorflow[and-cuda]

# Syncing the datasets
rsync -havzP --stats --delete $path_to_data /mnt/data/

# Run your tuning script with the passed config file
python 1_TrainingNN_tuner.py --config $CONFIG_FILE
