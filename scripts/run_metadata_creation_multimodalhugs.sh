#!/bin/bash
#SBATCH --job-name=metadata_creation_multimodalhugs-%j
#SBATCH -D .
#SBATCH --output=./logs/run_metadata_creation_multimodalhugs-%j.out
#SBATCH --error=./logs/run_metadata_creation_multimodalhugs-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -N1
#SBATCH --account <your_slurm_account>
#SBATCH --qos <your_slurm_qos>

source use_env_venv312.sh

python3 src/segments_to_train_dataset.py "$@"
