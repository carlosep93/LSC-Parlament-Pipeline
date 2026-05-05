#!/bin/bash
#SBATCH --job-name=test_checks-%j
#SBATCH -D .
#SBATCH --output=./logs/test_checks-%j.out
#SBATCH --error=./logs/test_checks-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --account <your_slurm_account>
#SBATCH --qos <your_slurm_qos>

module load nvidia-hpc-sdk ffmpeg

source use_env_venv312.sh

python src/check_test_files.py
