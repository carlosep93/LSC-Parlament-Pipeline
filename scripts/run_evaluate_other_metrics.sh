#!/bin/bash
#SBATCH --job-name=evaluate_other_multimodalhugs-%j
#SBATCH -D .
#SBATCH --output=./logs/evaluate_other_multimodalhugs-%j.out
#SBATCH --error=./logs/evaluate_other_multimodalhugs-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --account <your_slurm_account>
#SBATCH --qos <your_slurm_qos>

source use_env_venv_evaluation.sh

python src/evaluate_metrics.py "$@"
