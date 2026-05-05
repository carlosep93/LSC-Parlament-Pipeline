#!/bin/bash
#SBATCH --job-name=evaluate_other_multimodalhugs-%j
#SBATCH -D .
#SBATCH --output=./logs/evaluate_other_multimodalhugs-%j.out
#SBATCH --error=./logs/evaluate_other_multimodalhugs-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
##SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -N1
##SBATCH --account ehpc206
##SBATCH --qos=acc_ehpc
#SBATCH --qos=acc_debug
##SBATCH --qos=acc_bscls
#SBATCH --account bsc88

# should be run with an acc node
source use_env_venv_evaluation.sh

python src/evaluate_metrics.py "$@"