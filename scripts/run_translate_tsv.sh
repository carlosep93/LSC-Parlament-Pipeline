#!/bin/bash
#SBATCH --job-name=translate_tsv-%j
#SBATCH -D .
#SBATCH --output=./logs/translate_tsv-%j.out
#SBATCH --error=./logs/translate_tsv-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --account <your_slurm_account>
#SBATCH --qos <your_slurm_qos>

source use_env_venv312.sh

python src/translate_tsv.py "$@"
