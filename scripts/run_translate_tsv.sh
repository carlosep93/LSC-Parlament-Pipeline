#!/bin/bash
#SBATCH --job-name=translate_tsv-%j
#SBATCH -D .
#SBATCH --output=./logs/translate_tsv-%j.out
#SBATCH --error=./logs/translate_tsv-%j.err
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

source use_env_venv312.sh

python src/translate_tsv.py "$@"