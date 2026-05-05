#!/bin/bash
#SBATCH --job-name=restore_casing-%j
#SBATCH -D .
#SBATCH --output=./logs/restore_casing-%j.out
#SBATCH --error=./logs/restore_casing-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=8:00:00
#SBATCH -N1
#SBATCH --account <your_slurm_account>
#SBATCH --qos <your_slurm_qos>

module load python/3.10.15
source env3.10/bin/activate

python3 -u src/restore_casing_punctuation.py
