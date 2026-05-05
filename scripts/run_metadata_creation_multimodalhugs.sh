#!/bin/bash
#SBATCH --job-name=metadata_creation_multimodalhugs-%j
#SBATCH -D .
#SBATCH --output=./logs/run_metadata_creation_multimodalhugs-%j.out
#SBATCH --error=./logs/run_metadata_creation_multimodalhugs-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
##SBATCH --gres=gpu:0
#SBATCH --qos gp_debug
#SBATCH -N1
#SBATCH --account bsc88

module load python/3.10.15
source env3.10/bin/activate

python3 src/segments_to_train_dataset.py "$@"