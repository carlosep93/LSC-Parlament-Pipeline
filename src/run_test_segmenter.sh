#!/bin/bash
#SBATCH --job-name=translate_xml
#SBATCH --output=../slurm_logs/test_segmenter_%j.out
#SBATCH --error=../slurm_logs/test_segmenter_%j.err
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=20
#SBATCH --qos gp_debug
#SBATCH -N1
#SBATCH --account bsc88
##SBATCH --qos=acc_bscls


source load_ffmpeg.sh
source env3.10/bin/activate

python test_segmenter.py

