#!/bin/bash
#SBATCH --job-name=training_multimodalhugs-%j
#SBATCH -D .
#SBATCH --output=./logs/training_multimodalhugs-%j.out
#SBATCH --error=./logs/training_multimodalhugs-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --account <your_slurm_account>
#SBATCH --qos <your_slurm_qos>

source use_env_venv312.sh

CONFIG_PATH=$1
SPLIT_PATH=$2
LANGUAGE=$3

OUTPUT_PATH=trained/"$SLURM_JOB_ID"_parlament_"$LANGUAGE"_video
PROCESSOR_PATH="$SPLIT_PATH"/"$LANGUAGE"/setup/video2text_processor
MODEL_PATH="$SPLIT_PATH"/"$LANGUAGE"/setup/model
DATASET_DIR="$SPLIT_PATH"/"$LANGUAGE"/setup/datasets/parlament_"$LANGUAGE"_video

# ----------------------------------------------------------
# 1. Setup the Model and datasets
# ----------------------------------------------------------

multimodalhugs-setup --modality "video2text" \
    --config_path $CONFIG_PATH

# ----------------------------------------------------------
# 2. Train the Model
# ----------------------------------------------------------

multimodalhugs-train --task "seq2seq" \
    --config_path $CONFIG_PATH \
    --model_name_or_path $MODEL_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --dataset_dir $DATASET_DIR \
    --output_dir $OUTPUT_PATH \
    --visualize_prediction_prob 0.3 \
    --report_to none # Remove this if you use wandb or a similar logging tool
