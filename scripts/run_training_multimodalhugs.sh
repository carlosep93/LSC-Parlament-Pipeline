#!/bin/bash
#SBATCH --job-name=training_multimodalhugs-%j
#SBATCH -D .
#SBATCH --output=./logs/training_multimodalhugs-%j.out
#SBATCH --error=./logs/training_multimodalhugs-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
##SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH -N1
##SBATCH --account ehpc206
##SBATCH --qos=acc_ehpc
##SBATCH --qos=acc_debug
#SBATCH --qos=acc_bscaii
#SBATCH --account bsc88

# while true; do
#   echo "###################"
#   date
#   top -Hbn 1
#   sleep 20
#   echo "####################"
# done > monitoring-$SLURM_JOBID.txt &

# should be run with an acc node
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

#multimodalhugs-setup --modality "pose2text" \
 #   --config_path $CONFIG_PATH

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
    --report_to none # Si fas servir wandb o similar treu aixo


#output_dir="$2"/"$SLURM_JOB_ID"/"$1"

#multimodalhugs-setup --modality pose2text --config_path configs/config_"$1"_gerard.yaml
#multimodalhugs-train --task translation --config_path configs/config_"$1"_gerard.yaml --output_dir $output_dir --setup_path "$3"

