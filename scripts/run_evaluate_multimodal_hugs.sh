#!/bin/bash
#SBATCH --job-name=evaluate_multimodalhugs-%j
#SBATCH -D .
#SBATCH --output=./logs/evaluate_multimodalhugs-%j.out
#SBATCH --error=./logs/evaluate_multimodalhugs-%j.err
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
source use_env_venv312.sh

CONFIG_PATH=$1
SPLIT_PATH=$2
LANGUAGE=$3
OUTPUT_PATH=$4
CKPT_PATH=$5
METRIC_NAME=$6

PROCESSOR_PATH="$SPLIT_PATH"/"$LANGUAGE"/setup/video2text_processor
MODEL_PATH="$SPLIT_PATH"/"$LANGUAGE"/setup/model
DATASET_DIR="$SPLIT_PATH"/"$LANGUAGE"/setup/datasets/parlament_"$LANGUAGE"_video/

multimodalhugs-setup --modality "video2text" \
    --config_path $CONFIG_PATH

multimodalhugs-generate \
    --task "translation" \
    --metric_name $METRIC_NAME \
    --generate_output_dir $OUTPUT_PATH/$METRIC_NAME \
    --model_name_or_path $CKPT_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --dataset_dir $DATASET_DIR \
    --config_path $CONFIG_PATH
