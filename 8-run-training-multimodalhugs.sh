#!/bin/bash

export WANDB_DISABLED='true'
export HF_DATASETS_OFFLINE=1

sbatch scripts/run_training_multimodalhugs.sh \
    configs/config_ca.yaml \
    metadata/ \
    ca
