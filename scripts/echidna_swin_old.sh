#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --nodes 1
#SBATCH --output=/home/%u/batch_outputs/%j.out
#SBATCH --error=/home/%u/batch_outputs/%j.error
#SBATCH --gres gpu:v100:2

set -e

source "$(poetry env info --path)/bin/activate"

export WANDB__SERVICE_WAIT=300

CHECKPOINT_PATH="old-model-checkpoints"

srun python -m bts.swinunetr.train-old \
    --data-dir /home/vedatb/senior-project/data/btsed_dataset \
    --output $CHECKPOINT_PATH


srun python -m bts.swinunetr.inference \
    --data-dir /home/vedatb/senior-project/data/btsed_dataset \
    --model $CHECKPOINT_PATH/model.pt \
    --output-dir predictions-05
