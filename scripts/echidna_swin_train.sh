#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --nodes 1
#SBATCH --output=/home/%u/batch_outputs/%j.out
#SBATCH --error=/home/%u/batch_outputs/%j.error
#SBATCH --gres gpu:v100:3

set -e

source "$(poetry env info --path)/bin/activate"

export WANDB__SERVICE_WAIT=300

srun python -m bts.swinunetr.train \
    --data-dir /home/vedatb/senior-project/data/btsed_dataset \
    --output model-checkpoints

srun python -m bts.swinunetr.inference \
    --data-dir /home/vedatb/senior-project/data/btsed_dataset \
    --model model-checkpoints/model.pt \
    --output-dir predictions
