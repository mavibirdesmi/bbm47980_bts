#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --nodes 1
#SBATCH --output=/home/%u/batch_outputs/%j.out
#SBATCH --error=/home/%u/batch_outputs/%j.error
#SBATCH --gres gpu:v100:2

source "$(poetry env info --path)/bin/activate"

srun python -m bts.swinunetr.train \
    --data-dir /home/vedatb/senior-project/data/btsed_dataset \
    --output model-checkpoints
