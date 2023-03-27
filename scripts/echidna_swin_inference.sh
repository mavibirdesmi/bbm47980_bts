#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --nodes 1
#SBATCH --output=/home/%u/batch_outputs/%j.out
#SBATCH --error=/home/%u/batch_outputs/%j.error
#SBATCH --gres gpu:v100:1

source "$(poetry env info --path)/bin/activate"

srun python -m bts.swinunetr.inference \
    --data-dir /home/vedatb/senior-project/data/btsed_dataset \
    --model model-checkpoints/model.pt \
    --output-dir predictions
