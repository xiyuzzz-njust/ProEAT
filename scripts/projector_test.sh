#!/bin/bash
#
#SBATCH --job-name=projector_test
#SBATCH --output=output_test.txt
#SBATCH --error=error_test.txt
#
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=rtx
#SBATCH --nodelist=aias-compute-2
#SBATCH --gres=gpu:1

srun python llava/train/init_projector.py