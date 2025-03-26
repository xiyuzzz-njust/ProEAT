#!/bin/bash
#
#SBATCH --job-name=projector_solo_eps8_50
#SBATCH --output=output_solo3.txt
#SBATCH --error=error_solo3.txt
#
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a800
#SBATCH --qos=a800
#SBATCH --gres=gpu:1

srun python llava/train/projector_solo.py