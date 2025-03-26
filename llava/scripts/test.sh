#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=output1.txt
#SBATCH --error=error1.txt
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --nodelist=aias-compute-1
#SBATCH --gres=gpu:1

srun python /home/zengxiyu24/LLaVA/llava/train/attention.py