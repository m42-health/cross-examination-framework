#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=qwen3_translation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00

##############################
#         Job Start          #
##############################

echo "Job started on $(hostname) at $(date)"

# Activate your conda or virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name  # <-- Replace with the env that has transformers + datasets + requests

# Run the translation script
python /home/nsaadi/cef-translation/scripts/qwen3-translation.py

echo "Job finished at $(date)"  

