#!/bin/bash

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=qwen3_vllm
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=100:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --error=/home/yathagata/batch_jobs/outs/%J.out
#SBATCH --output=/home/yathagata/batch_jobs/outs/%J.out


##############################
#           Work             #
##############################


# Usage: ./launch_vllm_qwen3.sh 1

python /home/yathagata/cef-translation/src/evaluation/nllb_translator.py