#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=cef_model_translation_flores
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=120
#SBATCH --gpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=traha@m42.ae
#SBATCH --error=/home/yathagata/batch_jobs/outs/%J.out
#SBATCH --output=/home/yathagata/batch_jobs/outs/%J.out

##############################
#           Work             #
##############################

cd /home/yathagata/cef-translation
python scripts/models_translation_flores.py