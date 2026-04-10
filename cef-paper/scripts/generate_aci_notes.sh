#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=aci_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --mail-type=ALL
#SBATCH --error=/home/yathagata/batch_jobs/outs/%J.out
#SBATCH --output=/home/yathagata/batch_jobs/outs/%J.out


##############################
#           Work             #
##############################

# Change to the source directory
cd /home/yathagata/cef-translation/src/note-generation

# Run note generation
python generate_notes.py

# To use different models, modify the Config class in generate_notes.py:
# - API_URL: Change worker and port
# - MODEL_NAME: Change model name
# - SAVE_MODEL_NAME: Change output folder suffix

