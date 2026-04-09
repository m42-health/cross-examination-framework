#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=cef_server
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=100:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --error=/home/yathagata/batch_jobs/outs/%J.out
#SBATCH --output=/home/yathagata/batch_jobs/outs/%J.out


##############################
#           Work             #
##############################

JOB_NAME=$1
PORT="8893"

scontrol update job=$SLURM_JOB_ID JobName=$JOB_NAME
# cd /home/nsaadi/cef-translation/src/evaluation
#python /home/nsaadi/cef-translation/src/evaluation/eval_europarl.py
cd /home/yathagata/cef-translation/src/cef_framework
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT

# Example usage:
# sbatch run_cef_server.sh "worker-7" 8893
# 
# This will:
# 1. Submit a SLURM job with the name "cef_server"
# 2. Update the job name to the first argument (e.g., "worker-7")
# 3. Start a gunicorn server on the specified port (e.g., 8893)
# 4. The server will be accessible at http://worker-7:8893/evaluate
#
# To run the evaluation after starting the server:
# python eval_flores.py --worker worker-7 --port 8893 --eval_cef --debug
