#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=qwen-medic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=1000GB
#SBATCH --time=400:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=traha@m42.ae
#SBATCH --error=/home/yathagata/batch_jobs/outs/%J.out
#SBATCH --output=/home/yathagata/batch_jobs/outs/%J.out

##############################
#           Work             #
##############################
# Setup
export WANDB__SERVICE_WAIT=300
export PYTHONFAULTHANDLER=1
ip addr
pythonExec=$1

# model_name="/models_llm/Qwen2.5-72B-Instruct"
# model_name="CohereLabs/aya-expanse-32b"
# model_name="/models_llm/Llama-3.1-70B-Instruct"
# model_name="Qwen/Qwen3-235B-A22B"
# model_name="Qwen/Qwen3-14B"
# model_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
# model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct"
# model_name="Qwen/Qwen3-235B-A22B-Instruct-2507"
# model_name="meta-llama/Llama-3.3-70B-Instruct"


# model_name="google/gemma-3-27b-it"
# model_name="google/gemma-3-4b-it"
# model_name="google/gemma-3-270m-it"


# model_name="openai/gpt-oss-120b"
model_name="deepseek-ai/DeepSeek-V3.1"
# model_name="meta-llama/Llama-3.3-70B-Instruct"
# model_name="Qwen/Qwen3-235B-A22B-Instruct-2507"

# model_name="Qwen/Qwen3-4B"
# model_name="Qwen/Qwen3-1.7B"
# model_name="meta-llama/Llama-3.2-3B-Instruct"
# model_name="meta-llama/Llama-3.1-8B-Instruct"
# model_name="openai/gpt-oss-20b"


# python -m sglang.launch_server \
#   --model-path $model_name \
#   --mem-fraction-static 0.85 \
#   --context-length 32768 \
#   --port 8892 \
#   --host 0.0.0.0 \
#   --trust-remote-code \
#   --tp 4

python -m vllm.entrypoints.openai.api_server \
    --model $model_name \
    --port 8892 \
    --host 0.0.0.0 \
    --trust-remote-code  \
    --tensor-parallel-size 8 