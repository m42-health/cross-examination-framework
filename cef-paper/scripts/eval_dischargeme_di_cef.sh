#!/bin/bash

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=eval_di_cef_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=100:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --error=/home/yathagata/batch_jobs/outs/%J.out
#SBATCH --output=/home/yathagata/batch_jobs/outs/%J.out


##############################
#           Work             #
##############################

mkdir -p /data/cef/dischargeme/logs

MODELS=(
    "DeepSeek-V3.1"
    "Llama3-OpenBioLLM-70B"
    "MediPhi-Instruct"
    "Qwen2.5-72B-Instruct"
    "Qwen3-235B-A22B-Thinking-2507"
    "gpt-oss-120b"
    "gpt-oss-20b"
    "llama3-70b-instruct"
    "m42-health_Llama3-Med42-70B"
    "m42-health_Llama3-Med42-8B"
    "meta-llama_Llama-4-Maverick-17B-128E-Instruct"
    "microsoft_Phi-3.5-mini-instruct"
    "microsoft_phi-4"
    "mistralai_Mistral-7B-Instruct-v0.1"
    "mistralai_Mistral-Large-3-675B-Instruct-2512"
    "moonshotai_Kimi-K2-Thinking"
    "nvidia_Llama-3.1-Nemotron-70B-Instruct-HF"
    "qwen3-8B"
)

BASE_DIR="/data/cef/dischargeme/dischargeme"
REFERENCE_PATH="/data/cef/dischargeme/gt/discharge-me/1.3"
RESULTS_PATH="/data/cef/dischargeme/results_di_cef"

cd /home/yathagata/cef-translation/src/evaluation

echo "========================================="
echo "Starting DI CEF evaluation for all models"
echo "Job ID: $SLURM_JOB_ID"
echo "Total models: ${#MODELS[@]}"
echo "========================================="
echo ""

# Models that use .json format instead of _tmp.jsonl
JSON_MODELS=("gpt-oss-120b" "gpt-oss-20b" "llama3-70b-instruct" "qwen3-8B")

# Loop through all models
for MODEL_NAME in "${JSON_MODELS[@]}"; do
    echo "========================================="
    echo "DI CEF Evaluation: ${MODEL_NAME}"
    echo "Time: $(date)"
    echo "========================================="
    
    # Check if model is in JSON_MODELS array
    if [[ " ${JSON_MODELS[*]} " =~ " ${MODEL_NAME} " ]]; then
        PREDICTIONS_PATH="${BASE_DIR}/${MODEL_NAME}/predictions/${MODEL_NAME}_raw_di_responses.json"
    else
        PREDICTIONS_PATH="${BASE_DIR}/${MODEL_NAME}/predictions/${MODEL_NAME}_raw_di_responses_tmp.jsonl"
    fi
    
    if [ ! -f "${PREDICTIONS_PATH}" ]; then
        echo "WARNING: Predictions file not found: ${PREDICTIONS_PATH}"
        echo "Skipping ${MODEL_NAME}"
        echo ""
        continue
    fi
    
    python eval_dischargeme_di.py \
      --predictions_path="${PREDICTIONS_PATH}" \
      --reference_path="${REFERENCE_PATH}" \
      --model_name="${MODEL_NAME}" \
      --worker=worker-10 \
      --judge_worker=worker-13 \
      --judge_port=8000 \
      --judge_model="medicllama" \
      --eval_cef=True \
      --eval_traditional=False \
      --num_questions_to_generate=10 \
      --bootstrap_iterations=100 \
      --results_base_path="${RESULTS_PATH}"
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: ${MODEL_NAME}"
    else
        echo "ERROR: Failed for ${MODEL_NAME}"
    fi
    
    echo ""
done

echo "========================================="
echo "All models completed"
echo "Time: $(date)"
echo "========================================="

