#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=aci_eval
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

# Judge model settings
# Available judges:
# - worker-7:8892 → meta-llama/Llama-3.3-70B-Instruct
# - worker-0:8892 → Qwen/Qwen3-235B-A22B-Instruct-2507
# - worker-5:8892 → deepseek-ai/DeepSeek-V3
# - worker-2:8892 → openai/gpt-oss-120b
JUDGE_MODEL="deepseek-ai/DeepSeek-V3"
JUDGE_WORKER="worker-5"
JUDGE_PORT=8892

# Model to evaluate (change this for different models)
# MODEL="qwen3-1.7b"
# MODEL="qwen3-4b"
# MODEL="gpt-oss-20b"
# MODEL="llama3.1-8b"
MODEL="llama3.2-3b"
# MODEL="original"

# Data paths
DATA_PATH="/data/cef/aci/aci_${MODEL}"
REFERENCE_PATH="/data/cef/aci/aci_original"

# Results paths
RESULTS_BASE_PATH_CEF="/data/cef/aci/results_cef"
RESULTS_BASE_PATH_REF="/data/cef/aci/results_ref"
RESULTS_BASE_PATH_TRAD="/data/cef/aci/results_trad"

# CEF parameters
NUM_QUESTIONS_SRC=10
NUM_QUESTIONS_TRG=10

# Prompt path
PROMPT_PATH="/home/yathagata/cef-translation/src/cef_framework/prompts/note_generation.yaml"

# Change to the source directory
cd /home/yathagata/cef-translation/src/note-generation

# =============================================
# CEF Evaluation
# =============================================
# python eval_aci.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --reference_path "$REFERENCE_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_CEF"

# =============================================
# Traditional Metrics Evaluation (ROUGE, BERTScore)
# =============================================
# python eval_aci.py \
#     --eval_traditional \
#     --data_path "$DATA_PATH" \
#     --reference_path "$REFERENCE_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_TRAD"

# =============================================
# Debug mode (5 samples only)
# =============================================
# python eval_aci.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --reference_path "$REFERENCE_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_CEF" \
#     --debug

# =============================================
# Both CEF and Traditional
# =============================================
python eval_aci.py \
    --judge_model "$JUDGE_MODEL" \
    --judge_worker "$JUDGE_WORKER" \
    --judge_port $JUDGE_PORT \
    --eval_traditional \
    --num_questions_src $NUM_QUESTIONS_SRC \
    --num_questions_trg $NUM_QUESTIONS_TRG \
    --prompt_path "$PROMPT_PATH" \
    --data_path "$DATA_PATH" \
    --reference_path "$REFERENCE_PATH" \
    --results_base_path "$RESULTS_BASE_PATH_CEF"

# =============================================
# With Reference Mode
# Compares generated note against reference note (not conversation)
# Useful for measuring similarity to gold-standard clinical notes
# =============================================
# python eval_aci.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --with_reference \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --reference_path "$REFERENCE_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_REF"

