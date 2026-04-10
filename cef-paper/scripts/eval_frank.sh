#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=frank_eval
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
JUDGE_WORKER="worker-12"
JUDGE_PORT=8892

# Model to evaluate (change this for different models)
# Available models in FRANK: bart, bert_sum, bus, pgn, s2s
# Leave empty to evaluate all models
FILTER_MODEL=""
# FILTER_MODEL="bart"
# FILTER_MODEL="bert_sum"
# FILTER_MODEL="bus"
# FILTER_MODEL="pgn"
# FILTER_MODEL="s2s"

# Data paths
DATA_PATH="/data/cef/frank/frank_original"

# Results paths
RESULTS_BASE_PATH_CEF="/data/cef/frank/results_cef"
RESULTS_BASE_PATH_TRAD="/data/cef/frank/results_trad"
RESULTS_BASE_PATH_CORR="/data/cef/frank/results_correlation"

# CEF parameters
NUM_QUESTIONS_SRC=5
NUM_QUESTIONS_TRG=5

# Prompt path (reuse summarization prompts)
PROMPT_PATH="/home/yathagata/cef-translation/src/cef_framework/prompts/summarization.yaml"

# Change to the source directory
cd /home/yathagata/cef-translation/src/frank

# =============================================
# CEF Evaluation
# =============================================
# python eval_frank.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_CEF"

# =============================================
# CEF Evaluation with Model Filter
# =============================================
# python eval_frank.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_CEF" \
#     --filter_model "$FILTER_MODEL"

# =============================================
# Debug mode (5 samples only)
# =============================================
# python eval_frank.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_CEF" \
#     --debug

# =============================================
# CEF + Correlation with Human Judgments
# Main evaluation for FRANK benchmark
# =============================================
# python eval_frank.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --eval_correlation \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_CEF"

# =============================================
# Traditional Metrics Evaluation (ROUGE, BERTScore)
# Uses 'reference' column as gold-standard summary
# =============================================
python eval_frank.py \
    --eval_traditional \
    --data_path "$DATA_PATH" \
    --results_base_path "$RESULTS_BASE_PATH_TRAD"

# =============================================
# Traditional Metrics with Model Filter
# =============================================
# python eval_frank.py \
#     --eval_traditional \
#     --data_path "$DATA_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_TRAD" \
#     --filter_model "$FILTER_MODEL"

# =============================================
# Both CEF and Traditional Metrics
# =============================================
# python eval_frank.py \
#     --judge_model "$JUDGE_MODEL" \
#     --judge_worker "$JUDGE_WORKER" \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --eval_traditional \
#     --num_questions_src $NUM_QUESTIONS_SRC \
#     --num_questions_trg $NUM_QUESTIONS_TRG \
#     --prompt_path "$PROMPT_PATH" \
#     --data_path "$DATA_PATH" \
#     --results_base_path "$RESULTS_BASE_PATH_CEF"

# =============================================
# Correlation Only (after CEF evaluation is done)
# Use this to compute correlations on existing CEF results
# =============================================
# python eval_frank.py \
#     --eval_correlation \
#     --data_path "$RESULTS_BASE_PATH_CEF/<run_id>/results_ds" \
#     --results_base_path "$RESULTS_BASE_PATH_CORR"

