#!/bin/bash -l

##############################
#     Resource Request       #
##############################
#SBATCH --job-name=cef_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=100:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --error=/home/yathagata/batch_jobs/outs/%J.out
#SBATCH --output=/home/yathagata/batch_jobs/outs/%J.out


##############################
#           Work             #
##############################

WORKER=worker-13
PORT=8893
# Define evaluation parameters
NUM_QUESTIONS_TO_GENERATE=10
# DATASET="flores"s
# DATASET="ntrex"
# MODEL="gemma3-270m"
# MODEL="gemma3-4b"
# MODEL="gpt-oss-20b"
# MODEL="qwen3-0.6b"
# MODEL="qwen3-1.7b"
# MODEL="qwen2.5-7b"
# MODEL="qwen2.5-72b"
# MODEL="llama3.2-1b"
# MODEL="nllb-3.3B"
# MODEL="roundtrip_qwen3-4b"
# MODEL="azure"
# MODEL="roundtrip_google"
# MODEL="roundtrip_azure"
DATASET="wmt25"
MODEL="errors"


# TOKEN_PER_QUESTION=20
# DATA_PATH="/data/cef/${DATASET}/${DATASET}_original"
DATA_PATH="/data/cef/${DATASET}/${DATASET}_${MODEL}"
# DATA_PATH="/data/cef/${DATASET}/${DATASET}_random"

RESULTS_BASE_PATH_CEF="/data/cef/${DATASET}/results_cef"
RESULTS_BASE_PATH_TQ="/data/cef/${DATASET}/results_tq"
RESULTS_BASE_PATH_TRAD="/data/cef/${DATASET}/results_trad_new"
RESULTS_BASE_PATH_CC="/data/cef/${DATASET}/results_cc"

# Change to the source directory
cd /home/yathagata/cef-translation/src/evaluation

# Run the evaluation
JUDGE_MODEL="deepseek-ai/DeepSeek-V3"
JUDGE_WORKER="worker-1"
JUDGE_PORT=8892

# JUDGE_MODEL="/models_llm/Llama-3.1-70B-Instruct"
# JUDGE_WORKER="worker-0"
# JUDGE_PORT=8892

# JUDGE_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
# JUDGE_WORKER="worker-13"
# JUDGE_PORT=8892

PROMPT_CATALOGUE_PATH="../cef_framework/prompts/translation.yaml"

# python eval_flores.py \
#     --worker $WORKER \
#     --port $PORT \
#     --judge_model $JUDGE_MODEL \
#     --judge_worker $JUDGE_WORKER \
#     --judge_port $JUDGE_PORT \
#     --eval_cef \
#     --num_questions_to_generate $NUM_QUESTIONS_TO_GENERATE \
#     --data_path $DATA_PATH \
#     --results_base_path $RESULTS_BASE_PATH_CEF \
#     --splits "[\"en-jp\"]"


# python eval_flores.py \
#     --eval_translation_quality \
#     --translation_quality_worker worker-12 \
#     --data_path $DATA_PATH \
#     --results_base_path $RESULTS_BASE_PATH_TQ

# python eval_flores.py \
#     --eval_traditional \
#     --reference_translation_path /data/cef/${DATASET}/${DATASET}_original \
#     --data_path $DATA_PATH \
#     --results_base_path $RESULTS_BASE_PATH_TRAD

# python eval_flores.py \
#     --eval_coverage_consistency \
#     --coverage_consistency_worker worker-12 \
#     --data_path $DATA_PATH \
#     --results_base_path $RESULTS_BASE_PATH_CC \
#     --debug

# Translation quality evaluation with single score template (translation-quality.jinja)
python eval_flores.py \
    --eval_translation_quality \
    --translation_quality_model_name $JUDGE_MODEL \
    --translation_quality_port $JUDGE_PORT \
    --translation_quality_worker $JUDGE_WORKER \
    --translation_quality_use_three_scores False \
    --data_path $DATA_PATH \
    --results_base_path $RESULTS_BASE_PATH_TQ \
    --splits "[\"en-jp\"]"

# Translation quality evaluation with three-score template (translation-quality2.jinja)
python eval_flores.py \
    --eval_translation_quality \
    --translation_quality_model_name $JUDGE_MODEL \
    --translation_quality_port $JUDGE_PORT \
    --translation_quality_worker $JUDGE_WORKER \
    --translation_quality_use_three_scores True \
    --data_path $DATA_PATH \
    --results_base_path $RESULTS_BASE_PATH_TQ \
    --splits "[\"en-jp\"]" 