# Reproducing Paper Experiments

This document provides step-by-step instructions to reproduce all experiments from the CEF paper.
All experiments require the **CEF API server** to be running (unless stated otherwise).

> **Paper**: [CEF: Cross-Examination Framework for Evaluating LLMs in Document Generation Tasks](https://arxiv.org/pdf/2601.19350)

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Starting the CEF Server](#2-starting-the-cef-server)
3. [Experiment 1 — Machine Translation (FLORES-200)](#3-experiment-1--machine-translation-flores-200)
4. [Experiment 2 — Machine Translation (WMT25)](#4-experiment-2--machine-translation-wmt25)
5. [Experiment 3 — Abstractive Summarization (CNN/DM)](#5-experiment-3--abstractive-summarization-cnndm)
6. [Experiment 4 — Factuality Assessment (FRANK)](#6-experiment-4--factuality-assessment-frank)
7. [Experiment 5 — Clinical Note Generation (ACI-Bench)](#7-experiment-5--clinical-note-generation-aci-bench)
8. [Experiment 6 — Clinical Summarization (DischargeME)](#8-experiment-6--clinical-summarization-dischargeme)
9. [Experiment 7 — Data-to-Text (Rotowire)](#9-experiment-7--data-to-text-rotowire)
10. [Experiment 8 — Self-Preference Bias Study](#10-experiment-8--self-preference-bias-study)
11. [WMT25 Human/GEMBA Alignment Analysis](#11-wmt25-humangemba-alignment-analysis)

---

## 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate cross-examination-framework
```

Additional dependencies for evaluation scripts (install into the same environment):

```bash
pip install datasets transformers fire sacrebleu bert-score sentencepiece jinja2
```

---

## 2. Starting the CEF Server

The CEF server must be running before any evaluation. It serves the QA generation and cross-examination API.

**Configure** `src/cef_framework/pipeline_params.yaml`:
```yaml
base_endpoint: "http://<LLM-HOST>:<PORT>/v1/"
api_key: "EMPTY"   # or your actual key
```

**Start the server:**
```bash
cd src/cef_framework
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8893
```

Or use `scripts/run_cef_server.sh` (edit `WORKER` and `PORT` variables inside first).

**Verify:**
```bash
curl http://localhost:8893/evaluate -X POST \
  -H "Content-Type: application/json" \
  -d '{"original_document": "Hello world", "generated_document": "Bonjour le monde"}'
```

---

## 3. Experiment 1 — Machine Translation (FLORES-200)

**Task**: Evaluate machine translation quality on FLORES-200 using CEF, traditional metrics (BLEU, ROUGE, BERTScore), and LLM-based translation quality scoring.

**Data**: FLORES-200 dataset, saved as HuggingFace `DatasetDict` to disk.  
Expected structure: `/data/cef/flores/flores_<model_name>` where each entry has `src` and `trg` fields.

### Run CEF Evaluation

```bash
cd src/evaluation

python eval_flores.py \
    --worker <CEF-SERVER-HOST> \
    --port 8893 \
    --judge_model "deepseek-ai/DeepSeek-V3" \
    --judge_worker <JUDGE-HOST> \
    --judge_port 8892 \
    --eval_cef \
    --num_questions_to_generate 10 \
    --data_path /data/cef/flores/flores_<model_name> \
    --results_base_path /data/cef/flores/results_cef \
    --splits '["en-fr", "en-de", "en-es"]'
```

### Run Traditional Metrics (BLEU, ROUGE, BERTScore)

```bash
python eval_flores.py \
    --eval_traditional \
    --reference_translation_path /data/cef/flores/flores_original \
    --data_path /data/cef/flores/flores_<model_name> \
    --results_base_path /data/cef/flores/results_trad
```

### Run LLM-Based Translation Quality

```bash
python eval_flores.py \
    --eval_translation_quality \
    --translation_quality_model_name "deepseek-ai/DeepSeek-V3" \
    --translation_quality_worker <JUDGE-HOST> \
    --translation_quality_port 8892 \
    --data_path /data/cef/flores/flores_<model_name> \
    --results_base_path /data/cef/flores/results_tq
```

### Debug Mode (10 samples only)

```bash
python eval_flores.py --worker <HOST> --eval_cef --data_path /data/cef/flores/flores_original --debug
```

### SLURM

Edit `scripts/eval.sh` to set `DATASET=flores`, `MODEL`, and judge settings, then:
```bash
sbatch scripts/eval.sh
```

### Models Evaluated in Paper

The paper evaluates: `gemma3-270m`, `gemma3-4b`, `gpt-oss-20b`, `qwen3-0.6b`, `qwen3-1.7b`, `qwen2.5-7b`, `qwen2.5-72b`, `llama3.2-1b`, `nllb-3.3B`, `azure` (Azure Translator), `roundtrip_google`, `roundtrip_azure`, `roundtrip_qwen3-4b`.

---

## 4. Experiment 2 — Machine Translation (WMT25)

**Task**: Evaluate models on the WMT25 English→Japanese shared task using CEF and LLM-based translation quality scoring.

**Data**: WMT25 evaluation set. Raw human evaluations are in `wmt25/` (not tracked in git due to size; obtain from the WMT25 organizers).

### Run Evaluation

```bash
cd src/evaluation

# CEF evaluation on en-jp
python eval_flores.py \
    --worker <CEF-SERVER-HOST> \
    --port 8893 \
    --judge_model "deepseek-ai/DeepSeek-V3" \
    --judge_worker <JUDGE-HOST> \
    --judge_port 8892 \
    --eval_cef \
    --num_questions_to_generate 10 \
    --data_path /data/cef/wmt25/wmt25_<model_name> \
    --results_base_path /data/cef/wmt25/results_cef \
    --splits '["en-jp"]'
```

### SLURM

```bash
# In scripts/eval.sh, set:
# DATASET=wmt25
# MODEL=<model_name>
sbatch scripts/eval.sh
```

---

## 5. Experiment 3 — Abstractive Summarization (CNN/DM)

**Task**: Evaluate LLM-generated summaries on CNN/DailyMail using CEF, traditional metrics, and robustness across multiple judge models.

**Data**: CNN/DM articles with LLM-generated summaries, stored as HuggingFace DatasetDicts at `/data/cef/cnndm/cnndm_<model_name>`.

### Run CEF Evaluation

```bash
cd src/summarization

python eval_cnndm.py \
    --judge_model "deepseek-ai/DeepSeek-V3" \
    --judge_worker <HOST> \
    --judge_port 8892 \
    --eval_cef \
    --num_questions_src 5 \
    --num_questions_trg 5 \
    --prompt_path ../cef_framework/prompts/summarization.yaml \
    --data_path /data/cef/cnndm/cnndm_<model_name> \
    --reference_path /data/cef/cnndm/cnndm_original \
    --results_base_path /data/cef/cnndm/results_cef
```

### Run Traditional Metrics

```bash
python eval_cnndm.py \
    --eval_traditional \
    --data_path /data/cef/cnndm/cnndm_<model_name> \
    --reference_path /data/cef/cnndm/cnndm_original \
    --results_base_path /data/cef/cnndm/results_trad
```

### Debug Mode

```bash
python eval_cnndm.py --judge_model "..." --judge_worker <HOST> --eval_cef \
    --data_path /data/cef/cnndm/cnndm_llama3.1-8b \
    --reference_path /data/cef/cnndm/cnndm_original --debug
```

### SLURM

```bash
# Edit MODEL in scripts/eval_cnndm.sh, then:
sbatch scripts/eval_cnndm.sh
```

### Models Evaluated in Paper

`llama3.1-8b`, `llama3.2-3b`, `qwen3-4b`, `gpt-oss-20b`, `qwen3-1.7b`

### Robustness / Multi-Judge Analysis

After collecting CEF answers from multiple judges, compute disagreement and deviation scores:

```bash
# This is a Jupyter-style script — run in a Python session or convert to notebook
python src/summarization/robustness_scores.py
```

Results are printed as DataFrames showing per-judge disagreement rates and deviation from majority vote.

---

## 6. Experiment 4 — Factuality Assessment (FRANK)

**Task**: Evaluate CEF correlation with human factuality judgments on the [FRANK benchmark](https://github.com/artidoro/frank).

**Data**: FRANK dataset (`data/frank/`) — **not tracked in git** (see [FRANK repo](https://github.com/artidoro/frank) to download). Place data at `/data/cef/frank/frank_original`.

### Data Preparation

```bash
# After downloading FRANK data:
cd src/frank
python utils.py  # Prepares the FRANK dataset in HuggingFace format
```

### Run CEF Evaluation

```bash
cd src/frank

python eval_frank.py \
    --judge_model "deepseek-ai/DeepSeek-V3" \
    --judge_worker <HOST> \
    --judge_port 8892 \
    --eval_cef \
    --eval_correlation \
    --num_questions_src 5 \
    --num_questions_trg 5 \
    --prompt_path ../cef_framework/prompts/summarization.yaml \
    --data_path /data/cef/frank/frank_original \
    --results_base_path /data/cef/frank/results_cef
```

### Run Traditional Metrics

```bash
python eval_frank.py \
    --eval_traditional \
    --data_path /data/cef/frank/frank_original \
    --results_base_path /data/cef/frank/results_trad
```

### SLURM

```bash
sbatch scripts/eval_frank.sh
```

---

## 7. Experiment 5 — Clinical Note Generation (ACI-Bench)

**Task**: Evaluate generated SOAP / clinical notes from the [ACI-Bench](https://github.com/wyim/aci-bench) dataset.

**Data**: LLM-generated notes, stored at `/data/cef/aci/`.

### Generate Notes First

```bash
cd src/note-generation
bash scripts/generate_aci_notes.sh   # or edit and run generate_notes.py directly
```

### Run CEF Evaluation

```bash
cd src/note-generation

python eval_aci.py \
    --judge_model "deepseek-ai/DeepSeek-V3" \
    --judge_worker <HOST> \
    --judge_port 8892 \
    --eval_cef \
    --num_questions_src 5 \
    --num_questions_trg 5 \
    --prompt_path ../cef_framework/prompts/note_generation.yaml \
    --data_path /data/cef/aci/aci_<model_name> \
    --results_base_path /data/cef/aci/results_cef
```

### SLURM

```bash
sbatch scripts/eval_aci.sh
```

---

## 8. Experiment 6 — Clinical Summarization (DischargeME)

**Task**: Evaluate Brief Hospital Course (BHC) generation on the [DischargeME dataset](https://physionet.org/content/discharge-me/1.3/) from PhysioNet.

> ⚠️ **Access Required**: DischargeME data requires a PhysioNet credentialed account. Apply at https://physionet.org/content/discharge-me/1.3/

**Data**: Predictions JSONL at `/data/cef/dischargeme/<model>/predictions/...`. References at `/data/cef/dischargeme/gt/discharge-me/1.3`.

### Run CEF + Traditional Evaluation

```bash
cd src/evaluation

python eval_dischargeme.py \
    --predictions_path /data/cef/dischargeme/<model>/predictions/<model>_predictions.jsonl \
    --reference_path /data/cef/dischargeme/gt/discharge-me/1.3 \
    --model_name "<model>" \
    --worker <CEF-SERVER-HOST> \
    --judge_worker <JUDGE-HOST> \
    --judge_port 8892 \
    --eval_cef \
    --eval_traditional \
    --num_questions_to_generate 10
```

### Evaluate Traditional Metrics Only

```bash
python eval_dischargeme.py \
    --predictions_path /path/to/predictions.jsonl \
    --eval_traditional
```

### Debug Mode

```bash
python eval_dischargeme.py \
    --predictions_path /path/to/predictions.jsonl \
    --eval_traditional --debug
```

### SLURM

```bash
sbatch scripts/eval_dischargeme_cef.sh      # CEF evaluation
sbatch scripts/eval_dischargeme_traditional.sh  # Traditional metrics only
```

---

## 9. Experiment 7 — Data-to-Text (Rotowire)

**Task**: Evaluate LLM-generated sports reports from structured box-score data (Rotowire).

**Data**: Rotowire dataset (`data/boxscore-data/`) — **not tracked in git**. Download from the [Rotowire repo](https://github.com/harvardnlp/boxscore-data). Place JSON files at `data/boxscore-data/rotowire/`.

### Generate Summaries

```bash
cd src/rotowire

python generate_summaries.py \
    --judge_model "deepseek-ai/DeepSeek-V3" \
    --judge_worker <HOST> \
    --judge_port 8892 \
    --data_path /data/cef/boxscore/boxscore_original \
    --output_path /data/cef/boxscore/boxscore_<model_name>
```

### Run CEF Evaluation

```bash
python eval_boxscore.py \
    --judge_model "deepseek-ai/DeepSeek-V3" \
    --judge_worker <HOST> \
    --judge_port 8892 \
    --eval_cef \
    --num_questions_src 5 \
    --num_questions_trg 5 \
    --prompt_path ../cef_framework/prompts/boxscore.yaml \
    --data_path /data/cef/boxscore/boxscore_<model_name> \
    --results_base_path /data/cef/boxscore/results_cef
```

### SLURM

```bash
sbatch scripts/eval_boxscore.sh
```

---

## 10. Experiment 8 — Self-Preference Bias Study

**Task**: Measure whether LLMs show self-preference bias when acting as cross-examination judges (i.e., do they rate their own outputs higher?).

**Setup**: Run CEF on CNN/DM summaries using multiple judge models, then compute cross-judge statistics.

### Step 1 — Collect answers per judge

```bash
# Run eval_cnndm.py for each judge model to collect per-model answers
# Results will be saved to /data/cef/cnndm/cnndm_answers_<judge_model>
```

See `scripts/qwen3-launch.sh`, `scripts/qwen2.5-72b.sh`, etc. for per-model launch scripts.

### Step 2 — Run self-preference bias experiment

```bash
python scripts/self_preference_bias_experiment.py \
    --data_path /data/cef/cnndm \
    --models "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-235B-A22B-Instruct-2507" \
             "deepseek-ai/DeepSeek-V3" "openai/gpt-oss-120b" \
    --output_path figures/self_preference_scatter.png
```

### Step 3 — Compute robustness scores

```bash
# Run from a Jupyter environment or Python REPL:
# - src/summarization/robustness_scores.py  (CNN/DM)
# - src/rotowire/robustness_scores.py       (Rotowire)
# - src/note-generation/robustness_scores.py (ACI)
```

Each script prints **disagreement rates** and **deviation-from-majority** tables across all judge models and answer types (`src_src`, `src_trg`, `trg_trg`, `trg_src`).

---

## 11. WMT25 Human/GEMBA Alignment Analysis

**Task**: Analyze alignment between CEF mismatching questions and (a) GEMBA error annotations and (b) human expert annotations on WMT25 English→Japanese translations.

**Data**: WMT25 human evaluation JSONL — not tracked in git (123MB). Obtain from WMT25 organizers.

### Run GEMBA–CEF Alignment

```bash
cd wmt25
python gemba_compare.py \
    --cef_results_path /data/cef/wmt25/results_cef/<run_id>/results_ds \
    --humeval_path wmt25-genmt-humeval.jsonl \
    --output_path wmt25/gemba_alignment_results.json
```

### Run Human–CEF Alignment

```bash
python human_compare.py \
    --cef_results_path /data/cef/wmt25/results_cef/<run_id>/results_ds \
    --annotations_path wmt25-enja-extracted-samples-with-errors.jsonl \
    --output_path wmt25/human_alignment_results.json
```

### Run Semantic Fidelity Classification

```bash
python check_semantic_fidelity.py \
    --annotations_path wmt25-enja-extracted-samples-with-errors.jsonl \
    --output_path wmt25/semantic_fidelity_results.json
```

### Aggregate All Results

```bash
python read_all_results.py \
    --results_dir /data/cef/wmt25/results_cef \
    --output figures/task_heatmaps.png
```

---

## 📝 Notes

- All SLURM scripts in `scripts/` use `#SBATCH` headers configured for our internal cluster. Adjust `--error`, `--output`, `--mem`, and `--time` to match your cluster.
- Worker hostnames (e.g., `worker-7`, `worker-12`) refer to internal cluster nodes running vLLM servers. Replace with your own hostnames or use `localhost` for local inference.
- All result files are saved as HuggingFace `DatasetDict` (`.save_to_disk()`) plus a `report.json` summary.
- Large data files from `/data/cef/` are **not** included in this repository. You will need to provision your own storage and download the datasets as described above.
