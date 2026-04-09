#!/bin/bash
#SBATCH --job-name=gen_boxscore_summaries
#SBATCH --output=/home/yathagata/batch_jobs/outs/%j.out
#SBATCH --error=/home/yathagata/batch_jobs/outs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=normal

# Generate NBA Game Summaries from Boxscore Data
# This script runs the summary generation pipeline for the RotoWire dataset

set -e  # Exit on error

# --- Configuration ---
SRC_DIR="/home/yathagata/cef-translation/src/rotowire"

# --- Environment Setup ---
echo "Setting up environment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# --- Run Generation ---
echo "Starting summary generation..."
echo "Note: Configure model in generate_summaries.py before running"
echo ""
echo "Available generation models:"
echo "  - worker-3:8892 → Qwen/Qwen3-1.7B (qwen3-1.7b)"
echo "  - worker-0:8892 → Qwen/Qwen3-4B (qwen3-4b)"
echo "  - worker-9:8892 → meta-llama/Llama-3.2-3B-Instruct (llama3.2-3b)"
echo "  - worker-13:8892 → meta-llama/Llama-3.1-8B-Instruct (llama3.1-8b)"
echo "  (one model TBD)"
echo ""

cd ${SRC_DIR}
python generate_summaries.py

echo ""
echo "=== Generation Complete ==="

