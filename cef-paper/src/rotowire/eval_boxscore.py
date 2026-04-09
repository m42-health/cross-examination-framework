#!/usr/bin/env python3
"""
CEF Evaluation for RotoWire Boxscore-to-Summary Generation.

This script evaluates the quality of generated NBA game summaries using:
1. CEF Framework: Coverage, Conformity, Faithfulness, Consistency, Conciseness
2. Traditional Metrics: ROUGE, BERTScore

Usage:
    python eval_boxscore.py --model_name llama3.1-8b --judge_model "meta-llama/Llama-3.3-70B-Instruct" --judge_worker worker-7

Or via shell script:
    sbatch scripts/eval_boxscore.sh
"""

import os
import gc
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from typing import Dict, Any, List

from utils import (
    BoxscoreQualityEvaluator,
    BoxscoreTraditionalEvaluator,
    generate_cef_report,
    generate_traditional_report,
    compute_bootstrap_ci
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Configuration for boxscore-to-summary evaluation."""
    
    # Default paths
    BASE_DATA_PATH = Path("/data/cef/boxscore")
    PROMPT_CATALOGUE = Path("/home/yathagata/cef-translation/src/cef_framework/prompts/boxscore.yaml")
    
    # Judge model defaults
    # Available judges:
    # - worker-8:8892 → meta-llama/Llama-3.3-70B-Instruct
    # - worker-4:8892 → Qwen/Qwen3-235B-A22B-Instruct-2507
    # - worker-5:8892 → deepseek-ai/DeepSeek-V3
    # - worker-6:8892 → openai/gpt-oss-120b
    JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
    JUDGE_WORKER = "worker-8"
    JUDGE_PORT = 8892
    
    # CEF parameters
    NUM_QUESTIONS_SRC = 10  # Questions from boxscore
    NUM_QUESTIONS_TRG = 10  # Questions from summary
    
    # Processing
    NUM_PROCESSES = 10
    DEBUG = False


def run_cef_evaluation(
    dataset,
    evaluator: BoxscoreQualityEvaluator,
    split: str = "test"
) -> Any:
    """
    Run CEF evaluation on a dataset split.
    
    Args:
        dataset: Dataset with src (boxscore) and trg (summary)
        evaluator: Configured BoxscoreQualityEvaluator
        split: Dataset split to evaluate
        
    Returns:
        Dataset with cef_details added
    """
    logger.info(f"Running CEF evaluation on {split} split ({len(dataset[split])} examples)")
    
    def evaluate_example(example):
        cef_result = evaluator.evaluate_cef_sample(
            boxscore=example["src"],
            summary=example["trg"]
        )
        example["cef_details"] = cef_result
        return example
    
    dataset[split] = dataset[split].map(
        evaluate_example,
        num_proc=Config.NUM_PROCESSES,
        desc=f"CEF evaluation ({split})"
    )
    
    return dataset


def run_traditional_evaluation(dataset, split: str = "test") -> Dict:
    """
    Run traditional metric evaluation (ROUGE, BERTScore).
    
    Note: This requires a reference dataset with gold summaries.
    """
    # Load original dataset for reference summaries
    original_path = Config.BASE_DATA_PATH / "boxscore_original"
    if original_path.exists():
        original = load_from_disk(str(original_path))
        references = [ex["trg"] for ex in original[split]]
    else:
        logger.warning("Original dataset not found. Skipping traditional evaluation.")
        return {}
    
    predictions = [ex["trg"] for ex in dataset[split]]
    
    evaluator = BoxscoreTraditionalEvaluator()
    return evaluator.evaluate(predictions, references)


def main():
    parser = argparse.ArgumentParser(description="CEF Evaluation for Boxscore-to-Summary")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name of the model that generated summaries (e.g., llama3.1-8b)")
    parser.add_argument("--judge_model", type=str, default=Config.JUDGE_MODEL,
                       help="Judge model for CEF evaluation")
    parser.add_argument("--judge_worker", type=str, default=Config.JUDGE_WORKER,
                       help="Worker hosting the judge model")
    parser.add_argument("--judge_port", type=int, default=Config.JUDGE_PORT,
                       help="Port for the judge model")
    parser.add_argument("--num_questions_src", type=int, default=Config.NUM_QUESTIONS_SRC,
                       help="Number of questions to generate from boxscore")
    parser.add_argument("--num_questions_trg", type=int, default=Config.NUM_QUESTIONS_TRG,
                       help="Number of questions to generate from summary")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to dataset (default: /data/cef/boxscore/boxscore_{model_name})")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save results (default: same as data_path with _cef suffix)")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode with fewer examples")
    parser.add_argument("--eval_cef", action="store_true",
                       help="Run CEF evaluation")
    parser.add_argument("--eval_traditional", action="store_true",
                       help="Run traditional ROUGE/BERTScore evaluation")
    parser.add_argument("--with_reference", action="store_true",
                       help="Replace src with reference trg (compare generated vs reference summary)")
    parser.add_argument("--reference_path", type=str, default=None,
                       help="Path to reference dataset (default: /data/cef/boxscore/boxscore_original)")
    
    args = parser.parse_args()
    
    # Set up paths
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Config.BASE_DATA_PATH / f"boxscore_{args.model_name}"
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Config.BASE_DATA_PATH / f"boxscore_{args.model_name}_cef"
    
    logger.info(f"Loading dataset from: {data_path}")
    logger.info(f"Will save results to: {output_path}")
    
    # Load dataset
    try:
        dataset = load_from_disk(str(data_path))
        logger.info(f"Loaded dataset with splits: {list(dataset.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # With reference mode - replace src with reference trg
    # This compares generated summary against reference summary instead of boxscore
    if args.with_reference:
        logger.info("With reference mode: Replacing src with reference trg")
        try:
            # Determine reference path
            if args.reference_path:
                reference_data_path = Path(args.reference_path)
            else:
                # Extract model name from data_path and get reference path
                model_name = args.model_name
                reference_data_path = Config.BASE_DATA_PATH / "boxscore_original"
            
            logger.info(f"Loading reference dataset from: {reference_data_path}")
            ds_ref = load_from_disk(str(reference_data_path))
            
            for split in dataset:
                dataset[split] = dataset[split].map(
                    lambda x, idx: {"src": ds_ref[split][idx]["trg"]},
                    num_proc=16,
                    load_from_cache_file=False,
                    with_indices=True
                )
            logger.info("Successfully replaced src with reference trg (reference game summaries)")
        except Exception as e:
            logger.error(f"Failed to load reference dataset: {e}")
            logger.warning("Continuing with original src (boxscore data)")
    
    if args.debug:
        logger.info("DEBUG mode: limiting to 5 examples per split")
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(min(5, len(dataset[split]))))
    
    # Validate that at least one evaluation is selected
    if not args.eval_cef and not args.eval_traditional:
        logger.warning("No evaluation method selected. Use --eval_cef or --eval_traditional")
        return
    
    cef_report = {}
    traditional_report = {}
    
    # Run CEF evaluation if requested
    if args.eval_cef:
        # Initialize evaluator
        evaluator = BoxscoreQualityEvaluator(
            judge_model=args.judge_model,
            judge_worker=args.judge_worker,
            judge_port=args.judge_port,
            prompt_catalogue_path=str(Config.PROMPT_CATALOGUE),
            num_questions_src=args.num_questions_src,
            num_questions_trg=args.num_questions_trg
        )
        
        logger.info(f"Using judge model: {args.judge_model}")
        
        # Run CEF evaluation
        dataset = run_cef_evaluation(dataset, evaluator)
        
        # Generate CEF report
        cef_report = generate_cef_report(dataset)
        
        logger.info("\n" + "="*80)
        logger.info("CEF EVALUATION RESULTS")
        logger.info("="*80)
        for metric, values in cef_report.items():
            logger.info(f"{metric.capitalize()}: {values['mean']:.2f} (95% CI: [{values['ci_low']:.2f}, {values['ci_high']:.2f}])")
    
    # Run traditional evaluation if requested
    if args.eval_traditional:
        logger.info("\nRunning traditional evaluation...")
        traditional_report = run_traditional_evaluation(dataset)
        if traditional_report:
            logger.info("\n" + "="*80)
            logger.info("TRADITIONAL EVALUATION RESULTS")
            logger.info("="*80)
            for metric, value in traditional_report.items():
                logger.info(f"{metric}: {value:.2f}")
    
    # Save dataset with evaluation details
    os.makedirs(str(output_path), exist_ok=True)
    dataset.save_to_disk(str(output_path))
    logger.info(f"\nSaved evaluated dataset to: {output_path}")
    
    # Save report
    report = {
        "model": args.model_name,
        "judge_model": args.judge_model if args.eval_cef else None,
        "cef_scores": cef_report if cef_report else None,
        "traditional_scores": traditional_report if traditional_report else None
    }
    
    report_path = output_path / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation report to: {report_path}")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()

