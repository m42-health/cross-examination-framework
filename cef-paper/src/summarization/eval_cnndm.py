#!/usr/bin/env python3
"""
Evaluate CNN/DailyMail summarization using Cross Examination Framework (CEF).

This script evaluates the quality of generated summaries by:
1. CEF Evaluation: Generate questions from article/summary and cross-examine
2. Traditional Metrics: ROUGE, BERTScore
"""

import os
import sys
import json
import uuid
import logging
import traceback
import fire
from datasets import load_from_disk
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    SummarizationQualityEvaluator,
    SummarizationTraditionalEvaluator,
    generate_cef_report,
    generate_traditional_report
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def eval_cnndm(
    # Judge model settings
    judge_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    judge_worker: str = "worker-9",
    judge_port: int = 8892,
    
    # Evaluation options
    eval_cef: bool = False,
    eval_traditional: bool = False,
    with_reference: bool = False,
    
    # CEF parameters
    num_questions_src: int = 10,
    num_questions_trg: int = 5,
    
    # Prompt settings
    prompt_path: str = "/home/yathagata/cef-translation/src/cef_framework/prompts/summarization.yaml",
    
    # Data settings
    data_path: str = "/data/cef/cnndm/cnndm_llama3.1-8b",
    reference_path: str = "/data/cef/cnndm/cnndm_original",
    results_base_path: str = "/data/cef/cnndm/results_cef",
    
    # Bootstrap settings
    bootstrap_iterations: int = 100,
    bootstrap_fraction: float = 1.0,
    
    # Processing settings
    num_proc: int = 16,
    debug: bool = False,
    splits: list = None,
):
    """
    Evaluate CNN/DailyMail summarization quality.
    
    Args:
        judge_model: LLM model for CEF evaluation
        judge_worker: Worker hostname for judge model
        judge_port: Port for judge model API
        eval_cef: Run CEF evaluation
        eval_traditional: Run traditional metrics (ROUGE, BERTScore)
        num_questions_src: Number of questions from article (coverage)
        num_questions_trg: Number of questions from summary (conformity/consistency)
        prompt_path: Path to summarization prompts YAML
        data_path: Path to dataset with generated summaries
        reference_path: Path to original dataset with reference summaries
        results_base_path: Base path for saving results
        bootstrap_iterations: Number of bootstrap iterations
        bootstrap_fraction: Fraction for bootstrap sampling
        num_proc: Number of processes for parallel evaluation
        debug: Debug mode - limit samples
        splits: List of splits to evaluate (default: all)
        with_reference: If True, replace src with reference trg (compare generated vs reference summary)
    
    Example:
        python eval_cnndm.py --eval_cef --debug
        python eval_cnndm.py --eval_cef --with_reference  # Compare against reference summaries
    """
    
    # Print arguments
    print("=" * 60)
    print("CNN/DailyMail Summarization Evaluation")
    print("=" * 60)
    print(f"  judge_model: {judge_model}")
    print(f"  judge_worker: {judge_worker}")
    print(f"  judge_port: {judge_port}")
    print(f"  eval_cef: {eval_cef}")
    print(f"  eval_traditional: {eval_traditional}")
    print(f"  with_reference: {with_reference}")
    print(f"  num_questions_src: {num_questions_src}")
    print(f"  num_questions_trg: {num_questions_trg}")
    print(f"  prompt_path: {prompt_path}")
    print(f"  data_path: {data_path}")
    print(f"  reference_path: {reference_path}")
    print(f"  results_base_path: {results_base_path}")
    print(f"  debug: {debug}")
    print(f"  splits: {splits}")
    print("=" * 60)
    
    # Validate inputs
    if not eval_cef and not eval_traditional:
        print("Warning: No evaluation method selected. Use --eval_cef or --eval_traditional")
        return
    
    # Load dataset
    logger.info(f"Loading dataset from: {data_path}")
    try:
        ds = load_from_disk(data_path)
        logger.info(f"Loaded dataset with splits: {list(ds.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # With reference mode - replace src with reference trg
    # This compares generated summary against reference summary instead of article
    if with_reference:
        logger.info("With reference mode: Replacing src with reference trg")
        try:
            # Extract model name from data_path and get reference path
            model_name = data_path.split("_")[-1]
            reference_data_path = data_path.replace(f"_{model_name}", "_original")
            if reference_data_path == data_path:
                reference_data_path = reference_path
            
            logger.info(f"Loading reference dataset from: {reference_data_path}")
            ds_ref = load_from_disk(reference_data_path)
            
            for split in ds:
                ds[split] = ds[split].map(
                    lambda x, idx: {"src": ds_ref[split][idx]["trg"]},
                    num_proc=16,
                    load_from_cache_file=False,
                    with_indices=True
                )
            logger.info("Successfully replaced src with reference trg (reference summaries)")
        except Exception as e:
            logger.error(f"Failed to load reference dataset: {e}")
            logger.warning("Continuing with original src (articles)")
    
    # Debug mode - limit samples
    if debug:
        for split in ds:
            ds[split] = ds[split].select(range(min(5, len(ds[split]))))
        logger.info("DEBUG mode: Limited to 5 samples per split")
    
    # Determine splits to process
    if splits is not None:
        if isinstance(splits, str):
            selected_splits = json.loads(splits)
        else:
            selected_splits = splits
    else:
        selected_splits = list(ds.keys())
    
    logger.info(f"Processing splits: {selected_splits}")
    
    # CEF Evaluation
    if eval_cef:
        logger.info("Starting CEF evaluation...")
        
        evaluator = SummarizationQualityEvaluator(
            model_name=judge_model,
            worker=judge_worker,
            port=judge_port,
            prompt_path=prompt_path
        )
        
        for split in selected_splits:
            logger.info(f"Evaluating split: {split} ({len(ds[split])} samples)")
            try:
                ds[split] = ds[split].map(
                    lambda x: evaluator.evaluate_cef(x, num_questions_src, num_questions_trg),
                    num_proc=num_proc,
                    load_from_cache_file=False,
                    desc=f"CEF evaluation - {split}"
                )
            except Exception as e:
                logger.error(f"Error evaluating {split}: {e}")
                traceback.print_exc()
                continue
    
    # Traditional Metrics Evaluation
    # Note: We use a for loop instead of datasets.map() because the evaluate library
    # (ROUGE, BERTScore) contains ThreadLocalFileContext objects that cannot be pickled
    if eval_traditional:
        logger.info("Starting traditional metrics evaluation...")
        from tqdm import tqdm
        
        for split in selected_splits:
            logger.info(f"Evaluating split: {split}")
            try:
                evaluator = SummarizationTraditionalEvaluator(
                    reference_path=reference_path,
                    reference_split=split
                )
                
                # Use a for loop instead of map() to avoid pickling issues
                results = []
                for example in tqdm(ds[split], desc=f"Traditional metrics - {split}"):
                    result = evaluator.evaluate_sample(example)
                    # Merge the result with the original example
                    merged = {**example, **result}
                    results.append(merged)
                
                # Convert back to dataset
                from datasets import Dataset
                ds[split] = Dataset.from_list(results)
                
            except Exception as e:
                logger.error(f"Error evaluating {split}: {e}")
                traceback.print_exc()
                continue
    
    # Save results
    random_str = str(uuid.uuid4())
    results_dir = f"{results_base_path}/debug" if debug else f"{results_base_path}/{random_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save dataset
    dataset_path = os.path.join(results_dir, "results_ds")
    ds.save_to_disk(dataset_path)
    logger.info(f"Saved dataset to: {dataset_path}")
    
    # Generate and save reports
    if eval_cef:
        stats = generate_cef_report(
            ds,
            data_path,
            num_questions_src,
            num_questions_trg,
            bootstrap_iterations,
            bootstrap_fraction
        )
        print("\nCEF Evaluation Results:")
        print(json.dumps(stats, indent=2))
        
        report_path = os.path.join(results_dir, "cef_report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"CEF report saved to: {report_path}")
    
    if eval_traditional:
        stats = generate_traditional_report(
            ds,
            data_path,
            bootstrap_iterations,
            bootstrap_fraction
        )
        print("\nTraditional Metrics Results:")
        print(json.dumps(stats, indent=2))
        
        report_path = os.path.join(results_dir, "traditional_report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Traditional report saved to: {report_path}")
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    fire.Fire(eval_cnndm)

