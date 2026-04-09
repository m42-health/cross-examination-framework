#!/usr/bin/env python3
"""
Evaluate FRANK benchmark using Cross Examination Framework (CEF).

FRANK is a factuality evaluation benchmark for summarization.
The dataset already contains model-generated summaries (trg) with human annotations.

This script evaluates using CEF and computes correlations with human judgments.
"""

import os
import sys
import json
import uuid
import logging
import traceback
import fire
from datasets import load_from_disk

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frank.utils import (
    FrankQualityEvaluator,
    FrankTraditionalEvaluator,
    generate_cef_report,
    generate_correlation_report,
    generate_traditional_report,
    generate_traditional_correlation_report
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def eval_frank(
    # Judge model settings
    judge_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    judge_worker: str = "worker-9",
    judge_port: int = 8892,
    
    # Evaluation options
    eval_cef: bool = False,
    eval_traditional: bool = False,
    eval_correlation: bool = False,
    
    # CEF parameters
    num_questions_src: int = 10,
    num_questions_trg: int = 5,
    
    # Prompt settings (reuse summarization prompts)
    prompt_path: str = "/home/yathagata/cef-translation/src/cef_framework/prompts/summarization.yaml",
    
    # Data settings
    data_path: str = "/data/cef/frank/frank_original",
    results_base_path: str = "/data/cef/frank/results_cef",
    
    # Bootstrap settings
    bootstrap_iterations: int = 100,
    bootstrap_fraction: float = 1.0,
    
    # Processing settings
    num_proc: int = 16,
    debug: bool = False,
    splits: list = ["test"],
    
    # Filter by model (optional)
    filter_model: str = None,
):
    """
    Evaluate FRANK benchmark using CEF and traditional metrics.
    
    Args:
        judge_model: LLM model for CEF evaluation
        judge_worker: Worker hostname for judge model
        judge_port: Port for judge model API
        eval_cef: Run CEF evaluation
        eval_traditional: Run traditional metrics (ROUGE, BERTScore) using 'reference' column
        eval_correlation: Compute correlations with human judgments
        num_questions_src: Number of questions from article (coverage)
        num_questions_trg: Number of questions from summary (conformity/consistency)
        prompt_path: Path to summarization prompts YAML
        data_path: Path to FRANK dataset
        results_base_path: Base path for saving results
        bootstrap_iterations: Number of bootstrap iterations
        bootstrap_fraction: Fraction for bootstrap sampling
        num_proc: Number of processes for parallel evaluation
        debug: Debug mode - limit samples
        splits: List of splits to evaluate (default: all)
        filter_model: Filter to specific model (e.g., 'bart', 'pgn', 'bert_sum')
    
    Example:
        python eval_frank.py --eval_cef --debug
        python eval_frank.py --eval_traditional --debug
        python eval_frank.py --eval_cef --filter_model bart
        python eval_frank.py --eval_correlation  # After CEF evaluation
    """
    
    # Print arguments
    print("=" * 60)
    print("FRANK Benchmark Evaluation")
    print("=" * 60)
    print(f"  judge_model: {judge_model}")
    print(f"  judge_worker: {judge_worker}")
    print(f"  judge_port: {judge_port}")
    print(f"  eval_cef: {eval_cef}")
    print(f"  eval_traditional: {eval_traditional}")
    print(f"  eval_correlation: {eval_correlation}")
    print(f"  num_questions_src: {num_questions_src}")
    print(f"  num_questions_trg: {num_questions_trg}")
    print(f"  prompt_path: {prompt_path}")
    print(f"  data_path: {data_path}")
    print(f"  results_base_path: {results_base_path}")
    print(f"  debug: {debug}")
    print(f"  splits: {splits}")
    print(f"  filter_model: {filter_model}")
    print("=" * 60)
    
    # Validate inputs
    if not eval_cef and not eval_traditional and not eval_correlation:
        print("Warning: No evaluation method selected. Use --eval_cef, --eval_traditional, or --eval_correlation")
        return
    
    # Load dataset
    logger.info(f"Loading FRANK dataset from: {data_path}")
    try:
        ds = load_from_disk(data_path)
        logger.info(f"Loaded dataset with splits: {list(ds.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Filter by model if specified
    if filter_model:
        logger.info(f"Filtering to model: {filter_model}")
        for split in ds:
            original_size = len(ds[split])
            ds[split] = ds[split].filter(
                lambda x: x["model"] == filter_model,
                num_proc=num_proc
            )
            logger.info(f"  {split}: {original_size} -> {len(ds[split])} samples")
    
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
        
        evaluator = FrankQualityEvaluator(
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
    
    # Traditional Metrics Evaluation (ROUGE, BERTScore)
    # Note: We use a for loop instead of datasets.map() because the evaluate library
    # (ROUGE, BERTScore) contains ThreadLocalFileContext objects that cannot be pickled
    if eval_traditional:
        logger.info("Starting traditional metrics evaluation...")
        logger.info("Using 'reference' column as gold-standard summary")
        from tqdm import tqdm
        from datasets import Dataset
        
        for split in selected_splits:
            logger.info(f"Evaluating split: {split} ({len(ds[split])} samples)")
            try:
                evaluator = FrankTraditionalEvaluator()
                
                # Use a for loop instead of map() to avoid pickling issues
                results = []
                for example in tqdm(ds[split], desc=f"Traditional metrics - {split}"):
                    result = evaluator.evaluate_sample(example)
                    # Merge the result with the original example
                    merged = {**example, **result}
                    results.append(merged)
                
                # Convert back to dataset
                ds[split] = Dataset.from_list(results)
                
            except Exception as e:
                logger.error(f"Error evaluating {split}: {e}")
                traceback.print_exc()
                continue
    
    # Save results
    random_str = str(uuid.uuid4())
    if filter_model:
        results_dir = f"{results_base_path}/{filter_model}/debug" if debug else f"{results_base_path}/{filter_model}/{random_str}"
    else:
        results_dir = f"{results_base_path}/all_models/debug" if debug else f"{results_base_path}/all_models/{random_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save dataset
    dataset_path = os.path.join(results_dir, "results_ds")
    ds.save_to_disk(dataset_path)
    logger.info(f"Saved dataset to: {dataset_path}")
    
    # Generate and save CEF report
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
    
    # Generate and save traditional metrics report
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
        
        # Also compute correlations for traditional metrics
        logger.info("Computing correlations for traditional metrics with human judgments...")
        try:
            trad_corr_stats = generate_traditional_correlation_report(
                ds,
                bootstrap_iterations,
                bootstrap_fraction
            )
            print("\nTraditional Metrics Correlation with Human Judgments:")
            print(json.dumps(trad_corr_stats, indent=2))
            
            trad_corr_report_path = os.path.join(results_dir, "traditional_correlation_report.json")
            with open(trad_corr_report_path, "w") as f:
                json.dump(trad_corr_stats, f, indent=2)
            logger.info(f"Traditional correlation report saved to: {trad_corr_report_path}")
        except Exception as e:
            logger.error(f"Error computing traditional correlations: {e}")
            traceback.print_exc()
    
    # Generate and save CEF correlation report
    if eval_correlation:
        logger.info("Computing CEF correlations with human judgments...")
        try:
            corr_stats = generate_correlation_report(
                ds,
                bootstrap_iterations,
                bootstrap_fraction
            )
            print("\nCEF Correlation with Human Judgments:")
            print(json.dumps(corr_stats, indent=2))
            
            corr_report_path = os.path.join(results_dir, "cef_correlation_report.json")
            with open(corr_report_path, "w") as f:
                json.dump(corr_stats, f, indent=2)
            logger.info(f"CEF correlation report saved to: {corr_report_path}")
        except Exception as e:
            logger.error(f"Error computing CEF correlations: {e}")
            traceback.print_exc()
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    fire.Fire(eval_frank)

