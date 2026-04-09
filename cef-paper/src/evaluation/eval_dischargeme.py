# %%
import requests
from datasets import Dataset, DatasetDict, load_from_disk
import sys
import os
from transformers import AutoTokenizer
from utils import (
    evaluate_cef_sample, 
    generate_cef_report, 
    find_worker, 
    DischargeMEEvaluator,
    generate_dischargeme_report
)
import json
import uuid
import pandas as pd
# Add the parent directory to the path so we can import cef_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cef_framework.utils import load_yaml
import fire
import traceback


def evaluate_dischargeme_traditional_sample(example):
    """
    Standalone function to evaluate a single DischargeME sample with traditional metrics.
    This function creates its own evaluator instance to avoid pickling issues.
    
    Args:
        example (dict): Dictionary containing 'trg' (prediction) and 'reference'
        
    Returns:
        dict: Dictionary with evaluation metrics (bleu, rouge1, bertscore_f1)
    """
    evaluator = DischargeMEEvaluator()
    return evaluator.evaluate_sample({'prediction': example['trg'], 'reference': example['reference']})


def prepare_dischargeme_dataset(
    predictions_path: str,
    reference_path: str,
    model_name: str = None,
    debug: bool = False
):
    """
    Prepare HuggingFace dataset from DischargeME predictions and references.
    
    Args:
        predictions_path (str): Path to predictions JSONL file containing llm_gen_bhc
        reference_path (str): Path to reference CSV files directory
        model_name (str): Name of the model (used for identifying prediction file)
        debug (bool): If True, limit to 10 samples
        
    Returns:
        DatasetDict: Dataset containing src (full discharge summary), trg (predicted BHC), 
                     reference (ground truth BHC), and hadm_id
    """
    # Load predictions
    predictions = []
    with open(predictions_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))
    
    # Create mapping from hadm_id to prediction
    pred_dict = {}
    for pred in predictions:
        hadm_id = pred['hadm_id']
        # Extract the full discharge summary from input messages
        discharge_summary = ""
        if 'input' in pred and isinstance(pred['input'], list):
            for msg in pred['input']:
                if msg.get('role') == 'user':
                    discharge_summary = msg.get('content', '')
                    break
        
        pred_dict[hadm_id] = {
            'prediction': pred.get('llm_gen_bhc', ''),
            'discharge_summary': discharge_summary
        }
    
    # Load reference data
    reference1 = pd.read_csv(
        os.path.join(reference_path, "test_phase_1", "discharge_target.csv.gz"), 
        keep_default_na=False
    )
    reference2 = pd.read_csv(
        os.path.join(reference_path, "test_phase_2", "discharge_target.csv.gz"), 
        keep_default_na=False
    )
    reference = pd.concat([reference1, reference2], axis=0)
    
    # Create dataset entries
    dataset_entries = []
    for idx, row in reference.iterrows():
        hadm_id = row['hadm_id']
        if hadm_id in pred_dict:
            entry = {
                'hadm_id': hadm_id,
                'src': pred_dict[hadm_id]['discharge_summary'],  # Full discharge summary
                'trg': pred_dict[hadm_id]['prediction'],  # Predicted BHC
                'reference': row['brief_hospital_course']  # Reference BHC
            }
            dataset_entries.append(entry)
    
    print(f"Created dataset with {len(dataset_entries)} samples")
    print(f"Predictions found: {len(pred_dict)}")
    print(f"References found: {len(reference)}")
    
    if debug:
        dataset_entries = dataset_entries[:10]
        print(f"Debug mode: Limited to {len(dataset_entries)} samples")
    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_entries)
    
    # Create a DatasetDict with a single split
    ds = DatasetDict({
        "test": dataset
    })
    
    return ds


def eval_dischargeme(
    predictions_path: str,
    reference_path: str = "/data/cef/dischargeme/gt/discharge-me/1.3",
    model_name: str = "DeepSeek-V3.1",
    worker: str = None,
    slurm_job_name: str = None,
    port: int = 8893,
    eval_cef: bool = False,
    pipeline_params_path: str = "../cef_framework/pipeline_params.yaml",
    prompt_catalogue_path: str = "../cef_framework/prompts/hospital_course.yaml",
    judge_model: str = "medicllama",
    judge_worker: str = None,
    judge_port: int = 8000,
    num_questions_to_generate: int = None,
    token_per_question: int = None,
    bootstrap_iterations: int = 100,
    bootstrap_fraction: float = 1,
    eval_traditional: bool = False,
    debug: bool = False,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    results_base_path: str = "/data/cef/dischargeme/results_cef"
):
    """
    Evaluate DischargeME predictions using CEF and traditional metrics.
    
    Performs evaluation of generated Brief Hospital Course by comparing predicted
    and reference hospital courses using CEF (conformity, consistency, coverage) and 
    traditional metrics (BLEU, ROUGE-1, BERTScore).
    
    Args:
        predictions_path (str): Path to predictions JSONL file
        reference_path (str): Path to reference data directory
        model_name (str): Name of the model being evaluated
        worker (str): Hostname of evaluation API server
        slurm_job_name (str): Slurm job name
        port (int): API server port (default: 8893)
        eval_cef (bool): Run CEF evaluation (default: False)
        pipeline_params_path (str): Path to pipeline config YAML
        prompt_catalogue_path (str): Path to prompt templates YAML
        judge_model (str): Model to use for CEF evaluation
        judge_worker (str): Hostname of judge API server
        judge_port (int): Judge API server port
        num_questions_to_generate (int): Questions per sample
        token_per_question (int): Auto-calculate questions from source length
        bootstrap_iterations (int): Bootstrap iterations for confidence intervals
        bootstrap_fraction (float): Bootstrap sample fraction
        eval_traditional (bool): Run traditional metrics (default: True)
        debug (bool): Debug mode - limit to 10 samples
        tokenizer_name (str): HuggingFace tokenizer for token counting
        results_base_path (str): Base path for saving results
    
    Returns:
        None: Saves results to disk and prints statistics to stdout
    
    Example:
        eval_dischargeme(
            predictions_path="/data/cef/dischargeme/dischargeme/DeepSeek-V3.1/predictions/DeepSeek-V3.1_raw_bhc_responses_tmp.jsonl",
            worker="worker-7", 
            eval_cef=True,
            debug=True
        )
    """
    print("Arguments:")
    print(f"  predictions_path: {predictions_path}")
    print(f"  reference_path: {reference_path}")
    print(f"  model_name: {model_name}")
    print(f"  worker: {worker}")
    print(f"  slurm_job_name: {slurm_job_name}")
    print(f"  port: {port}")
    print(f"  eval_cef: {eval_cef}")
    print(f"  judge_model: {judge_model}")
    print(f"  judge_worker: {judge_worker}")
    print(f"  judge_port: {judge_port}")
    print(f"  pipeline_params_path: {pipeline_params_path}")
    print(f"  prompt_catalogue_path: {prompt_catalogue_path}")
    print(f"  num_questions_to_generate: {num_questions_to_generate}")
    print(f"  token_per_question: {token_per_question}")
    print(f"  bootstrap_iterations: {bootstrap_iterations}")
    print(f"  bootstrap_fraction: {bootstrap_fraction}")
    print(f"  eval_traditional: {eval_traditional}")
    print(f"  debug: {debug}")
    print(f"  tokenizer_name: {tokenizer_name}")
    print(f"  results_base_path: {results_base_path}")
    
    # Validate inputs
    if slurm_job_name is not None:
        worker = find_worker(slurm_job_name)
    elif slurm_job_name is None and worker is None and eval_cef:
        raise ValueError("Either slurm_job_name or worker must be provided if eval_cef is True")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    ds = prepare_dischargeme_dataset(
        predictions_path=predictions_path,
        reference_path=reference_path,
        model_name=model_name,
        debug=debug
    )
    
    print(f"\nDataset splits: {list(ds.keys())}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} samples")
    
    # Initialize tokenizer if needed
    tok = None
    if eval_cef:
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Evaluate with CEF
    if eval_cef:
        api_url = f"http://{worker}:{port}/evaluate"
        judge_api_url = f"http://{judge_worker}:{judge_port}/v1"
        pipeline_params = load_yaml(pipeline_params_path)
        
        print("\nRunning CEF evaluation...")
        for split in ds:
            try:
                # For DischargeME, we evaluate predicted BHC (trg) against reference BHC
                # We need to modify the evaluation to compare trg vs reference instead of src vs trg
                ds[split] = ds[split].map(
                    lambda x: evaluate_cef_sample(
                        {'src': x['reference'], 'trg': x['trg']},  # Compare reference vs prediction
                        api_url=api_url, 
                        pipeline_params=pipeline_params, 
                        prompt_catalogue_path=prompt_catalogue_path, 
                        tokenizer=tok, 
                        num_questions_to_generate=num_questions_to_generate, 
                        token_per_question=token_per_question,
                        judge_api_url=judge_api_url,
                        judge_model=judge_model,
                        self_evaluate=None
                    ), 
                    num_proc=16,
                    load_from_cache_file=False
                )
            except Exception as e:
                print(f"Error evaluating {split}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
    
    # Evaluate with traditional metrics
    if eval_traditional:
        print("\nRunning traditional metrics evaluation...")
        for split in ds:
            ds[split] = ds[split].map(
                evaluate_dischargeme_traditional_sample,
                num_proc=1,
                load_from_cache_file=False
            )
    
    # Save results
    random_str = str(uuid.uuid4())
    results_dir = f"{results_base_path}/debug" if debug else f"{results_base_path}/{model_name}_{random_str}"
    os.makedirs(results_dir, exist_ok=True)
    dataset_path = os.path.join(results_dir, "results_ds")
    ds.save_to_disk(dataset_path)
    
    print(f"\nDataset saved to: {dataset_path}")
    
    # Generate and save reports
    if eval_cef or eval_traditional:
        print("\nGenerating evaluation report...")
        stats = generate_dischargeme_report(
            ds,
            predictions_path,
            bootstrap_iterations,
            bootstrap_fraction
        )
        
        # Add metadata
        stats["metadata"] = {
            "model_name": model_name,
            "predictions_path": predictions_path,
            "reference_path": reference_path,
            "eval_cef": eval_cef,
            "eval_traditional": eval_traditional,
            "debug": debug
        }
        
        if eval_cef:
            stats["cef_params"] = {
                "judge_model": judge_model,
                "num_questions_to_generate": num_questions_to_generate,
                "token_per_question": token_per_question,
                "prompt_catalogue_path": prompt_catalogue_path,
                "pipeline_params_path": pipeline_params_path
            }
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(json.dumps(stats, indent=2))
        
        # Save the evaluation report
        report_path = os.path.join(results_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    fire.Fire(eval_dischargeme)

# %%
# Example usage:
# python eval_dischargeme.py \
#   --predictions_path=/data/cef/dischargeme/dischargeme/DeepSeek-V3.1/predictions/DeepSeek-V3.1_raw_bhc_responses_tmp.jsonl \
#   --eval_traditional=True \
#   --debug=True

# python eval_dischargeme.py \
#   --predictions_path=/data/cef/dischargeme/dischargeme/DeepSeek-V3.1/predictions/DeepSeek-V3.1_raw_bhc_responses_tmp.jsonl \
#   --worker=worker-7 \
#   --judge_worker=worker-12 \
#   --eval_cef=True \
#   --eval_traditional=True \
#   --num_questions_to_generate=10 \
#   --debug=True

