# %%
import requests
from datasets import load_from_disk
import sys
import os
from transformers import AutoTokenizer
from utils import evaluate_cef_sample, generate_cef_report, evaluate_traditional_sample, find_worker, TraditionalEvaluator, TranslationQualityEvaluator, CoverageConsistencyEvaluator
import json
import uuid
# Add the parent directory to the path so we can import cef_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cef_framework.utils import load_yaml
import fire
from utils import generate_traditional_report, generate_translation_quality_report
import traceback


def eval_flores(
    worker: str = None,
    slurm_job_name: str = None,
    port: int = 8893,
    eval_cef: bool = False,
    with_reference: bool = False,
    pipeline_params_path: str = "../cef_framework/pipeline_params.yaml",
    prompt_catalogue_path: str = "../cef_framework/prompts/translation.yaml",
    judge_model: str = "/models_llm/Qwen2.5-72B-Instruct",
    judge_worker: str = None,
    judge_port: int = 8892,
    num_questions_to_generate: int = None,
    token_per_question: int = None,
    bootstrap_iterations: int = 100,
    bootstrap_fraction: float = 1,
    self_evaluate: str = None,
    eval_traditional: bool = False,
    reference_translation_path: str = None,
    eval_translation_quality: bool = False,
    translation_quality_model_name: str = "/models_llm/Qwen2.5-72B-Instruct",
    translation_quality_worker: str = None,
    translation_quality_port: int = 8892,
    translation_quality_template_path: str = "translation-quality.jinja",
    translation_quality_use_three_scores: bool = False,
    eval_coverage_consistency: bool = False,
    coverage_consistency_model_name: str = "/models_llm/Qwen2.5-72B-Instruct",
    coverage_consistency_worker: str = None,
    coverage_consistency_port: int = 8892,
    debug: bool = False,
    splits: list[str] = None,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    data_path: str = "/data/cef/flores/flores_original",
    results_base_path: str = "/data/cef/flores/results_cef"
):
    """
    Evaluate FLORES dataset using Cross Examination Framework (CEF).
    
    Performs machine translation quality evaluation by comparing source and target documents
    using AI-generated questions. Measures conformity, consistency, and coverage scores.
    
    Args:
        worker (str): Hostname of evaluation API server
        slurm_job_name (str): Slurm job name
        port (int): API serve   r port (default: 8893)
        eval_cef (bool): Run CEF evaluation (default: True)
        pipeline_params_path (str): Path to pipeline config YAML
        prompt_catalogue_path (str): Path to prompt templates YAML
        num_questions_to_generate (int): Questions per sample (default: 10)
        token_per_question (int, optional): Auto-calculate questions from source length
        bootstrap_iterations (int): Bootstrap iterations for confidence intervals (default: 100)
        bootstrap_fraction (float): Bootstrap sample fraction (default: 1.0)
        eval_traditional (bool): Run traditional metrics (default: False)
        debug (bool): Debug mode - limit to 10 samples per split (default: False)
        tokenizer_name (str): HuggingFace tokenizer for token counting
        data_path (str): Path to FLORES dataset
        results_base_path (str): Base path for saving results
    
    Returns:
        None: Saves results to disk and prints statistics to stdout
    
    Example:
        eval_flores(worker="worker-7", debug=True)
    """
    print("Arguments:")
    print(f"  worker: {worker}")
    print(f"  slurm_job_name: {slurm_job_name}")
    print(f"  port: {port}")
    print(f"  eval_cef: {eval_cef}")
    print(f"  judge_model: {judge_model}")
    print(f"  judge_worker: {judge_worker}")
    print(f"  judge_port: {judge_port}")
    print(f"  pipeline_params_path: {pipeline_params_path}")
    print(f"  prompt_catalogue_path: {prompt_catalogue_path}")
    print(f"  with_reference: {with_reference}")
    print(f"  num_questions_to_generate: {num_questions_to_generate}")
    print(f"  token_per_question: {token_per_question}")
    print(f"  bootstrap_iterations: {bootstrap_iterations}")
    print(f"  bootstrap_fraction: {bootstrap_fraction}")
    print(f"  eval_traditional: {eval_traditional}")
    print(f"  self_evaluate: {self_evaluate}")
    print(f"  reference_translation_path: {reference_translation_path}")
    print(f"  eval_translation_quality: {eval_translation_quality}")
    print(f"  translation_quality_model_name: {translation_quality_model_name}")
    print(f"  translation_quality_worker: {translation_quality_worker}")
    print(f"  translation_quality_port: {translation_quality_port}")
    print(f"  translation_quality_template_path: {translation_quality_template_path}")
    print(f"  translation_quality_use_three_scores: {translation_quality_use_three_scores}")
    print(f"  debug: {debug}")
    print(f"  tokenizer_name: {tokenizer_name}")
    print(f"  data_path: {data_path}")
    print(f"  results_base_path: {results_base_path}")
    print(f"  splits: {splits}")
    if slurm_job_name is not None:
        worker = find_worker(slurm_job_name)
    elif slurm_job_name is None and worker is None and eval_cef:
        raise ValueError("Either slurm_job_name or worker must be provided if eval_cef is True")
    elif translation_quality_worker is None and eval_translation_quality:
        raise ValueError("translation_quality_worker must be provided if eval_translation_quality is True")

    api_url = f"http://{worker}:{port}/evaluate"
    judge_api_url = f"http://{judge_worker}:{judge_port}/v1"
    pipeline_params = load_yaml(pipeline_params_path)
    
    ds = load_from_disk(data_path)
    if with_reference:
        model_name = data_path.split("_")[-1]
        reference_data_path = data_path.replace(model_name, "original")
        ds2 = load_from_disk(reference_data_path)
        for split in ds:
            ds[split] = ds[split].map(
                lambda x, idx: {"src": ds2[split][idx]["trg"]},
                num_proc=16,
                load_from_cache_file=False,
                with_indices=True
            )
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if debug:
        for split in ds:
            ds[split] = ds[split].select(range(10))

    if eval_cef:
        # Parse splits from command line if provided as string
        if splits is not None and isinstance(splits, str):
            selected_splits = json.loads(splits)
        elif splits is not None and isinstance(splits, list):
            selected_splits = splits
        else:
            selected_splits = ds.keys()
        for split in selected_splits:
            try:
                ds[split] = ds[split].map(
                    lambda x: evaluate_cef_sample(
                        x, 
                        api_url=api_url, 
                        pipeline_params=pipeline_params, 
                        prompt_catalogue_path=prompt_catalogue_path, 
                        tokenizer=tok, 
                        num_questions_to_generate=num_questions_to_generate, 
                        token_per_question=token_per_question,
                        judge_api_url=judge_api_url,
                        judge_model=judge_model,
                        self_evaluate=self_evaluate
                    ), 
                    num_proc=16,
                    load_from_cache_file=False
                )
            except Exception as e:
                print(f"Error evaluating {split}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
    
    if eval_traditional:
        if reference_translation_path is None:
            raise ValueError("reference_translation_path must be provided if eval_traditional is True")
        for split in ds:
            evaluator = TraditionalEvaluator(reference_translation_path=reference_translation_path, reference_split=split)
            ds[split] = ds[split].map(
                lambda x: evaluator.evaluate_traditional_sample(x),
                num_proc=1,
                load_from_cache_file=False
            )
    
    if eval_translation_quality:
        evaluator = TranslationQualityEvaluator(
            model_name=translation_quality_model_name,
            worker=translation_quality_worker,
            port=translation_quality_port,
            template_path=translation_quality_template_path,
            use_three_scores=translation_quality_use_three_scores
        )
        for split in ds:
            ds[split] = ds[split].map(
                lambda x: evaluator.evaluate_translation_quality(x, split),
                num_proc=1,
                load_from_cache_file=False
            )

    if eval_coverage_consistency:
        evaluator = CoverageConsistencyEvaluator(
            model_name=coverage_consistency_model_name,
            worker=coverage_consistency_worker,
            port=coverage_consistency_port
        )
        for split in ds:
            ds[split] = ds[split].map(
                lambda x: evaluator.evaluate_coverage_consistency(x, split),
                num_proc=1,
                load_from_cache_file=False
            )
    random_str = str(uuid.uuid4())
    results_dir = f"{results_base_path}/debug" if debug else f"{results_base_path}/{random_str}"
    os.makedirs(results_dir, exist_ok=True)
    dataset_path = os.path.join(results_dir, "results_ds")
    ds.save_to_disk(dataset_path)

    if eval_cef:
        stats = generate_cef_report(
            ds,
            data_path,
            pipeline_params,
            prompt_catalogue_path,
            judge_model,
            num_questions_to_generate, 
            token_per_question,
            bootstrap_iterations,
            bootstrap_fraction
        )
        print(json.dumps(stats, indent=2))
        
        
        # Save the evaluation report
        report_path = os.path.join(results_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save the dataset with evaluation results
        
        print(f"Results saved to: {results_dir}")

    if eval_traditional:
        stats = generate_traditional_report(
            ds,
            data_path,
            bootstrap_iterations,
            bootstrap_fraction
        )
        print(json.dumps(stats, indent=2))
        report_path = os.path.join(results_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {results_dir}")

    if eval_translation_quality:
        stats = generate_translation_quality_report(
            ds,
            data_path,
            bootstrap_iterations,
            bootstrap_fraction,
            use_three_scores=translation_quality_use_three_scores
        )
        print(json.dumps(stats, indent=2))
        report_path = os.path.join(results_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {results_dir}")

    if eval_coverage_consistency:
        # TODO: Implement coverage consistency evaluation
        pass

if __name__ == "__main__":
    fire.Fire(eval_flores)

# %%
# Example usage:
# python eval_flores.py --worker worker-7 --port 8893 --eval_cef --debug
