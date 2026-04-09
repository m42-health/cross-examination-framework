import requests
from datasets import load_from_disk
import sys
import os
import requests
from datasets import load_from_disk
import sys
import os
from transformers import AutoTokenizer
from utils import evaluate_cef_sample, generate_cef_report, evaluate_traditional_sample, find_worker, TraditionalEvaluator, TranslationQualityEvaluator
import json
import uuid
from transformers import AutoTokenizer
from utils import evaluate_cef_sample, generate_cef_report, evaluate_traditional_sample
from utils import evaluate_cef_sample, generate_cef_report, evaluate_traditional_sample, find_worker, TraditionalEvaluator, TranslationQualityEvaluator
import json
import uuid
# Add the parent directory to the path so we can import cef_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cef_framework.utils import load_yaml
import fire


def eval_europarl(
    worker: str = "worker-12",
    slurm_job_name: str = None,
    port: int = 8893,
    eval_cef: bool = False,
    translation_column= "fr_iter_1",
    pipeline_params_path: str = "../cef_framework/pipeline_params.yaml",
    prompt_catalogue_path: str = "../cef_framework/prompts/translation.yaml",
    num_questions_to_generate: int = 10,
    token_per_question: int = None,
    bootstrap_iterations: int = 100,
    bootstrap_fraction: float = 1,
    eval_traditional: bool = True,
    debug: bool = False,
    reference_translation_path: str = "/data/cef/NTREX",
    eval_translation_quality: bool = False,
    translation_quality_worker: str = "worker-12",
    translation_quality_port: int = 8892,
    translation_quality_model_name: str = "/models_llm/Qwen2.5-72B-Instruct",
    tokenizer_name: str = "Helsinki-NLP/opus-mt-en-fr",
    #tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
    data_path: str = "/data/cef/europarl/ntrex_helsinki_translation/ar_jp_languages",
    results_base_path: str = "/data/cef/europarl/ntrex_helsinki_translation/ar_jp_languages/results_trad"
): 
    """
    Evaluate Europarl dataset using Cross Examination Framework (CEF).
    
    Performs machine translation quality evaluation by comparing source and target documents
    using AI-generated questions. Measures conformity, consistency, and coverage scores.
    
    Args:
        worker (str): Hostname of evaluation API server
        port (int): API server port (default: 8893)
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
        data_path (str): Path to Europarl dataset
        results_base_path (str): Base path for saving results
    
    Returns:
        None: Saves results to disk and prints statistics to stdout
    
    Example:
        eval_europarl(worker="worker-7", debug=True)
    """
    api_url = f"http://{worker}:{port}/evaluate"
    pipeline_params = load_yaml(pipeline_params_path)
    
    ds = load_from_disk(data_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if debug:
        for split in ds:
            ds[split] = ds[split].select(range(10))

    if eval_cef:
        for split in ds:
            ds = ds.map(
                lambda x: evaluate_cef_sample(
                    x, 
                    api_url=api_url, 
                    pipeline_params=pipeline_params, 
                    prompt_catalogue_path=prompt_catalogue_path, 
                    tokenizer=tok, 
                    num_questions_to_generate=num_questions_to_generate, 
                    token_per_question=token_per_question,
                    #translation_column=split_translation_columns["split"]
                ), 
                num_proc=16,
                load_from_cache_file=False
            )
    
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
            port=translation_quality_port
        )
        for split in ds:
            ds[split] = ds[split].map(
                lambda x: evaluator.evaluate_translation_quality(x, split),
                num_proc=1,
                load_from_cache_file=False
            )

    random_str = str(uuid.uuid4())
    results_dir = f"{results_base_path}/debug" if debug else f"{results_base_path}/{random_str}"
    os.makedirs(results_dir, exist_ok=True)
    dataset_path = os.path.join(results_dir, "europarl_eval_results")
    ds.save_to_disk(dataset_path)

    if eval_cef:
        stats = generate_cef_report(
            ds,
            data_path,
            pipeline_params,
            prompt_catalogue_path,
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
            bootstrap_fraction
        )
        print(json.dumps(stats, indent=2))
        report_path = os.path.join(results_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    fire.Fire(eval_europarl)

        
#python eval_europarl_hels.py --worker worker-2 --port 8893 --eval_cef --debug 
#python eval_europarl.py --worker worker-12 --port 8893 --eval_translation_quality --debug

#python eval_europarl_hels.py --worker worker-3 --port 8893 --eval_trad --debug  


