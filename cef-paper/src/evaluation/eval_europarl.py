import requests
from datasets import load_from_disk
import sys
import os
from transformers import AutoTokenizer
from utils import evaluate_cef_sample, generate_cef_report, evaluate_traditional_sample
import json
import uuid
# Add the parent directory to the path so we can import cef_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cef_framework.utils import load_yaml
import fire


def eval_europarl(
    worker: str = None,
    port: int = 8893,
    eval_cef: bool = True,
    pipeline_params_path: str = "../cef_framework/pipeline_params.yaml",
    prompt_catalogue_path: str = "../cef_framework/prompts/translation.yaml",
    num_questions_to_generate: int = 10,
    token_per_question: int = None,
    bootstrap_iterations: int = 100,
    bootstrap_fraction: float = 1,
    eval_traditional: bool = False,
    debug: bool = False,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    data_path: str = "/data/cef/europarl/filtered_1381/europarl_Llama3.2-1b-Instruct_all_languages",
    results_base_path: str = "/data/cef/europarl/filtered_1381/europarl_Llama3.2-1b-Instruct_all_languages/results"
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
        ds = ds.map(
            lambda x: evaluate_cef_sample(
                x, 
                api_url=api_url, 
                pipeline_params=pipeline_params, 
                prompt_catalogue_path=prompt_catalogue_path, 
                tokenizer=tok, 
                num_questions_to_generate=num_questions_to_generate, 
                token_per_question=token_per_question
            ), 
            num_proc=16
        )
        
        stats = generate_cef_report(
            ds, 
            pipeline_params,
            prompt_catalogue_path, 
            num_questions_to_generate, 
            token_per_question, 
            bootstrap_iterations, 
            bootstrap_fraction
        )
        
        print(json.dumps(stats, indent=2))
        
        random_str = str(uuid.uuid4())
        results_dir = f"{results_base_path}/cef_debug" if debug else f"{results_base_path}/cef_{random_str}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the evaluation report
        report_path = os.path.join(results_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save the dataset with evaluation results
        dataset_path = os.path.join(results_dir, "europarl_eval_results")
        ds.save_to_disk(dataset_path)
        
        print(f"Results saved to: {results_dir}")

    if eval_traditional:
        # TODO: Implement traditional evaluation if needed
        pass
    


if __name__ == "__main__":
    fire.Fire(eval_europarl)
    
#python eval_europarl.py --worker worker-2 --port 8893 --eval_cef --debug 


