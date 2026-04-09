import requests
from transformers import AutoTokenizer
import numpy as np
from scipy import stats
import sentencepiece as spm
from sacrebleu.tokenizers import BaseTokenizer
import evaluate
from jinja2 import Template
import time
import os
from ast import literal_eval
from jsonfinder import jsonfinder
from datasets import load_from_disk

def find_worker(slurm_job_name):
    '''
    Find the worker hostname from the slurm job name.
    '''
    # TODO: Implement this
    pass

# class TokenizerSPM(BaseTokenizer):
#     def signature(self):
#         return self.name
#     def __init__(self):
#         self.name = "spm"
#         self.sp = spm.SentencePieceProcessor()
#         model_path = "/data/cef/sacrebleu_tokenizer_spm.model"
#         self.sp.Load(model_path)
#     def __call__(self, line):
#         return self.sp.EncodeAsPieces(line)


class TraditionalEvaluator:
    def __init__(self, reference_translation_path=None, reference_split=None):
        # self.spm_tok = TokenizerSPM()
        # self.bleu = evaluate.load("bleu")
        self.bert_model = "bert-base-multilingual-cased"
        self.bleu = evaluate.load("sacrebleu")
        self.chrf = evaluate.load("chrf")
        self.bertscore = evaluate.load("bertscore")
        self.comet = evaluate.load("comet")
        self.bertscore.compute(predictions=[""], references=[[""]], model_type=self.bert_model)
        reference_translation_ds = load_from_disk(reference_translation_path)[reference_split]
        self.reference_translations = {x["id"]: x["trg"] for x in reference_translation_ds}

    def evaluate_traditional_sample(self, example):
        '''
        Evaluate a sample on traditional metrics like spBLEU, bertscore and chrf++.
        '''
        if example["id"] not in self.reference_translations or not isinstance(self.reference_translations[example["id"]], str):
            print(f"Warning: Reference translation not found for example {example['id']}")
            return {
                "bleu": -1,
                "chrf": -1,
                "bertscore": -1,
                "comet": -1,
                "reference_translation": None
            }
        src = example["src"]
        reference_translation = self.reference_translations[example["id"]]
        trg = example["trg"]
        bleu = self.bleu.compute(predictions=[trg], references=[[reference_translation]], tokenize="flores101")['score']
        chrf = self.chrf.compute(predictions=[trg], references=[[reference_translation]])['score']
        bertscore = self.bertscore.compute(predictions=[trg], references=[[reference_translation]], model_type=self.bert_model)['f1'][0]
        comet = self.comet.compute(predictions=[trg], references=[reference_translation], sources=[src])['mean_score']
        return {
            "bleu": bleu,
            "chrf": chrf,
            "bertscore": bertscore,
            "comet": comet,
            "reference_translation": reference_translation
        }


class TranslationQualityEvaluator:
    def __init__(self, model_name, worker, port, template_path="translation-quality.jinja", use_three_scores=False):
        self.system_prompt = "You are a helpful multilingual assistant."
        self.headers = {"Content-Type": "application/json"}
        self.api_url = f"http://{worker}:{port}/v1/chat/completions"
        self.model_name = model_name
        self.use_three_scores = use_three_scores
        # Determine template path based on use_three_scores flag, unless explicitly provided
        if template_path == "translation-quality.jinja" or (not template_path or template_path == ""):
            # Use default template based on use_three_scores flag
            if self.use_three_scores:
                template_path = "translation-quality2.jinja"
            else:
                template_path = "translation-quality.jinja"
        # If template_path is relative, assume it's in the same directory as this file
        if not os.path.isabs(template_path):
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            except (NameError, AttributeError):
                # __file__ might not be defined in some contexts, use current working directory
                base_dir = os.getcwd()
            template_path = os.path.join(base_dir, template_path)
        # Normalize the path
        template_path = os.path.normpath(template_path)
        # Read template file with proper error handling
        if not os.path.exists(template_path):
            try:
                base_dir_info = os.path.dirname(os.path.abspath(__file__))
            except (NameError, AttributeError):
                base_dir_info = os.getcwd()
            raise FileNotFoundError(f"Template file not found: {template_path} (resolved from base_dir: {base_dir_info})")
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_source = f.read()
        except Exception as e:
            raise IOError(f"Failed to read template file {template_path}: {e}")
        if not template_source or not template_source.strip():
            raise ValueError(f"Template file is empty: {template_path}")
        # Store template source as string (pickleable) instead of compiled Template object
        self.template_source = template_source
        # Cache for compiled template (will be recreated if needed after unpickling)
        self._jinja_template = None
    
    def call_llm_with_retry(self, payload, max_retries=5, timeout=180, retry_delay=30):
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=timeout
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    print(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries + 1}), waiting {retry_delay} seconds...")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print("Max retries reached for rate limit")
                        return None
                
                # Try to parse response
                try:
                    response_data = response.json()["choices"][0]["message"]["content"]
                    parsed_response = self.parse_json_from_response(response_data)
                    return parsed_response
                except ValueError as e:
                    print(f"Failed to parse JSON response (attempt {attempt + 1}/{max_retries + 1}): {response.text}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print("Max retries reached for JSON parsing")
                        return None
                        
            except requests.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Max retries reached for request errors")
                    return None
        
        return None
    
    def parse_json_from_response(self, response_data):
        json_datas = list(jsonfinder(response_data))
        for tmp in json_datas:
            json_data = tmp[2]
            if json_data is None or not isinstance(json_data, dict):
                continue
            
            if self.use_three_scores:
                # Handle translation-quality2.jinja format with 3 scores
                if "scores" in json_data and "analysis" in json_data:
                    scores = json_data["scores"]
                    analysis = json_data["analysis"]
                    if isinstance(scores, dict) and "coverage" in scores and "conformity" in scores and "consistency" in scores:
                        return {
                            "translation_quality_coverage_score": float(scores["coverage"]),
                            "translation_quality_conformity_score": float(scores["conformity"]),
                            "translation_quality_consistency_score": float(scores["consistency"]),
                            "translation_quality_analysis": analysis
                        }
            else:
                # Handle translation-quality.jinja format with single score
                if "score" in json_data and "analysis" in json_data:
                    score = json_data["score"]
                    analysis = json_data["analysis"]
                    return {
                        "translation_quality_score": float(score),
                        "translation_quality_analysis": analysis
                    }
        raise ValueError(f"No JSON found in response: {response_data}")
    
    @property
    def jinja_template(self):
        """Lazy-load template to support pickling for multiprocessing."""
        if self._jinja_template is None:
            try:
                self._jinja_template = Template(self.template_source)
            except Exception as e:
                raise ValueError(f"Failed to create Jinja2 template: {e}")
        return self._jinja_template
    
    def evaluate_translation_quality(self, example, ds_split):
        src_lang, trg_lang = ds_split.split("-")
        rendered_prompt = self.jinja_template.render(
            source_language=src_lang,
            target_language=trg_lang,
            source=example["src"],
            translation=example["trg"],
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": rendered_prompt}],
            "temperature": 0.2,
            "max_tokens": 4000,
            "top_p": 0.9,
        }
        response_data = self.call_llm_with_retry(payload)
        if response_data is None:
            if self.use_three_scores:
                return {
                    "translation_quality_coverage_score": -1,
                    "translation_quality_conformity_score": -1,
                    "translation_quality_consistency_score": -1,
                    "translation_quality_analysis": "Failed to evaluate translation quality"
                }
            else:
                return {
                    "translation_quality_score": -1,
                    "translation_quality_analysis": "Failed to evaluate translation quality"
                }
        return response_data

class CoverageConsistencyEvaluator(TranslationQualityEvaluator):
    def __init__(self, model_name, worker, port, template_path="coverage_consistency.jinja"):
        super().__init__(model_name, worker, port, template_path)
    
    def evaluate_coverage_consistency(self, example, ds_split):
        src_lang, trg_lang = ds_split.split("-")
        rendered_prompt = self.jinja_template.render(
            source_language=src_lang,
            target_language=trg_lang,
            source_text=example["src"],
            translated_text=example["trg"],
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": rendered_prompt}],
            "temperature": 0.2,
            "max_tokens": 4000,
            "top_p": 0.9,
        }
        response_data = self.call_llm_with_retry(payload)
        if response_data is None:
            return {
                "information_loss": [],
                "information_contradiction": []
            }
        return response_data
    
    def parse_json_from_response(self, response_data):
        json_datas = list(jsonfinder(response_data))
        for tmp in json_datas:
            json_data = tmp[2]
            if (json_data is None or not isinstance(json_data, dict) or "information_loss" not in json_data or "information_contradiction" not in json_data):
                continue
            information_loss = json_data["information_loss"]
            information_contradiction = json_data["information_contradiction"]
            return {
                "information_loss": information_loss,
                "information_contradiction": information_contradiction
            }
        raise ValueError(f"No JSON found in response: {response_data}")
    

def evaluate_cef_sample(example, api_url, pipeline_params, prompt_catalogue_path, headers={"Content-Type": "application/json"}, request_timeout=500, tokenizer=None, judge_model="/models_llm/Qwen2.5-72B-Instruct", judge_api_url="http://worker-12:8892/v1/", num_questions_to_generate=None, token_per_question=None, self_evaluate=False):
    '''
    Evaluate CEF (Cross Examination Framework) on a sample.
    
    This function sends a document pair (source and target) to an evaluation API endpoint
    that generates questions and evaluates the quality of the target document against the source.
    
    Args:
        example (dict): Dictionary containing the document pair with keys:
            - "src": Source document text
            - "trg": Target document text
        api_url (str): URL of the evaluation API endpoint
        pipeline_params (dict): Configuration parameters for the evaluation pipeline
        prompt_catalogue_path (str): Path to the prompt catalogue file
        headers (dict, optional): HTTP headers for the API request. Defaults to {"Content-Type": "application/json"}
        request_timeout (int, optional): Timeout for the API request in seconds. Defaults to 180
        tokenizer: Tokenizer instance for calculating token counts. Required if token_per_question is provided
        num_questions_to_generate (int, optional): Number of questions to generate for evaluation
        token_per_question (int, optional): Number of tokens per question to calculate question count
    
    Returns:
        dict: Evaluation results containing:
            - "scores": Dictionary of evaluation scores
            - "details": Dictionary of detailed evaluation information
    
    Raises:
        ValueError: If neither num_questions_to_generate nor token_per_question is provided,
                   or if token_per_question is provided without a tokenizer
        requests.RequestException: If the API request fails
    
    Note:
        Either num_questions_to_generate or token_per_question must be provided.
        If token_per_question is used, the number of questions is calculated as:
        questions_count = len(tokenizer.encode(example["src"])) // token_per_question
    '''    
    if num_questions_to_generate is not None:
        questions_count = num_questions_to_generate
    elif token_per_question is not None and tokenizer is not None:
        tokens = len(tokenizer.encode(example["src"]))
        questions_count = tokens // token_per_question
    else: 
        raise ValueError("Either num_questions_to_generate or token_per_question must be provided")
    
    # Update pipeline parameters
    pipeline_params["gen_qa_document"]["num_questions_to_generate"] = questions_count
    pipeline_params["gen_qa_summary"]["num_questions_to_generate"] = questions_count
    
    pipeline_params["gen_qa_document"]["llm_params"]["llm_model"] = judge_model
    pipeline_params["gen_qa_summary"]["llm_params"]["llm_model"] = judge_model
    pipeline_params["cross_examine"]["llm_params"]["llm_model"] = judge_model
    
    pipeline_params["base_endpoint"] = judge_api_url
    pipeline_params["gen_qa_document"]["base_endpoint"] = judge_api_url
    pipeline_params["gen_qa_summary"]["base_endpoint"] = judge_api_url
    pipeline_params["cross_examine"]["base_endpoint"] = judge_api_url
    payload = {
        "original_document": example["src"] if not self_evaluate else example[self_evaluate],
        "generated_document": example["trg"] if not self_evaluate else example[self_evaluate],
        "prompt_catalogue_path": prompt_catalogue_path,
        "config": pipeline_params
    }
    # print(payload)
    response = requests.post(api_url, headers=headers, json=payload, timeout=request_timeout)
    response_data = response.json()
    scores = response_data.get("scores", {})
    details = response_data.get("details", {})
    return {"cef_scores": scores, "cef_details": details}

def evaluate_traditional_sample(example):
    '''
    Evaluate a sample on traditional metrics on spBLEU, bertscore and chrf++.
    '''
    payload = {
        "original_document": example["src"],
        "generated_document": example["trg"]}

def calculate_stats(scores, score_name, bootstrap_iterations=100, bootstrap_fraction=1):
    if not scores:
        return {score_name: {"error": "No scores found"}}
    n_iterations = bootstrap_iterations
    n_samples = int(len(scores) * bootstrap_fraction)
    initial_mean = np.mean(scores)
    bootstrap_means = []
    
    for _ in range(n_iterations):
        # Bootstrap sampling with replacement
        bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate confidence interval from bootstrap distribution
    sorted_means = np.sort(np.array(bootstrap_means))
    percentile = 95
    conf_interval = [
        sorted_means[int((1-percentile/100) * len(sorted_means))], 
        sorted_means[int((percentile/100) * len(sorted_means))]
    ]
    
    mean_score = np.mean(bootstrap_means)
    std_score = np.std(bootstrap_means, ddof=1)
    n = len(scores)
    if score_name == "bleu_score" or score_name == "chrf_score":
        return {
            "mean": round(mean_score, 2),
            "confidence_interval": f"+{round(conf_interval[1] - mean_score, 2)}/-{round(mean_score - conf_interval[0], 2)}",
            "sample_size": n
        }
    else:
        return {
            "mean": round(mean_score * 100, 2),
            "confidence_interval": f"+{round((conf_interval[1] - mean_score) * 100, 2)}/-{round((mean_score - conf_interval[0]) * 100, 2)}",
            "sample_size": n
        }


def generate_cef_report(ds, data_path, pipeline_params, prompt_catalogue_path, judge_model, num_questions_to_generate, token_per_question, bootstrap_iterations=100, bootstrap_fraction=1):
    """
    Calculate mean and confidence intervals for conformity, consistency, and coverage scores.
    
    Args:
        ds: Dataset containing evaluation results with 'scores' field
        
    Returns:
        dict: Dictionary containing statistics for each score type
    """
    # Extract scores from the dataset
    report = {}
    report["scores"] = {}
    for split in ds:
        report["scores"][split] = {}
        if 'cef_scores' in ds[split][0]:
            conformity_scores = [item['cef_scores']['conformity_score'] for item in ds[split]]
            consistency_scores = [item['cef_scores']['consistency_score'] for item in ds[split]]
            coverage_scores = [item['cef_scores']['coverage_score'] for item in ds[split]]
            report["scores"][split]["conformity_score"] = calculate_stats(conformity_scores, "conformity_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["consistency_score"] = calculate_stats(consistency_scores, "consistency_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["coverage_score"] = calculate_stats(coverage_scores, "coverage_score", bootstrap_iterations, bootstrap_fraction)
        else:
            print(f"Warning: Dataset split '{split}' does not contain CEF scores")
            continue
    report["cef_params"] = {
        "data_path": data_path,
        "prompt_catalogue_path": prompt_catalogue_path,
        "judge_model": judge_model,
        "num_questions_to_generate": num_questions_to_generate,
        "token_per_question": token_per_question,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_fraction": bootstrap_fraction
    }
    return report

def generate_translation_quality_report(ds, data_path, bootstrap_iterations=100, bootstrap_fraction=1, use_three_scores=False):
    """
    Calculate mean and confidence intervals for translation quality scores.
    Supports both single score format and three-score format (coverage, conformity, consistency).
    """
    report = {}
    report["scores"] = {}
    for split in ds:
        report["scores"][split] = {}
        
        if use_three_scores:
            # Handle three-score format (coverage, conformity, consistency)
            if "translation_quality_coverage_score" in ds[split][0]:
                coverage_scores = [item.get('translation_quality_coverage_score', -1) for item in ds[split]]
                conformity_scores = [item.get('translation_quality_conformity_score', -1) for item in ds[split]]
                consistency_scores = [item.get('translation_quality_consistency_score', -1) for item in ds[split]]
                
                valid_coverage = [float(s) for s in coverage_scores if isinstance(s, (int, float)) and s >= 0]
                valid_conformity = [float(s) for s in conformity_scores if isinstance(s, (int, float)) and s >= 0]
                valid_consistency = [float(s) for s in consistency_scores if isinstance(s, (int, float)) and s >= 0]
                
                if valid_coverage:
                    report["scores"][split]["translation_quality_coverage_score"] = calculate_stats(valid_coverage, "translation_quality_coverage_score", bootstrap_iterations, bootstrap_fraction)
                if valid_conformity:
                    report["scores"][split]["translation_quality_conformity_score"] = calculate_stats(valid_conformity, "translation_quality_conformity_score", bootstrap_iterations, bootstrap_fraction)
                if valid_consistency:
                    report["scores"][split]["translation_quality_consistency_score"] = calculate_stats(valid_consistency, "translation_quality_consistency_score", bootstrap_iterations, bootstrap_fraction)
            else:
                print(f"Warning: Dataset split '{split}' does not contain three-score translation quality scores")
                continue
        else:
            # Handle single score format
            if "translation_quality_score" in ds[split][0]:
                translation_quality_scores = [item['translation_quality_score'] for item in ds[split]]
                valid_scores = []
                for score in translation_quality_scores:
                    if isinstance(score, (int, float)) and score >= 0:
                        valid_scores.append(float(score))
                if not valid_scores:
                    print(f"Warning: Dataset split '{split}' does not contain valid translation quality scores")
                    continue
                translation_quality_scores = valid_scores
                report["scores"][split]["translation_quality_score"] = calculate_stats(translation_quality_scores, "translation_quality_score", bootstrap_iterations, bootstrap_fraction)
            else:
                print(f"Warning: Dataset split '{split}' does not contain translation quality scores")
                continue
    report["translation_quality_params"] = {
        "data_path": data_path,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_fraction": bootstrap_fraction,
        "use_three_scores": use_three_scores
    }
    return report

def generate_traditional_report(ds, data_path, bootstrap_iterations=100, bootstrap_fraction=1):
    """
    Calculate mean and confidence intervals for traditional metrics.
    """
    report = {}
    report["scores"] = {}
    for split in ds:
        report["scores"][split] = {}
        if "bleu" in ds[split][0] and "chrf" in ds[split][0] and "bertscore" in ds[split][0]:
            bleu_scores = [item['bleu'] for item in ds[split]]
            chrf_scores = [item['chrf'] for item in ds[split]]
            bertscore_scores = [item['bertscore'] for item in ds[split]]
            comet_scores = [item['comet'] for item in ds[split]]
            report["scores"][split]["bleu_score"] = calculate_stats(bleu_scores, "bleu_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["chrf_score"] = calculate_stats(chrf_scores, "chrf_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["bertscore_score"] = calculate_stats(bertscore_scores, "bertscore_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["comet_score"] = calculate_stats(comet_scores, "comet_score", bootstrap_iterations, bootstrap_fraction)
        else:
            print(f"Warning: Dataset split '{split}' does not contain traditional metrics")
            continue
    report["traditional_params"] = {
        "data_path": data_path,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_fraction": bootstrap_fraction
    }
    return report


class DischargeMEEvaluator:
    """Evaluator for DischargeME dataset with traditional metrics (BLEU, ROUGE-1, BERTScore)."""
    
    def __init__(self):
        # Don't load evaluate modules in __init__ to make class pickle-friendly
        self._bleu = None
        self._rouge = None
        self._bertscore = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of evaluate modules for pickle compatibility."""
        if not self._initialized:
            self._bleu = evaluate.load("sacrebleu")
            self._rouge = evaluate.load("rouge")
            self._bertscore = evaluate.load("bertscore")
            # Warm up bertscore with empty prediction
            self._bertscore.compute(predictions=[""], references=[[""]], model_type="microsoft/deberta-xlarge-mnli")
            self._initialized = True
    
    def evaluate_sample(self, example):
        """
        Evaluate a single sample with BLEU, ROUGE-1, and BERTScore.
        
        Args:
            example (dict): Dictionary containing:
                - "prediction": Generated brief hospital course
                - "reference": Reference brief hospital course
                
        Returns:
            dict: Dictionary with evaluation metrics
        """
        self._ensure_initialized()
        
        prediction = example.get("prediction", "")
        reference = example.get("reference", "")
        
        if not prediction or not reference:
            return {
                "bleu": -1,
                "rouge1": -1,
                "bertscore_f1": -1
            }
        
        # Calculate BLEU
        bleu_score = self._bleu.compute(predictions=[prediction], references=[[reference]])['score']
        
        # Calculate ROUGE-1
        rouge_scores = self._rouge.compute(predictions=[prediction], references=[reference])
        rouge1_score = rouge_scores['rouge1']
        
        # Calculate BERTScore F1
        bertscore_results = self._bertscore.compute(
            predictions=[prediction], 
            references=[reference], 
            model_type="microsoft/deberta-xlarge-mnli"
        )
        bertscore_f1 = bertscore_results['f1'][0]
        
        return {
            "bleu": bleu_score,
            "rouge1": rouge1_score,
            "bertscore_f1": bertscore_f1
        }


def generate_dischargeme_report(ds, data_path, bootstrap_iterations=100, bootstrap_fraction=1):
    """
    Calculate mean and confidence intervals for DischargeME metrics.
    
    Args:
        ds: Dataset containing evaluation results
        data_path (str): Path to the data
        bootstrap_iterations (int): Number of bootstrap iterations
        bootstrap_fraction (float): Fraction of data to use in each bootstrap
        
    Returns:
        dict: Dictionary containing statistics for each metric
    """
    report = {}
    report["scores"] = {}
    
    for split in ds:
        report["scores"][split] = {}
        
        # Check if traditional metrics exist
        if "bleu" in ds[split][0]:
            bleu_scores = [item['bleu'] for item in ds[split] if item['bleu'] >= 0]
            rouge1_scores = [item['rouge1'] for item in ds[split] if item['rouge1'] >= 0]
            bertscore_f1_scores = [item['bertscore_f1'] for item in ds[split] if item['bertscore_f1'] >= 0]
            
            if bleu_scores:
                report["scores"][split]["bleu_score"] = calculate_stats(bleu_scores, "bleu_score", bootstrap_iterations, bootstrap_fraction)
            if rouge1_scores:
                report["scores"][split]["rouge1_score"] = calculate_stats(rouge1_scores, "rouge1_score", bootstrap_iterations, bootstrap_fraction)
            if bertscore_f1_scores:
                report["scores"][split]["bertscore_f1_score"] = calculate_stats(bertscore_f1_scores, "bertscore_f1_score", bootstrap_iterations, bootstrap_fraction)
        
        # Check if CEF metrics exist
        if "cef_scores" in ds[split][0]:
            conformity_scores = [item['cef_scores']['conformity_score'] for item in ds[split]]
            consistency_scores = [item['cef_scores']['consistency_score'] for item in ds[split]]
            coverage_scores = [item['cef_scores']['coverage_score'] for item in ds[split]]
            
            report["scores"][split]["conformity_score"] = calculate_stats(conformity_scores, "conformity_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["consistency_score"] = calculate_stats(consistency_scores, "consistency_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["coverage_score"] = calculate_stats(coverage_scores, "coverage_score", bootstrap_iterations, bootstrap_fraction)
    
    report["evaluation_params"] = {
        "data_path": data_path,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_fraction": bootstrap_fraction
    }
    
    return report

