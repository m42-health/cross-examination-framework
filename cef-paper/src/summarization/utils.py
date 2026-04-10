import requests
from transformers import AutoTokenizer
import numpy as np
from scipy import stats
import evaluate
from jinja2 import Template
import time
import os
from ast import literal_eval
from jsonfinder import jsonfinder
from datasets import load_from_disk
import yaml


def find_worker(slurm_job_name):
    '''
    Find the worker hostname from the slurm job name.
    '''
    # TODO: Implement this
    pass


class SummarizationTraditionalEvaluator:
    """Evaluator for summarization with traditional metrics (ROUGE, BERTScore, BLEU)."""
    
    def __init__(self, reference_path=None, reference_split=None):
        self.bert_model = "bert-base-uncased"
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.bleu = evaluate.load("bleu")
        # Warm up bertscore
        self.bertscore.compute(predictions=[""], references=[[""]], model_type=self.bert_model)
        
        if reference_path and reference_split:
            reference_ds = load_from_disk(reference_path)[reference_split]
            self.reference_summaries = {x["id"]: x["trg"] for x in reference_ds}
        else:
            self.reference_summaries = None

    def evaluate_sample(self, example):
        '''
        Evaluate a sample on traditional summarization metrics.
        '''
        if self.reference_summaries:
            if example["id"] not in self.reference_summaries:
                print(f"Warning: Reference summary not found for example {example['id']}")
                return {
                    "rouge1": -1,
                    "rouge2": -1,
                    "rougeL": -1,
                    "bertscore": -1,
                    "bleu": -1,
                    "reference_summary": None
                }
            reference = self.reference_summaries[example["id"]]
        else:
            # Use the original trg as reference if no separate reference path
            reference = example.get("reference_trg", example.get("trg"))
        
        prediction = example["trg"]
        
        if not prediction or not reference:
            return {
                "rouge1": -1,
                "rouge2": -1,
                "rougeL": -1,
                "bertscore": -1,
                "bleu": -1,
                "reference_summary": reference
            }
        
        rouge_scores = self.rouge.compute(predictions=[prediction], references=[reference])
        bertscore = self.bertscore.compute(
            predictions=[prediction], 
            references=[[reference]], 
            model_type=self.bert_model
        )['f1'][0]
        
        # BLEU score - expects references as list of lists
        try:
            bleu_score = self.bleu.compute(
                predictions=[prediction], 
                references=[[reference]]
            )['bleu']
        except Exception:
            # BLEU can fail on very short texts
            bleu_score = -1
        
        return {
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL'],
            "bertscore": bertscore,
            "bleu": bleu_score,
            "reference_summary": reference
        }


class SummarizationQualityEvaluator:
    """LLM-based summarization quality evaluator."""
    
    def __init__(self, model_name, worker, port, prompt_path):
        self.system_prompt = "You are a helpful assistant specialized in summarization evaluation."
        self.headers = {"Content-Type": "application/json"}
        self.api_url = f"http://{worker}:{port}/v1/chat/completions"
        self.model_name = model_name
        
        with open(prompt_path) as f:
            prompts = yaml.safe_load(f)
        self.qa_gen_article_prompt = prompts["user_instruction_set"]["qa_gen_article_v0"]
        self.qa_gen_summary_prompt = prompts["user_instruction_set"]["qa_gen_summary_v0"]
        self.gen_answers_prompt = prompts["user_instruction_set"]["gen_answers_v0"]
        
        # Store template source for pickling
        self._template_source = None
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
                
                if response.status_code == 429:
                    print(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries + 1}), waiting {retry_delay} seconds...")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return None
                
                try:
                    response_data = response.json()["choices"][0]["message"]["content"]
                    return response_data
                except Exception as e:
                    print(f"Failed to parse response (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return None
                        
            except requests.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                else:
                    return None
        
        return None
    
    def parse_questions_from_response(self, response_data):
        """Parse JSON question list from LLM response."""
        if response_data is None:
            return None
        json_datas = list(jsonfinder(response_data))
        for tmp in json_datas:
            json_data = tmp[2]
            if json_data is None or not isinstance(json_data, list):
                continue
            valid_questions = []
            for item in json_data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    valid_questions.append(item)
            if valid_questions:
                return valid_questions
        return None
    
    def parse_answer_from_response(self, response_data):
        """Parse YES/NO/IDK answer from LLM response."""
        if response_data is None:
            return None
        response_data = response_data.strip().upper()
        if "YES" in response_data:
            return "YES"
        elif "NO" in response_data:
            return "NO"
        elif "IDK" in response_data:
            return "IDK"
        return None
    
    def generate_questions(self, text, num_questions, from_summary=False):
        """Generate questions from text (article or summary)."""
        if from_summary:
            prompt = self.qa_gen_summary_prompt.format(text=text, num_questions=num_questions)
        else:
            prompt = self.qa_gen_article_prompt.format(text=text, num_questions=num_questions)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 8192,
            "top_p": 0.9,
        }
        
        response = self.call_llm_with_retry(payload)
        return self.parse_questions_from_response(response)
    
    def answer_question(self, text, question):
        """Answer a question based on text."""
        prompt = self.gen_answers_prompt.format(text=text, question=question)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 100,
            "top_p": 0.9,
        }
        
        response = self.call_llm_with_retry(payload)
        return self.parse_answer_from_response(response)
    
    def evaluate_cef(self, example, num_questions_src=10, num_questions_trg=5):
        """
        Evaluate summarization using Cross-Examination Framework.
        
        Coverage: Questions from article, answered using summary (how much of article is in summary)
        Conformity: Questions from summary, answered using article (is summary faithful to article)
        Consistency: Questions from summary, answered using summary (internal consistency)
        """
        src = example["src"]  # Article
        trg = example["trg"]  # Generated summary
        
        # Generate questions from article (for coverage)
        questions_from_src = self.generate_questions(src, num_questions_src, from_summary=False)
        
        # Generate questions from summary (for conformity and consistency)
        questions_from_trg = self.generate_questions(trg, num_questions_trg, from_summary=True)
        
        # Calculate Coverage: Questions from article answered using summary
        coverage_answers = []
        if questions_from_src:
            for q in questions_from_src:
                answer = self.answer_question(trg, q["question"])
                coverage_answers.append(answer)
        
        # Calculate Conformity: Questions from summary answered using article
        conformity_answers = []
        if questions_from_trg:
            for q in questions_from_trg:
                answer = self.answer_question(src, q["question"])
                conformity_answers.append(answer)
        
        # Calculate Consistency: Questions from summary answered using summary
        consistency_answers = []
        if questions_from_trg:
            for q in questions_from_trg:
                answer = self.answer_question(trg, q["question"])
                consistency_answers.append(answer)
        
        # Calculate scores
        def calc_score(answers):
            if not answers:
                return 0.0
            yes_count = sum(1 for a in answers if a == "YES")
            return yes_count / len(answers)
        
        coverage_score = calc_score(coverage_answers)
        conformity_score = calc_score(conformity_answers)
        consistency_score = calc_score(consistency_answers)
        
        return {
            "cef_scores": {
                "coverage_score": coverage_score,
                "conformity_score": conformity_score,
                "consistency_score": consistency_score
            },
            "cef_details": {
                "questions_from_src": questions_from_src,
                "questions_from_trg": questions_from_trg,
                "coverage_answers": coverage_answers,
                "conformity_answers": conformity_answers,
                "consistency_answers": consistency_answers
            }
        }


def evaluate_cef_sample(example, evaluator, num_questions_src=10, num_questions_trg=5):
    """Wrapper function for dataset.map()"""
    return evaluator.evaluate_cef(example, num_questions_src, num_questions_trg)


def calculate_stats(scores, score_name, bootstrap_iterations=100, bootstrap_fraction=1):
    """Calculate bootstrap statistics for scores."""
    if not scores:
        return {score_name: {"error": "No scores found"}}
    
    n_iterations = bootstrap_iterations
    n_samples = int(len(scores) * bootstrap_fraction)
    bootstrap_means = []
    
    for _ in range(n_iterations):
        bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    sorted_means = np.sort(np.array(bootstrap_means))
    percentile = 95
    conf_interval = [
        sorted_means[int((1-percentile/100) * len(sorted_means))], 
        sorted_means[int((percentile/100) * len(sorted_means))]
    ]
    
    mean_score = np.mean(bootstrap_means)
    
    return {
        "mean": round(mean_score * 100, 2),
        "confidence_interval": f"+{round((conf_interval[1] - mean_score) * 100, 2)}/-{round((mean_score - conf_interval[0]) * 100, 2)}",
        "sample_size": len(scores)
    }


def generate_cef_report(ds, data_path, num_questions_src, num_questions_trg, bootstrap_iterations=100, bootstrap_fraction=1):
    """Generate CEF evaluation report for summarization."""
    report = {}
    report["scores"] = {}
    
    for split in ds:
        report["scores"][split] = {}
        if 'cef_scores' in ds[split][0]:
            coverage_scores = [item['cef_scores']['coverage_score'] for item in ds[split]]
            conformity_scores = [item['cef_scores']['conformity_score'] for item in ds[split]]
            consistency_scores = [item['cef_scores']['consistency_score'] for item in ds[split]]
            
            report["scores"][split]["coverage_score"] = calculate_stats(coverage_scores, "coverage_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["conformity_score"] = calculate_stats(conformity_scores, "conformity_score", bootstrap_iterations, bootstrap_fraction)
            report["scores"][split]["consistency_score"] = calculate_stats(consistency_scores, "consistency_score", bootstrap_iterations, bootstrap_fraction)
        else:
            print(f"Warning: Dataset split '{split}' does not contain CEF scores")
            continue
    
    report["cef_params"] = {
        "data_path": data_path,
        "num_questions_src": num_questions_src,
        "num_questions_trg": num_questions_trg,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_fraction": bootstrap_fraction
    }
    return report


def generate_traditional_report(ds, data_path, bootstrap_iterations=100, bootstrap_fraction=1):
    """Generate traditional metrics report for summarization."""
    report = {}
    report["scores"] = {}
    
    for split in ds:
        report["scores"][split] = {}
        if "rouge1" in ds[split][0]:
            rouge1_scores = [item['rouge1'] for item in ds[split] if item['rouge1'] >= 0]
            rouge2_scores = [item['rouge2'] for item in ds[split] if item['rouge2'] >= 0]
            rougeL_scores = [item['rougeL'] for item in ds[split] if item['rougeL'] >= 0]
            bertscore_scores = [item['bertscore'] for item in ds[split] if item['bertscore'] >= 0]
            bleu_scores = [item['bleu'] for item in ds[split] if item.get('bleu', -1) >= 0]
            
            if rouge1_scores:
                report["scores"][split]["rouge1_score"] = calculate_stats(rouge1_scores, "rouge1_score", bootstrap_iterations, bootstrap_fraction)
            if rouge2_scores:
                report["scores"][split]["rouge2_score"] = calculate_stats(rouge2_scores, "rouge2_score", bootstrap_iterations, bootstrap_fraction)
            if rougeL_scores:
                report["scores"][split]["rougeL_score"] = calculate_stats(rougeL_scores, "rougeL_score", bootstrap_iterations, bootstrap_fraction)
            if bertscore_scores:
                report["scores"][split]["bertscore_score"] = calculate_stats(bertscore_scores, "bertscore_score", bootstrap_iterations, bootstrap_fraction)
            if bleu_scores:
                report["scores"][split]["bleu_score"] = calculate_stats(bleu_scores, "bleu_score", bootstrap_iterations, bootstrap_fraction)
        else:
            print(f"Warning: Dataset split '{split}' does not contain traditional metrics")
            continue
    
    report["traditional_params"] = {
        "data_path": data_path,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_fraction": bootstrap_fraction
    }
    return report

