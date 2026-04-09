"""
Utility classes for CEF and traditional evaluation of boxscore-to-summary generation.
Handles quality evaluation for NBA game summaries generated from boxscore data.
"""

import os
import time
import json
import requests
import yaml
import logging
import numpy as np
import evaluate
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BoxscoreQualityEvaluator:
    """
    Evaluates boxscore-to-summary quality using the CEF framework.
    
    Computes:
    - Coverage: What proportion of boxscore facts are in the summary?
    - Faithfulness: Is the summary accurate to the boxscore?
    - Consistency: Is the summary internally consistent?
    - Conciseness: Length ratio of summary to boxscore
    """
    
    def __init__(
        self,
        judge_model: str,
        judge_worker: str,
        judge_port: int,
        prompt_catalogue_path: str,
        num_questions_src: int = 10,
        num_questions_trg: int = 10
    ):
        self.judge_model = judge_model
        self.api_url = f"http://{judge_worker}:{judge_port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        self.num_questions_src = num_questions_src
        self.num_questions_trg = num_questions_trg
        
        # Load prompts
        with open(prompt_catalogue_path) as f:
            prompts = yaml.safe_load(f)
        
        self.system_prompt = prompts["system_instruction_set"]["basic_v0"]
        self.qa_gen_boxscore = prompts["user_instruction_set"]["qa_gen_boxscore_v0"]
        self.qa_gen_summary = prompts["user_instruction_set"]["qa_gen_summary_v0"]
        self.gen_answers = prompts["user_instruction_set"]["gen_answers_v0"]
    
    def _call_llm(self, messages: List[Dict], max_retries: int = 5, timeout: int = 180) -> Optional[str]:
        """Make LLM API call with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.judge_model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 8192,
                    "top_p": 0.9
                }
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=timeout)
                
                if response.status_code == 429:
                    if attempt < max_retries:
                        time.sleep(30)
                        continue
                    return None
                
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(10)
                    continue
                return None
        return None
    
    def _parse_questions(self, response: str) -> List[Dict]:
        """Parse questions from LLM response."""
        if not response:
            return []
        
        from jsonfinder import jsonfinder
        try:
            for _, _, json_data in jsonfinder(response):
                if isinstance(json_data, list):
                    valid = [q for q in json_data if isinstance(q, dict) and "question" in q]
                    if valid:
                        return valid
        except Exception as e:
            logger.warning(f"Failed to parse questions: {e}")
        return []
    
    def _parse_answer(self, response: str) -> str:
        """Parse answer from LLM response."""
        if not response:
            return "IDK"
        
        response = response.strip().upper()
        if "YES" in response:
            return "YES"
        elif "NO" in response:
            return "NO"
        return "IDK"
    
    def generate_questions(self, text: str, source_type: str) -> List[Dict]:
        """Generate questions from boxscore or summary."""
        if source_type == "boxscore":
            prompt = self.qa_gen_boxscore.format(text=text, num_questions=self.num_questions_src)
        else:
            prompt = self.qa_gen_summary.format(text=text, num_questions=self.num_questions_trg)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self._call_llm(messages)
        return self._parse_questions(response)
    
    def answer_question(self, text: str, question: str) -> str:
        """Answer a question based on given text."""
        prompt = self.gen_answers.format(text=text, question=question)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self._call_llm(messages)
        return self._parse_answer(response)
    
    def evaluate_cef_sample(self, boxscore: str, summary: str) -> Dict:
        """
        Evaluate a single boxscore-summary pair using CEF.
        
        Returns:
            Dictionary with questions, answers, and computed scores
        """
        result = {
            "questions_from_src": [],
            "questions_from_trg": [],
            "coverage_answers": [],      # Q_boxscore -> A_summary
            "faithfulness_answers": [],   # Q_summary -> A_boxscore
            "consistency_answers": [],    # Q_summary -> A_summary
            "coverage_score": 0.0,
            "conformity_score": 0.0,
            "faithfulness_score": 0.0,
            "consistency_score": 0.0,
            "conciseness_score": 0.0
        }
        
        # Generate questions from boxscore
        questions_from_boxscore = self.generate_questions(boxscore, "boxscore")
        result["questions_from_src"] = questions_from_boxscore
        
        # Generate questions from summary
        questions_from_summary = self.generate_questions(summary, "summary")
        result["questions_from_trg"] = questions_from_summary
        
        # Coverage: Questions from boxscore, answered using summary
        # "How much of the boxscore info is in the summary?"
        for q in questions_from_boxscore:
            answer = self.answer_question(summary, q["question"])
            result["coverage_answers"].append(answer)
        
        # Faithfulness: Questions from summary, answered using boxscore
        # "Is the summary accurate to the boxscore?"
        for q in questions_from_summary:
            answer = self.answer_question(boxscore, q["question"])
            result["faithfulness_answers"].append(answer)
        
        # Consistency: Questions from summary, answered using summary
        # "Is the summary internally consistent?"
        for q in questions_from_summary:
            answer = self.answer_question(summary, q["question"])
            result["consistency_answers"].append(answer)
        
        # Calculate scores
        # Coverage: 100 - %(IDK) from coverage_answers
        if result["coverage_answers"]:
            idk_count = result["coverage_answers"].count("IDK")
            no_count = result["coverage_answers"].count("NO")
            result["coverage_score"] = 100 - (idk_count / len(result["coverage_answers"]) * 100)
            result["conformity_score"] = 100 - (no_count / len(result["coverage_answers"]) * 100)
        
        # Faithfulness: 100 - %(NO) from faithfulness_answers
        if result["faithfulness_answers"]:
            no_count = result["faithfulness_answers"].count("NO")
            result["faithfulness_score"] = 100 - (no_count / len(result["faithfulness_answers"]) * 100)
        
        # Consistency: 100 - %(IDK) from consistency_answers
        if result["consistency_answers"]:
            idk_count = result["consistency_answers"].count("IDK")
            result["consistency_score"] = 100 - (idk_count / len(result["consistency_answers"]) * 100)
        
        # Conciseness
        if len(boxscore) > 0:
            result["conciseness_score"] = (len(summary) / len(boxscore)) * 100
        
        return result


class BoxscoreTraditionalEvaluator:
    """
    Traditional evaluation metrics for boxscore-to-summary generation.
    Uses ROUGE and BERTScore for comparison with reference summaries.
    """
    
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Compute traditional metrics.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            Dictionary with ROUGE and BERTScore metrics
        """
        # ROUGE scores
        rouge_results = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        # BERTScore
        bert_results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        
        return {
            "rouge1": rouge_results["rouge1"] * 100,
            "rouge2": rouge_results["rouge2"] * 100,
            "rougeL": rouge_results["rougeL"] * 100,
            "bertscore_precision": np.mean(bert_results["precision"]) * 100,
            "bertscore_recall": np.mean(bert_results["recall"]) * 100,
            "bertscore_f1": np.mean(bert_results["f1"]) * 100
        }


def compute_bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    boot_means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n_bootstrap)]
    lower = (1 - ci) / 2 * 100
    upper = (1 + ci) / 2 * 100
    return np.percentile(boot_means, lower), np.percentile(boot_means, upper)


def generate_cef_report(dataset: Dataset, split: str = "test") -> Dict:
    """
    Generate a CEF evaluation report from a dataset with CEF scores.
    
    Args:
        dataset: Dataset with cef_details column
        split: Dataset split to use
        
    Returns:
        Dictionary with aggregated scores and confidence intervals
    """
    metrics = {
        "coverage": [],
        "conformity": [],
        "faithfulness": [],
        "consistency": [],
        "conciseness": []
    }
    
    for example in dataset[split]:
        cef = example.get("cef_details", {})
        if cef:
            metrics["coverage"].append(cef.get("coverage_score", 0))
            metrics["conformity"].append(cef.get("conformity_score", 0))
            metrics["faithfulness"].append(cef.get("faithfulness_score", 0))
            metrics["consistency"].append(cef.get("consistency_score", 0))
            metrics["conciseness"].append(cef.get("conciseness_score", 0))
    
    report = {}
    for metric, values in metrics.items():
        if values:
            mean = np.mean(values)
            ci_low, ci_high = compute_bootstrap_ci(values)
            report[metric] = {
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high
            }
    
    return report


def generate_traditional_report(predictions: List[str], references: List[str]) -> Dict:
    """
    Generate traditional evaluation report.
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        
    Returns:
        Dictionary with traditional metrics
    """
    evaluator = BoxscoreTraditionalEvaluator()
    return evaluator.evaluate(predictions, references)

