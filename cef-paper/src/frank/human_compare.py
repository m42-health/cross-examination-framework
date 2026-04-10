# %%
"""
FRANK Human Annotation vs CEF Comparison Script

This script compares CEF mismatching questions with FRANK human annotations
to determine if CEF captures the same factuality errors identified by humans.

Two-way comparison:
1. Human Error → CEF: Does any CEF mismatch correspond to this human error? (Recall)
2. CEF Mismatch → Human: Does any human error correspond to this CEF mismatch? (Precision)
"""

import requests
import json
import time
from jinja2 import Template
from jsonfinder import jsonfinder
from datasets import load_from_disk
import os

# %%
# Configuration
cef_results_path = "/data/cef/frank/frank_original_with_cef"
output_path = "/data/cef/frank/human_compare"
prompt_template_path = "/home/yathagata/cef-translation/src/frank/prompts/frank_human_compare.jinja"
cef_prompt_template_path = "/home/yathagata/cef-translation/src/frank/prompts/cef_frank_compare.jinja"

judge_model = "deepseek-ai/DeepSeek-V3"
judge_model_worker = "worker-12"
judge_model_port = 8892
debug = False

# %%
# Load CEF dataset
cef_dataset = load_from_disk(cef_results_path)
if hasattr(cef_dataset, 'keys'):
    cef_ds = cef_dataset
else:
    cef_ds = cef_dataset

if debug:
    if hasattr(cef_ds, 'keys'):
        for key in cef_ds.keys():
            cef_ds[key] = cef_ds[key].select(range(min(10, len(cef_ds[key]))))
    else:
        cef_ds = cef_ds.select(range(min(10, len(cef_ds))))

# Filter to only process samples where sentence_annotation after json loads is empty
print(f"Filtering samples with empty sentence_annotation...")
def is_empty_sentence_annotation(x):
    sa = x.get('sentence_annotation', '{}')
    # Parse if string
    if isinstance(sa, str):
        try:
            sa = json.loads(sa)
        except Exception:
            return False  # If invalid JSON, treat as not empty so we don't filter it in
    return bool(sa)

if hasattr(cef_ds, 'keys'):
    for key in cef_ds.keys():
        original_len = len(cef_ds[key])
        cef_ds[key] = cef_ds[key].filter(is_empty_sentence_annotation)
        filtered_len = len(cef_ds[key])
        print(f"  {key}: {original_len} -> {filtered_len} samples")
else:
    original_len = len(cef_ds)
    cef_ds = cef_ds.filter(is_empty_sentence_annotation)
    filtered_len = len(cef_ds)
    print(f"  {original_len} -> {filtered_len} samples")

# %%
# Load prompt templates
with open(prompt_template_path, 'r', encoding='utf-8') as f:
    prompt_template_str = f.read()

with open(cef_prompt_template_path, 'r', encoding='utf-8') as f:
    cef_prompt_template_str = f.read()

# %%
def extract_human_errors(example):
    """
    Extract human-annotated errors from FRANK sentence_annotation.
    
    Returns list of dicts with 'sentence' and 'error_types' keys.
    """
    sentence_annotation = example.get('sentence_annotation', '{}')
    
    # Parse JSON string if needed
    if isinstance(sentence_annotation, str):
        try:
            sentence_annotation = json.loads(sentence_annotation)
        except json.JSONDecodeError:
            return []
    
    if not sentence_annotation:
        return []
    
    human_errors = []
    for sentence, error_types in sentence_annotation.items():
        # Skip if no errors or only "NoE"
        if not error_types or error_types == ["NoE"]:
            continue
        # Filter out "NoE" from error types
        filtered_errors = [e for e in error_types if e != "NoE"]
        if filtered_errors:
            human_errors.append({
                'sentence': sentence,
                'error_types': filtered_errors
            })
    
    return human_errors


def extract_cef_mismatches(cef_details):
    """
    Extract CEF mismatching questions from both coverage and conformity.
    
    Coverage mismatches: questions_from_src where predicted != original
    Conformity mismatches: questions_from_trg where predicted != answer
    """
    mismatching_questions = []
    
    if not cef_details:
        return mismatching_questions
    
    # Coverage mismatches: Questions from article, answered using summary
    # Mismatch indicates summary doesn't cover this info from article
    questions_from_src = cef_details.get('questions_from_src', [])
    coverage_answers = cef_details.get('coverage_answers', [])
    
    for i, q_item in enumerate(questions_from_src):
        if isinstance(q_item, dict) and 'question' in q_item:
            original = q_item.get('original', q_item.get('answer', 'YES'))
            predicted = q_item.get('predicted', '')
            
            # Also check coverage_answers if predicted not in q_item
            if not predicted and i < len(coverage_answers):
                predicted = coverage_answers[i]
            
            if original in ['YES', 'NO'] and predicted in ['YES', 'NO', 'IDK']:
                if predicted != original:
                    mismatching_questions.append({
                        'question': q_item.get('question', ''),
                        'original': original,
                        'predicted': predicted,
                        'type': 'coverage',
                        'description': f"Article says {original}, but summary says {predicted}"
                    })
    
    # Conformity mismatches: Questions from summary, answered using article
    # Mismatch indicates summary claims something not in article (potential error)
    questions_from_trg = cef_details.get('questions_from_trg', [])
    conformity_answers = cef_details.get('conformity_answers', [])
    
    for i, q_item in enumerate(questions_from_trg):
        if isinstance(q_item, dict) and 'question' in q_item:
            answer = q_item.get('answer', 'YES')  # What summary claims
            predicted = q_item.get('predicted', '')  # What article says
            
            # Also check conformity_answers if predicted not in q_item
            if not predicted and i < len(conformity_answers):
                predicted = conformity_answers[i]
            
            if answer in ['YES', 'NO'] and predicted in ['YES', 'NO', 'IDK']:
                if predicted != answer:
                    mismatching_questions.append({
                        'question': q_item.get('question', ''),
                        'original': answer,  # What summary claims
                        'predicted': predicted,  # What article says
                        'type': 'conformity',
                        'description': f"Summary claims {answer}, but article says {predicted}"
                    })
    
    return mismatching_questions


# %%
class FrankCompareEvaluator:
    def __init__(self, model_name, worker, port, prompt_template_str, cef_prompt_template_str):
        self.system_prompt = "You are a helpful assistant specialized in summarization evaluation."
        self.headers = {"Content-Type": "application/json"}
        self.api_url = f"http://{worker}:{port}/v1/chat/completions"
        self.model_name = model_name
        self.prompt_template = Template(prompt_template_str)
        self.cef_prompt_template = Template(cef_prompt_template_str)
        
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
                        print("Max retries reached for rate limit")
                        return None
                
                try:
                    response_data = response.json()["choices"][0]["message"]["content"]
                    parsed_response = self.parse_json_from_response(response_data)
                    return parsed_response
                except ValueError as e:
                    print(f"Failed to parse JSON response (attempt {attempt + 1}/{max_retries + 1}): {e}")
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
            if "explanation" in json_data and "verdict" in json_data:
                if json_data["verdict"] in ["YES", "NO"]:
                    return json_data
        raise ValueError(f"No valid JSON found in response: {response_data}")
    
    def compare_human_error(self, article, summary, cef_mismatches, human_error):
        """
        Check if a human-annotated error corresponds to any CEF mismatch.
        (Human Error → CEF direction, for measuring recall)
        """
        # Format CEF mismatching questions
        if not cef_mismatches:
            cef_questions_str = "None (CEF found no mismatches)"
        else:
            formatted_questions = []
            for q_item in cef_mismatches:
                q_type = q_item.get('type', 'unknown')
                question = q_item.get('question', '')
                original = q_item.get('original', '')
                predicted = q_item.get('predicted', '')
                description = q_item.get('description', '')
                
                if q_type == 'coverage':
                    formatted_questions.append(
                        f"- [COVERAGE] Question: {question}\n"
                        f"  Article answer: {original}, Summary answer: {predicted}\n"
                        f"  Issue: {description}"
                    )
                elif q_type == 'conformity':
                    formatted_questions.append(
                        f"- [CONFORMITY] Question: {question}\n"
                        f"  Summary claims: {original}, Article says: {predicted}\n"
                        f"  Issue: {description}"
                    )
                else:
                    formatted_questions.append(f"- Question: {question}")
            
            cef_questions_str = "\n".join(formatted_questions)
        
        # Render prompt
        rendered_prompt = self.prompt_template.render(
            article=article,
            summary=summary,
            cef_mismatching_questions=cef_questions_str,
            error_sentence=human_error['sentence'],
            error_types=human_error['error_types']
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": rendered_prompt}
            ],
            "temperature": 0,
            "max_tokens": 2000,
            "top_p": 0.9,
        }
        
        return self.call_llm_with_retry(payload)
    
    def compare_cef_mismatch(self, article, summary, cef_mismatch, human_errors):
        """
        Check if a CEF mismatch corresponds to any human-annotated error.
        (CEF Mismatch → Human direction, for measuring precision)
        """
        # Format CEF mismatching question
        q_type = cef_mismatch.get('type', 'unknown')
        question = cef_mismatch.get('question', '')
        original = cef_mismatch.get('original', '')
        predicted = cef_mismatch.get('predicted', '')
        description = cef_mismatch.get('description', '')
        
        if q_type == 'coverage':
            cef_question_str = (
                f"[COVERAGE] Question: {question}\n"
                f"Article answer: {original}, Summary answer: {predicted}\n"
                f"Issue: {description}"
            )
        elif q_type == 'conformity':
            cef_question_str = (
                f"[CONFORMITY] Question: {question}\n"
                f"Summary claims: {original}, Article says: {predicted}\n"
                f"Issue: {description}"
            )
        else:
            cef_question_str = f"Question: {question}"
        
        # Format human errors
        if not human_errors:
            human_errors_str = "None (Human annotators found no errors)"
        else:
            formatted_errors = []
            for err in human_errors:
                sentence = err.get('sentence', '')
                error_types = err.get('error_types', [])
                formatted_errors.append(
                    f"- Sentence: \"{sentence}\"\n  Error Types: {error_types}"
                )
            human_errors_str = "\n".join(formatted_errors)
        
        # Render prompt
        rendered_prompt = self.cef_prompt_template.render(
            article=article,
            summary=summary,
            cef_mismatching_question=cef_question_str,
            human_errors=human_errors_str
        )
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": rendered_prompt}
            ],
            "temperature": 0,
            "max_tokens": 2000,
            "top_p": 0.9,
        }
        
        return self.call_llm_with_retry(payload)


# %%
def process_sample(example, idx):
    """Process a single sample: extract mismatches and errors, then compare"""
    # Create evaluator (needed for multiprocessing compatibility)
    evaluator = FrankCompareEvaluator(
        judge_model, judge_model_worker, judge_model_port, 
        prompt_template_str, cef_prompt_template_str
    )
    
    # Get article and summary
    article = example.get('src', '')
    summary = example.get('trg', '')
    
    # Extract CEF mismatches
    cef_details = example.get('cef_details', {})
    cef_mismatches = extract_cef_mismatches(cef_details)
    
    # Extract human errors
    human_errors = extract_human_errors(example)
    
    # Direction 1: Human Error → CEF (Recall)
    # For each human error, check if any CEF mismatch captures it
    human_to_cef_results = []
    if human_errors:
        for human_error in human_errors:
            result = evaluator.compare_human_error(
                article=article,
                summary=summary,
                cef_mismatches=cef_mismatches,
                human_error=human_error
            )
            human_to_cef_results.append({
                'human_error': human_error,
                'comparison_result': result
            })
    
    # Direction 2: CEF Mismatch → Human (Precision)
    # For each CEF mismatch, check if any human error corresponds to it
    cef_to_human_results = []
    if cef_mismatches:
        for cef_mismatch in cef_mismatches:
            result = evaluator.compare_cef_mismatch(
                article=article,
                summary=summary,
                cef_mismatch=cef_mismatch,
                human_errors=human_errors
            )
            cef_to_human_results.append({
                'cef_mismatch': cef_mismatch,
                'comparison_result': result
            })
    
    return {
        "human_errors": human_errors if human_errors else None,
        "cef_mismatches": cef_mismatches if cef_mismatches else None,
        "human_to_cef_results": human_to_cef_results if human_to_cef_results else None,
        "cef_to_human_results": cef_to_human_results if cef_to_human_results else None
    }


# %%
# Process all samples
print("Processing samples...")
if hasattr(cef_ds, 'keys'):
    for key in cef_ds.keys():
        print(f"Processing split: {key} ({len(cef_ds[key])} samples)")
        cef_ds[key] = cef_ds[key].map(
            process_sample,
            with_indices=True,
            num_proc=10,
            load_from_cache_file=False,
            desc=f"Comparing {key}"
        )
else:
    cef_ds = cef_ds.map(
        process_sample,
        with_indices=True,
        num_proc=10,
        load_from_cache_file=False,
        desc="Comparing samples"
    )

# %%
# Save the dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cef_ds.save_to_disk(output_path)
print(f"Saved dataset with comparison results to: {output_path}")

# %%
cef_ds = load_from_disk(output_path)


# %%
# Compute and print summary statistics
def compute_statistics(ds):
    """Compute recall and precision statistics from comparison results."""
    stats = {}
    
    for split in (ds.keys() if hasattr(ds, 'keys') else ['all']):
        split_ds = ds[split] if hasattr(ds, 'keys') else ds
        
        # Recall: Human errors caught by CEF
        total_human_errors = 0
        caught_human_errors = 0
        
        # Precision: CEF mismatches that correspond to real errors
        total_cef_mismatches = 0
        valid_cef_mismatches = 0
        
        for sample in split_ds:
            # Recall calculation
            human_to_cef = sample.get('human_to_cef_results', [])
            if human_to_cef:
                for result in human_to_cef:
                    total_human_errors += 1
                    comparison = result.get('comparison_result', {})
                    if comparison and comparison.get('verdict') == 'YES':
                        caught_human_errors += 1
            
            # Precision calculation
            cef_to_human = sample.get('cef_to_human_results', [])
            if cef_to_human:
                for result in cef_to_human:
                    total_cef_mismatches += 1
                    comparison = result.get('comparison_result', {})
                    if comparison and comparison.get('verdict') == 'YES':
                        valid_cef_mismatches += 1
        
        recall = caught_human_errors / total_human_errors if total_human_errors > 0 else 0
        precision = valid_cef_mismatches / total_cef_mismatches if total_cef_mismatches > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        stats[split] = {
            'total_human_errors': total_human_errors,
            'caught_human_errors': caught_human_errors,
            'recall': round(recall * 100, 2),
            'total_cef_mismatches': total_cef_mismatches,
            'valid_cef_mismatches': valid_cef_mismatches,
            'precision': round(precision * 100, 2),
            'f1': round(f1 * 100, 2)
        }
    
    return stats

stats = compute_statistics(cef_ds)
print("\n" + "=" * 60)
print("FRANK Human vs CEF Comparison Statistics")
print("=" * 60)
for split, split_stats in stats.items():
    print(f"\nSplit: {split}")
    print(f"  Recall (Human errors caught by CEF):")
    print(f"    {split_stats['caught_human_errors']}/{split_stats['total_human_errors']} = {split_stats['recall']}%")
    print(f"  Precision (CEF mismatches that are real errors):")
    print(f"    {split_stats['valid_cef_mismatches']}/{split_stats['total_cef_mismatches']} = {split_stats['precision']}%")
    print(f"  F1 Score: {split_stats['f1']}%")

# Save statistics
stats_path = os.path.join(output_path, "statistics.json")
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\nStatistics saved to: {stats_path}")

# %%
# Compute error type breakdown for human_to_cef_results
from collections import defaultdict

def compute_error_type_stats(ds):
    """Compute catch rate by human error type."""
    error_type_stats = defaultdict(lambda: {'total': 0, 'caught': 0})
    
    for split in (ds.keys() if hasattr(ds, 'keys') else ['all']):
        split_ds = ds[split] if hasattr(ds, 'keys') else ds
        
        for sample in split_ds:
            human_to_cef = sample.get('human_to_cef_results', [])
            if human_to_cef:
                for result in human_to_cef:
                    human_error = result.get('human_error', {})
                    error_types = human_error.get('error_types', [])
                    comparison = result.get('comparison_result', {})
                    caught = comparison and comparison.get('verdict') == 'YES'
                    
                    for error_type in error_types:
                        error_type_stats[error_type]['total'] += 1
                        if caught:
                            error_type_stats[error_type]['caught'] += 1
    
    return dict(error_type_stats)

error_stats = compute_error_type_stats(cef_ds)

# Print table
print("\n" + "=" * 60)
print("Human Error Types Caught by CEF (human_to_cef_results)")
print("=" * 60)
print(f"{'Error Type':<20} | {'Caught':>7} | {'Total':>7} | {'Percentage':>10}")
print("-" * 60)

for error_type, stats in sorted(error_stats.items(), key=lambda x: x[1]['total'], reverse=True):
    pct = (stats['caught'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"{error_type:<20} | {stats['caught']:>7} | {stats['total']:>7} | {pct:>9.1f}%")

# Overall row
print("-" * 60)
total_caught = sum(s['caught'] for s in error_stats.values())
total_total = sum(s['total'] for s in error_stats.values())
overall_pct = (total_caught / total_total * 100) if total_total > 0 else 0
print(f"{'OVERALL':<20} | {total_caught:>7} | {total_total:>7} | {overall_pct:>9.1f}%")

# %%

