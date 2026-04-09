# %%
import requests
from transformers import AutoTokenizer
import numpy as np
from scipy import stats
import sentencepiece as spm
from sacrebleu.tokenizers import BaseTokenizer
import evaluate
from jinja2 import Template
import time
from ast import literal_eval
from jsonfinder import jsonfinder
from datasets import load_from_disk
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import json


# %%
# Judge models for robustness study
# - worker-8:8892 → meta-llama/Llama-3.3-70B-Instruct
# - worker-4:8892 → Qwen/Qwen3-235B-A22B-Instruct-2507
# - worker-5:8892 → deepseek-ai/DeepSeek-V3
# - worker-6:8892 → openai/gpt-oss-120b
judge_models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V3", "openai/gpt-oss-120b"]

# Answer type combinations:
# - src_src: Questions from boxscore, answered using boxscore
# - src_trg: Questions from boxscore, answered using summary (COVERAGE)
# - trg_trg: Questions from summary, answered using summary (CONSISTENCY)
# - trg_src: Questions from summary, answered using boxscore (FAITHFULNESS)
answer_types = ["src_src", "src_trg", "trg_trg", "trg_src"]

debug = False

# %%
# Try to load all datasets first
print("Loading datasets for all judge models...")
all_datasets = {}
for judge_model in judge_models:
    ds_path = f"/data/cef/boxscore/boxscore_answers_{judge_model.replace('/', '+++')}"
    try:
        ds = load_from_disk(ds_path)
        all_datasets[judge_model] = ds
        print(f"  ✓ Loaded: {judge_model.split('/')[-1]}")
    except Exception as e:
        print(f"  ✗ Could not load dataset for {judge_model}: {e}")

if debug and all_datasets:
    first_ds = list(all_datasets.values())[0]
    for split in first_ds.keys():
        for model in all_datasets:
            all_datasets[model][split] = all_datasets[model][split].select(range(min(5, len(all_datasets[model][split]))))


# %%
def calculate_disagreement_rate(ds, split, answer_type):
    """
    Calculate the average disagreement rate for a single judge model across the dataset.
    
    Args:
        ds: Dataset
        split: Dataset split name
        answer_type: One of 'src_src', 'src_trg', 'trg_trg', 'trg_src'
    """
    # Determine which question list to use
    if answer_type.startswith("src"):
        question_list_key = "question_list_src"
    else:
        question_list_key = "question_list_trg"
    
    # Get all answer columns for this answer type
    answer_columns = [col for col in ds[list(ds.keys())[0]].column_names 
                      if col.startswith(f'answers_{answer_type}_')]
    
    ds_split = ds[split]
    disagreement_rates = []
    
    for example in ds_split:
        question_list = example[question_list_key]
        if question_list is None:
            continue
        for idx, question in enumerate(question_list):
            answers = []
            for col in answer_columns:
                if example[col] is not None and len(example[col]) > idx:
                    answers.append(example[col][idx])
            if len(answers) < 2:
                continue
            disagreements = 0
            for i in range(len(answers)):
                if answers[i] != "YES":
                    disagreements += 1
            disagreement_rates.append(disagreements / len(answers))
    
    return np.mean(disagreement_rates) if disagreement_rates else 0.0


# %%
# Calculate disagreement rates for each answer type
print("\n" + "=" * 80)
print("DISAGREEMENT RATES FOR BOXSCORE-TO-SUMMARY GENERATION")
print("=" * 80)

for answer_type in answer_types:
    print(f"\n--- Answer Type: {answer_type} ---")
    if answer_type == "src_src":
        print("(Questions from boxscore, answered using boxscore)")
    elif answer_type == "src_trg":
        print("(Questions from boxscore, answered using summary) - COVERAGE")
    elif answer_type == "trg_trg":
        print("(Questions from summary, answered using summary) - CONSISTENCY")
    elif answer_type == "trg_src":
        print("(Questions from summary, answered using boxscore) - FAITHFULNESS")
    
    results = []
    for judge_model in judge_models:
        if judge_model not in all_datasets:
            continue
        ds = all_datasets[judge_model]
            
        model_row = {'model': judge_model.split('/')[-1]}
        total_disagreement_rate = 0
        split_count = 0
        
        for split in ds:
            disagreement_rate = calculate_disagreement_rate(ds, split, answer_type)
            model_row[split] = round(disagreement_rate * 100, 2)
            total_disagreement_rate += disagreement_rate
            split_count += 1
        
        if split_count > 0:
            avg_disagreement_rate = total_disagreement_rate / split_count
            model_row['average'] = round(avg_disagreement_rate * 100, 2)
        
        results.append(model_row)
    
    if results:
        df = pd.DataFrame(results)
        df_rankings = df.copy()
        for col in df.columns:
            if col != 'model':
                rankings = df[col].rank(method='min').astype(int)
                df_rankings[col] = rankings
                df[col] = df[col].astype(str) + ' (' + rankings.astype(str) + ')'
        try:
            display(df)
        except:
            print(df.to_string(index=False))


# %%
def create_dataset_with_unique_questions(dss, split, answer_type):
    """
    Create a dataset with unique questions and their answers from all models.
    """
    if answer_type.startswith("src"):
        question_list_key = "question_list_src"
    else:
        question_list_key = "question_list_trg"
    
    all_questions = []
    all_answers = []
    
    first_model = list(dss.keys())[0]
    for idx, example in enumerate(dss[first_model][split]):
        questions_with_answers = {}
        for model_name, ds in dss.items():
            question_list = ds[split][idx][question_list_key]
            if question_list is None:
                continue
            sample_questions = [x["question"] for x in question_list]
            for idx2, sample_question in enumerate(sample_questions):
                if sample_question in questions_with_answers:
                    continue
                answers = {}
                for cur_model in judge_models:
                    if cur_model == model_name:
                        answers[cur_model] = "YES"
                    else:
                        answer_col = f"answers_{answer_type}_{cur_model}"
                        if answer_col in ds[split][idx] and ds[split][idx][answer_col] is not None:
                            if len(ds[split][idx][answer_col]) > idx2:
                                answers[cur_model] = ds[split][idx][answer_col][idx2]
                questions_with_answers[sample_question] = answers
        
        for question, answers in questions_with_answers.items():
            all_questions.append(question)
            all_answers.append(answers)
    
    return all_questions, all_answers


def find_majority_answer(answers):
    all_answers = [x for x in answers.values()]
    count = Counter(all_answers)
    max_count = max(count.values())
    majority_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    if_none_present = None in all_answers
    return majority_answers, max_count, if_none_present


# %%
# Calculate deviation scores for each answer type
print("\n" + "=" * 80)
print("DEVIATION SCORES FOR BOXSCORE-TO-SUMMARY GENERATION")
print("=" * 80)

for answer_type in answer_types:
    print(f"\n--- Answer Type: {answer_type} ---")
    if answer_type == "src_src":
        print("(Questions from boxscore, answered using boxscore)")
    elif answer_type == "src_trg":
        print("(Questions from boxscore, answered using summary) - COVERAGE")
    elif answer_type == "trg_trg":
        print("(Questions from summary, answered using summary) - CONSISTENCY")
    elif answer_type == "trg_src":
        print("(Questions from summary, answered using boxscore) - FAITHFULNESS")
    
    results = []
    
    for judge_model in judge_models:
        model_row = {'Model': judge_model.split('/')[-1]}
        total_deviation_score = 0
        split_count = 0
        
        if not all_datasets:
            continue
        
        first_ds = list(all_datasets.values())[0]
        for split in tqdm(first_ds, desc=f"Processing {judge_model.split('/')[-1]}"):
            answer_deviation_scores = []
            questions, answers = create_dataset_with_unique_questions(all_datasets, split, answer_type)
            
            for question, answer in zip(questions, answers):
                majority_answer, max_count, if_none_present = find_majority_answer(answer)
                if max_count < 3:  # Require at least 3 judges to agree
                    continue
                if judge_model in answer and answer[judge_model] in majority_answer:
                    answer_deviation_scores.append(0)
                else:
                    answer_deviation_scores.append(1)
            
            if answer_deviation_scores:
                deviation_score = np.mean(answer_deviation_scores)
                model_row[split] = round(deviation_score * 100, 2)
                total_deviation_score += deviation_score
                split_count += 1
        
        if split_count > 0:
            avg_deviation_score = total_deviation_score / split_count
            model_row['average'] = round(avg_deviation_score * 100, 2)
        
        results.append(model_row)
    
    if results:
        df = pd.DataFrame(results)
        df_rankings = df.copy()
        for col in df.columns:
            if col != 'Model':
                rankings = df[col].rank(method='min').astype(int)
                df_rankings[col] = rankings
                df[col] = df[col].astype(str) + ' (' + rankings.astype(str) + ')'
        try:
            display(df)
        except:
            print(df.to_string(index=False))


# %%
# Summary comparison across answer types
print("\n" + "=" * 80)
print("SUMMARY: Average Disagreement Rates by Answer Type")
print("=" * 80)

summary_data = []
for answer_type in answer_types:
    row = {'Answer Type': answer_type}
    
    for judge_model in judge_models:
        if judge_model not in all_datasets:
            continue
        ds = all_datasets[judge_model]
        
        total_rate = 0
        count = 0
        for split in ds:
            rate = calculate_disagreement_rate(ds, split, answer_type)
            total_rate += rate
            count += 1
        if count > 0:
            row[judge_model.split('/')[-1]] = round(total_rate / count * 100, 2)
    
    summary_data.append(row)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    try:
        display(summary_df)
    except:
        print(summary_df.to_string(index=False))


# %%
# Answer distribution analysis
print("\n" + "=" * 80)
print("ANSWER DISTRIBUTION ANALYSIS")
print("=" * 80)

for answer_type in answer_types:
    print(f"\n--- Answer Type: {answer_type} ---")
    
    all_answers_by_model = {model: [] for model in judge_models}
    
    for judge_model in judge_models:
        if judge_model not in all_datasets:
            continue
        ds = all_datasets[judge_model]
        
        for split in ds:
            for example in ds[split]:
                for other_model in judge_models:
                    if other_model == judge_model:
                        continue
                    col_name = f"answers_{answer_type}_{other_model}"
                    if col_name in example and example[col_name]:
                        all_answers_by_model[other_model].extend(example[col_name])
    
    dist_data = []
    for model, answers in all_answers_by_model.items():
        if not answers:
            continue
        answers = [a for a in answers if a is not None]
        if not answers:
            continue
        counter = Counter(answers)
        total = len(answers)
        row = {
            'Model': model.split('/')[-1],
            'YES (%)': round(counter.get('YES', 0) / total * 100, 2),
            'NO (%)': round(counter.get('NO', 0) / total * 100, 2),
            'IDK (%)': round(counter.get('IDK', 0) / total * 100, 2),
            'Total': total
        }
        dist_data.append(row)
    
    if dist_data:
        dist_df = pd.DataFrame(dist_data)
        try:
            display(dist_df)
        except:
            print(dist_df.to_string(index=False))


# %%
# Save consolidated results
print("\n" + "=" * 80)
print("SAVING CONSOLIDATED RESULTS")
print("=" * 80)

consolidated_results = {
    "judge_models": judge_models,
    "answer_types": answer_types,
    "summary_by_answer_type": summary_data,
}

output_path = "/data/cef/boxscore/robustness_scores_consolidated.json"
try:
    with open(output_path, 'w') as f:
        json.dump(consolidated_results, f, indent=2)
    print(f"Saved consolidated results to: {output_path}")
except Exception as e:
    print(f"Could not save results: {e}")


# %%
# Interpretation guide
print("\n" + "=" * 80)
print("INTERPRETATION GUIDE:")
print("=" * 80)
print("""
For the Boxscore-to-Summary task:

1. src_src (Q: Boxscore → A: Boxscore):
   - Questions from boxscore data, answered using boxscore
   - Expected: High YES rate (questions should be answerable)
   - Low deviation indicates good question quality

2. src_trg (Q: Boxscore → A: Summary) - COVERAGE:
   - Questions about boxscore facts, checked against generated summary
   - This measures COVERAGE (how much boxscore info is in summary)
   - High IDK = missing information from summary
   - High NO = contradictory information in summary

3. trg_trg (Q: Summary → A: Summary) - CONSISTENCY:
   - Questions from summary, answered using summary
   - Expected: High YES rate (internal consistency)
   - Low deviation indicates consistent summary

4. trg_src (Q: Summary → A: Boxscore) - FAITHFULNESS:
   - Questions about summary claims, verified against boxscore
   - This measures FAITHFULNESS (is summary accurate?)
   - High NO = hallucinated facts in summary
   - High IDK = summary mentions things not in boxscore

Disagreement Rate:
   - Measures how often judges disagree on an answer
   - Lower is better (more consistent judgment)

Deviation Score:
   - Measures how often a model deviates from the majority answer
   - Lower is better (more aligned with consensus)
""")

# %%
