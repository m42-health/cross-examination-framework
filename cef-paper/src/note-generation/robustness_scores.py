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


# %%
judge_models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V3", "openai/gpt-oss-120b"]

# Answer type combinations:
# - src_src: Questions from conversation, answered using conversation
# - src_trg: Questions from conversation, answered using clinical note
# - trg_trg: Questions from clinical note, answered using clinical note
# - trg_src: Questions from clinical note, answered using conversation
answer_types = ["src_src", "src_trg", "trg_trg", "trg_src"]

# %%
judge_index = 0
judge_model = judge_models[judge_index]

# %%
ds_path = f"/data/cef/aci/aci_answers_{judge_model.replace('/', '+++')}"
ds = load_from_disk(ds_path)

# %%
def calculate_disagreement_rate(ds, split, answer_type):
    """
    Calculate the average disagreement rate for a single judge model across the dataset.
    """
    if answer_type.startswith("src"):
        question_list_key = "question_list_src"
    else:
        question_list_key = "question_list_trg"
    
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
print("=" * 80)
print("DISAGREEMENT RATES FOR ACI CLINICAL NOTE GENERATION")
print("=" * 80)

for answer_type in answer_types:
    print(f"\n--- Answer Type: {answer_type} ---")
    if answer_type == "src_src":
        print("(Questions from conversation, answered using conversation)")
    elif answer_type == "src_trg":
        print("(Questions from conversation, answered using clinical note)")
    elif answer_type == "trg_trg":
        print("(Questions from clinical note, answered using clinical note)")
    elif answer_type == "trg_src":
        print("(Questions from clinical note, answered using conversation)")
    
    results = []
    for judge_model in judge_models:
        ds_path = f"/data/cef/aci/aci_answers_{judge_model.replace('/', '+++')}"
        try:
            ds = load_from_disk(ds_path)
        except:
            print(f"Could not load dataset for {judge_model}")
            continue
            
        model_row = {'model': judge_model}
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
        display(df)


# %%
def create_dataset_with_unique_questions(dss, split, answer_type):
    """Create a dataset with unique questions and their answers from all models."""
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
print("DEVIATION SCORES FOR ACI CLINICAL NOTE GENERATION")
print("=" * 80)

for answer_type in answer_types:
    print(f"\n--- Answer Type: {answer_type} ---")
    
    results = []
    all_questions = {}
    all_answers = {}
    
    for judge_model in judge_models:
        model_row = {'Model': judge_model}
        total_deviation_score = 0
        split_count = 0
        
        dss = {}
        for jm in judge_models:
            ds_path = f"/data/cef/aci/aci_answers_{jm.replace('/', '+++')}"
            try:
                ds_loaded = load_from_disk(ds_path)
                dss[jm] = ds_loaded
            except:
                continue
        
        if not dss:
            continue
        
        first_ds = list(dss.values())[0]
        for split in tqdm(first_ds, desc=f"Processing {judge_model}"):
            answer_deviation_scores = []
            questions, answers = create_dataset_with_unique_questions(dss, split, answer_type)
            all_questions[split] = questions
            all_answers[split] = answers
            
            for question, answer in zip(questions, answers):
                majority_answer, max_count, if_none_present = find_majority_answer(answer)
                if max_count < 3:  # Adjusted for 4 judges
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
        display(df)


# %%
# Summary comparison across answer types
print("\n" + "=" * 80)
print("SUMMARY: Average Disagreement Rates by Answer Type")
print("=" * 80)

summary_data = []
for answer_type in answer_types:
    row = {'Answer Type': answer_type}
    
    for judge_model in judge_models:
        ds_path = f"/data/cef/aci/aci_answers_{judge_model.replace('/', '+++')}"
        try:
            ds = load_from_disk(ds_path)
            total_rate = 0
            count = 0
            for split in ds:
                rate = calculate_disagreement_rate(ds, split, answer_type)
                total_rate += rate
                count += 1
            if count > 0:
                row[judge_model.split('/')[-1]] = round(total_rate / count * 100, 2)
        except:
            continue
    
    summary_data.append(row)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

65# %%

