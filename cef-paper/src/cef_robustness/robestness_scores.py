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
translation_type = "original"
judge_models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V3", "google/gemma-3-27b-it", "openai/gpt-oss-120b"]

# %%
judge_index = 0
judge_model = judge_models[judge_index]

# %%
ds_path = f"/data/cef/robust/ntrex_answers_lr_{translation_type}_{judge_model.replace('/', '+++')}"
ds = load_from_disk(ds_path)

# %%
def calculate_disagreement_rate(ds, split):
    """
    Calculate the average disagreement rate for a single judge model across all language pairs.
    """
    # Get all answer columns (judge models)
    answer_columns = [col for col in ds[list(ds.keys())[0]].column_names if col.startswith('answers_')]
    ds_split = ds[split]
    disagreement_rates = []
    for example in ds_split:
        question_list = example['question_list']
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
            disagreement_rates.append(disagreements/len(answers))
    return np.mean(disagreement_rates)

# %%

results = []
for judge_model in judge_models:
    ds_path = f"/data/cef/robust/ntrex_answers_lr_{translation_type}_{judge_model.replace('/', '+++')}" 
    ds = load_from_disk(ds_path)
    model_row = {'model': judge_model}
    total_disagreement_rate = 0
    split_count = 0
    
    for split in ds:
        disagreement_rate = calculate_disagreement_rate(ds, split)
        model_row[split] = round(disagreement_rate * 100, 2)
        total_disagreement_rate += disagreement_rate
        split_count += 1
    
    avg_disagreement_rate = total_disagreement_rate / split_count
    model_row['average'] = round(avg_disagreement_rate * 100, 2)
    
    results.append(model_row)


# %%
print("Disagreement rate for each split for each model for translation type: ", translation_type)
df = pd.DataFrame(results)
# Create a separate dataframe for rankings
df_rankings1 = df.copy()
# Add rankings for each column (lower score = better rank)
for col in df.columns:
    if col != 'model':
        rankings = df[col].rank(method='min').astype(int)
        df_rankings1[col] = rankings
        df[col] = df[col].astype(str) + ' (' + rankings.astype(str) + ')'

display(df)

# %%
def create_dataset_with_unique_questions(dss, split):
    all_questions = []
    all_answers = []
    for idx, example in enumerate(dss["Qwen/Qwen3-235B-A22B-Instruct-2507"][split]):
        questions_with_answers = {}
        for model_name, ds in dss.items():
            question_list = ds[split][idx]['question_list']
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
                        answers[cur_model] = ds[split][idx][f"answers_{cur_model}"][idx2]
                questions_with_answers[sample_question] = answers
        for question, answers in questions_with_answers.items():
            all_questions.append(question)
            all_answers.append(answers)
    return all_questions, all_answers

def find_majority_answer(answers):
    all_answers = [x for x in answers.values()]
    count = Counter(all_answers)
    max_count = max(count.values())
    majority_answers = [answer for answer, count in count.items() if count == max_count]
    if_none_present = None in all_answers
    return majority_answers, max_count, if_none_present
# %%
results = []

all_questions = {}
all_answers = {}

for judge_model in judge_models:
    model_row = {'Model': judge_model}
    total_deviation_score = 0
    split_count = 0
    
    for split in tqdm(ds):
        answer_deviation_scores = []
        dss = {}
        for jm in judge_models:
            ds_path = f"/data/cef/robust/ntrex_answers_lr_{translation_type}_{jm.replace('/', '+++')}" 
            ds_loaded = load_from_disk(ds_path)
            dss[jm] = ds_loaded
        questions, answers = create_dataset_with_unique_questions(dss, split)
        all_questions[split] = questions
        all_answers[split] = answers
        for question, answer in zip(questions, answers):
            majority_answer, max_count, if_none_present = find_majority_answer(answer)
            if (max_count < 4):
                continue
            if judge_model in answer and answer[judge_model] in majority_answer:
                answer_deviation_scores.append(0)
            else:
                answer_deviation_scores.append(1)
        
        deviation_score = np.mean(answer_deviation_scores)
        model_row[split] = round(deviation_score * 100, 2)
        total_deviation_score += deviation_score
        split_count += 1
    
    avg_deviation_score = total_deviation_score / split_count
    model_row['average'] = round(avg_deviation_score * 100, 2)
    results.append(model_row)

# %%
print("Deviation score for each split for each model for translation type: ", translation_type)
df = pd.DataFrame(results)
# Add rankings to the dataframe
df_rankings2 = df.copy()
for col in df.columns:
    if col != 'Model':
        # Create rankings (lower score = better rank)
        rankings = df[col].rank(method='min').astype(int)
        df_rankings2[col] = rankings
        df[col] = df[col].astype(str) + ' (' + rankings.astype(str) + ')'

display(df)



# # %%
# for split in all_questions:
#     questions = all_questions[split]
#     answers = all_answers[split]
#     model_answers = defaultdict(list)
#     count_list = []
#     for answer_dict in answers:
#         for model, answer_value in answer_dict.items():
#             model_answers[model].append(answer_value)
#         majority_answer, count, if_none_present = find_majority_answer(answer_dict)
#         count_list.append(str(count) + (" with None: " if if_none_present else ""))
#     print("For split: ", split)
#     proportion_data = []
#     for model, answers in model_answers.items():
#         counter = Counter(answers)
#         total = len(answers)
#         proportions = {answer: count/total*100 for answer, count in counter.items()}
#         row = {'Model': model}
#         row.update(proportions)
#         proportion_data.append(row)
    
#     proportion_df = pd.DataFrame(proportion_data).fillna(0).round(2)
#     display(proportion_df)
#     print("Counter of majority answers: ", Counter(count_list))
# %%
# Combine rankings from both dataframes to compute average rank and deviation
combined_rankings = []

for i in range(len(df_rankings2)):
    model_name = df_rankings2.iloc[i]['Model']
    combined_row = {'Model': model_name}
    
    for col in df_rankings2.columns:
        if col != 'Model':
            rank1 = df_rankings1.iloc[i][col]
            rank2 = df_rankings2.iloc[i][col]
            avg_rank = (rank1 + rank2) / 2
            deviation = abs(rank1 - rank2) / 2
            combined_row[f'{col}'] = str(round(avg_rank, 1)) + ' (' + str(round(deviation, 1)) + ')'
    
    combined_rankings.append(combined_row)

combined_df = pd.DataFrame(combined_rankings)

print("\nCombined Rankings - Average Rank and Deviation:")
print("avg = average of both rankings ( dev = absolute difference between rankings / 2)")
display(combined_df)

# %%
