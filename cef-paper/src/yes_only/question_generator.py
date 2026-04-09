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

# %%
ds_path = "/data/cef/ntrex"
translation_type = "original"
judge_model = "deepseek-ai/DeepSeek-V3"
judge_model_worker = "worker-3"
judge_model_port = 8892
prompt_path = "/home/yathagata/cef-translation/src/cef_framework/prompts/translation2.yaml"
prompt_name = "qa_gen_document_v0"
num_questions = 10
debug = False

# %%
ds = load_from_disk(os.path.join(ds_path, f"ntrex_{translation_type}"))
if debug:
    for split in ds.keys():
        ds[split] = ds[split].select(range(2))

# %%
class TranslationQualityEvaluator:
    def __init__(self, model_name, worker, port, prompt_path, prompt_name, num_questions):
        self.system_prompt = "You are a helpful multilingual assistant."
        self.headers = {"Content-Type": "application/json"}
        self.api_url = f"http://{worker}:{port}/v1/chat/completions"
        self.model_name = model_name
        with open(prompt_path) as f:
            prompts = yaml.safe_load(f)
        self.prompt = prompts["user_instruction_set"][prompt_name]
        self.num_questions = num_questions
        
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
            if (json_data is None or not isinstance(json_data, list)):
                continue
            # Validate that it's a list of dictionaries with the expected keys
            valid_questions = []
            for item in json_data:
                if (isinstance(item, dict) and "question" in item and "answer" in item):
                    valid_questions.append(item)
            if valid_questions:
                return valid_questions
        raise ValueError(f"No valid JSON question list found in response: {response_data}")
    
    def generate_questions(self, example, column_name):
        rendered_prompt = self.prompt.format(text=example[column_name], num_questions=self.num_questions, document_task="document")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": rendered_prompt}],
            "temperature": 0,
            "max_tokens": 20000,
            "top_p": 0.9,
        }
        response_data = self.call_llm_with_retry(payload)
        if response_data is None:
            return None
        return response_data

# %%
translator = TranslationQualityEvaluator(judge_model, judge_model_worker, judge_model_port, prompt_path, prompt_name, num_questions)
ds = ds.map(lambda x: {"question_list_trg": translator.generate_questions(x, "trg")}, num_proc=10)
ds["en-fr"] = ds["en-fr"].map(lambda x: {"question_list_src": translator.generate_questions(x, "src")}, num_proc=10)

#%%
prompt_name = "gen_answers_v0"

# %%
class AnswerGenerator:
    def __init__(self, model_name, worker, port, prompt_path, prompt_name):
        self.system_prompt = "You are a helpful multilingual assistant."
        self.headers = {"Content-Type": "application/json"}
        self.api_url = f"http://{worker}:{port}/v1/chat/completions"
        self.model_name = model_name
        with open(prompt_path) as f:
            prompts = yaml.safe_load(f)
        self.prompt = prompts["user_instruction_set"][prompt_name]
        
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
                    answer = self.parse_response(response_data)
                    return answer
                except Exception as e:
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
    
    def parse_response(self, response_data):
        response_data = response_data.strip()
        answer = ""
        answer_found = False
        if "yes" in response_data.lower():
            answer = "YES"
            answer_found = True
        elif "no" in response_data.lower():
            answer = "NO"
            if answer_found:
                raise ValueError(f"Multiple answers found in response: {response_data}")
            answer_found = True
        elif "idk" in response_data.lower():
            answer = "IDK"
            if answer_found:
                raise ValueError(f"Multiple answers found in response: {response_data}")
            answer_found = True
        else:
            raise ValueError(f"Invalid response format. Expected YES, NO, or IDK, got: {response_data}")
        return answer
    
    def generate_answers(self, example, column_name):
        questions = example["question_list_" + column_name]
        answers = []
        if questions is None:
            return answers
        for question in questions:
            rendered_prompt = self.prompt.format(text=example[column_name], question=question["question"])
            payload = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": rendered_prompt}],
                "temperature": 0,
                "max_tokens": 20000,
                "top_p": 0.9,
            }
            response_data = self.call_llm_with_retry(payload)
            if response_data is None:
                answers.append(None)
            else:
                answers.append(response_data)
        new_question_list = []
        for question, answer in zip(questions, answers):
            new_question_list.append({
                "question": question["question"],
                "previous_answer": question["answer"],
                "answer": answer
            })
        return {
            "question_list_" + column_name: new_question_list,
        }

# %%
answer_generator = AnswerGenerator(judge_model, judge_model_worker, judge_model_port, prompt_path, prompt_name)
ds = ds.map(lambda x: answer_generator.generate_answers(x, "trg"), num_proc=10)
ds["en-fr"] = ds["en-fr"].map(lambda x: answer_generator.generate_answers(x, "src"), num_proc=10)


# %%
prompt_path_last_name = prompt_path.split("/")[-1].split(".")[0]
ds.save_to_disk(os.path.join("/data/cef/yes_only", f"{prompt_path_last_name}"))


# %%
prompt = "translation"
ds = load_from_disk(os.path.join("/data/cef/yes_only", f"{prompt}"))
# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Extract answers and previous answers for confusion matrix
all_answers = []
all_previous_answers = []

ANSWER_COLUMN = "question_list_trg"

for split_name, split_data in ds.items():
    for example in split_data:
        if ANSWER_COLUMN in example and example[ANSWER_COLUMN] is not None:
            for qa_pair in example[ANSWER_COLUMN]:
                if qa_pair["answer"] is not None and qa_pair["previous_answer"] is not None:
                    all_answers.append(qa_pair["answer"])
                    all_previous_answers.append(qa_pair["previous_answer"])

# Define the labels
labels = ["YES", "NO", "IDK"]

# Create confusion matrix
cm = confusion_matrix(all_previous_answers, all_answers, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis_r', 
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 24}, square=False)
if prompt == "translation":
    plt.title('YES-ONLY prompt', fontsize=26, pad=8)
else:
    plt.title('Mixed-Answer prompt', fontsize=26, pad=8)
plt.xlabel('Re-Evaluated Answer', fontsize=22)
plt.ylabel('Generated Answer', fontsize=22)
plt.tick_params(axis='both', labelsize=20)

# Make colorbar labels bigger
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig(f"../../scripts/plots/{prompt}.pdf", format='pdf', bbox_inches='tight')
plt.show()
# Print statistics
print(f"Total question pairs: {len(all_answers)}")
print(f"Confusion Matrix:\n{cm}")

# Calculate accuracy and other metrics
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(all_previous_answers, all_answers, labels=labels))
# %%
