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

# %%
# Judge models for robustness study
judge_models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V3", "openai/gpt-oss-120b"]
judge_model_workers = [7, 0, 5, 2]  # worker-7, worker-0, worker-5, worker-2
judge_index = 3  # Change this - the model that generated the questions
judge_model = judge_models[judge_index]
judge_model_worker = "worker-" + str(judge_model_workers[judge_index])
judge_model_port = 8892
debug = False
prompt_path = "/home/yathagata/cef-translation/src/cef_framework/prompts/note_generation.yaml"
ds_path = f"/data/cef/aci/aci_questions_{judge_model.replace('/', '+++')}"

# %%
ds = load_from_disk(ds_path)
if debug:
    for split in ds.keys():
        ds[split] = ds[split].select(range(2))

# %%
class ACIAnswerGenerator:
    def __init__(self, model_name, worker, port, prompt_path):
        self.system_prompt = "You are a helpful medical assistant specialized in clinical documentation and note generation."
        self.headers = {"Content-Type": "application/json"}
        self.api_url = f"http://{worker}:{port}/v1/chat/completions"
        self.model_name = model_name
        with open(prompt_path) as f:
            prompts = yaml.safe_load(f)
        self.prompt = prompts["user_instruction_set"]["gen_answers_v0"]
        
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
    
    def generate_answers(self, example, question_list_key, text_key):
        """
        Generate answers for questions based on a given text.
        
        Args:
            example: Dataset example
            question_list_key: Key for question list ('question_list_src' or 'question_list_trg')
            text_key: Key for text to use for answering ('src' or 'trg')
        """
        questions = example[question_list_key]
        answers = []
        if questions is None:
            return answers
        
        for question in questions:
            rendered_prompt = self.prompt.format(text=example[text_key], question=question["question"])
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": rendered_prompt}
                ],
                "temperature": 0,
                "max_tokens": 20000,
                "top_p": 0.9,
            }
            response_data = self.call_llm_with_retry(payload)
            if response_data is None:
                answers.append(None)
            else:
                answers.append(response_data)
        return answers

# %%
# Generate answers for all combinations:
# 1. Questions from src (conversation), answered using src (should be YES)
# 2. Questions from src (conversation), answered using trg (clinical note)
# 3. Questions from trg (clinical note), answered using trg (should be YES)
# 4. Questions from trg (clinical note), answered using src (conversation)

for idx, model in enumerate(judge_models):
    if idx == judge_index:
        # Skip the model that generated questions (answers should be YES)
        continue
    cur_judge_model = model
    cur_judge_model_worker = "worker-" + str(judge_model_workers[idx])
    answer_generator = ACIAnswerGenerator(
        cur_judge_model, cur_judge_model_worker, judge_model_port, prompt_path
    )
    
    # Generate answers for all 4 combinations
    ds = ds.map(
        lambda x: {
            # Questions from src, answered using src
            f"answers_src_src_{cur_judge_model}": answer_generator.generate_answers(x, "question_list_src", "src"),
            # Questions from src, answered using trg
            f"answers_src_trg_{cur_judge_model}": answer_generator.generate_answers(x, "question_list_src", "trg"),
            # Questions from trg, answered using trg
            f"answers_trg_trg_{cur_judge_model}": answer_generator.generate_answers(x, "question_list_trg", "trg"),
            # Questions from trg, answered using src
            f"answers_trg_src_{cur_judge_model}": answer_generator.generate_answers(x, "question_list_trg", "src"),
        }, 
        num_proc=10
    )

# %%
output_path = f"/data/cef/aci/aci_answers_{judge_model.replace('/', '+++')}"
ds.save_to_disk(output_path)
print(f"Saved answers to: {output_path}")
# %%

