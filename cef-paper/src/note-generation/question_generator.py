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
ds_path = "/data/cef/aci/aci_original"
# Judge models for robustness study
judge_models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V3", "openai/gpt-oss-120b"]
judge_model_workers = [7, 0, 5, 2]  # worker-7, worker-0, worker-5, worker-2
judge_index = 3  # Change this to use a different model
judge_model = judge_models[judge_index]
judge_model_worker = "worker-" + str(judge_model_workers[judge_index])
judge_model_port = 8892
debug = False
prompt_path = "/home/yathagata/cef-translation/src/cef_framework/prompts/note_generation.yaml"
num_questions_src = 10  # Questions from conversation
num_questions_trg = 10   # Questions from clinical note

# %%
ds = load_from_disk(ds_path)
if debug:
    for split in ds.keys():
        ds[split] = ds[split].select(range(5))

# %%
tokenizer = AutoTokenizer.from_pretrained(judge_model)


# %%
class ACIQAGenerator:
    def __init__(self, model_name, worker, port, prompt_path, num_questions_src, num_questions_trg):
        self.system_prompt = "You are a helpful medical assistant specialized in clinical documentation and note generation."
        self.headers = {"Content-Type": "application/json"}
        self.api_url = f"http://{worker}:{port}/v1/chat/completions"
        self.model_name = model_name
        with open(prompt_path) as f:
            prompts = yaml.safe_load(f)
        # Prompt for generating questions from conversation (src)
        self.prompt_conversation = prompts["user_instruction_set"]["qa_gen_conversation_v0"]
        # Prompt for generating questions from clinical note (trg)
        self.prompt_note = prompts["user_instruction_set"]["qa_gen_note_v0"]
        self.num_questions_src = num_questions_src
        self.num_questions_trg = num_questions_trg
        
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
    
    def generate_questions_from_src(self, example):
        """Generate questions from the source conversation (src)"""
        rendered_prompt = self.prompt_conversation.format(
            text=example["src"], 
            num_questions=self.num_questions_src
        )
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
            return None
        return response_data
    
    def generate_questions_from_trg(self, example):
        """Generate questions from the target clinical note (trg)"""
        rendered_prompt = self.prompt_note.format(
            text=example["trg"], 
            num_questions=self.num_questions_trg
        )
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
            return None
        return response_data

# %%
qa_generator = ACIQAGenerator(
    judge_model, judge_model_worker, judge_model_port, prompt_path, num_questions_src, num_questions_trg
)

# Generate questions from both src (conversation) and trg (clinical note)
ds = ds.map(
    lambda x: {
        "question_list_src": qa_generator.generate_questions_from_src(x),
        "question_list_trg": qa_generator.generate_questions_from_trg(x)
    }, 
    num_proc=10
)

# %%
output_path = f"/data/cef/aci/aci_questions_{judge_model.replace('/', '+++')}"
ds.save_to_disk(output_path)
print(f"Saved questions to: {output_path}")
# %%

