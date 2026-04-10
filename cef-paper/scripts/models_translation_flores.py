#!/usr/bin/env python3
"""
vLLM Translation Script for Flores Dataset (Hugging Face Format)
Translates source text using multiple models to French, Spanish, and German
"""

import os
import gc
import torch
import logging
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, DatasetDict, load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DATASET = "flores"
DATASET = "ntrex"

MAX_INPUT_CHARS = 2000

def truncate_text(text):
    return text if len(text) <= MAX_INPUT_CHARS else text[:MAX_INPUT_CHARS]

class TranslationPipeline:
    def __init__(self):
        self.models = {
            "qwen2.5-7b":"Qwen/Qwen2.5-7B-Instruct",
            # "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
            # "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct"
        }

        self.target_languages = {
            "en-fr": "French",
            "en-es": "Spanish", 
            "en-de": "German"
        }

        self.data_path = f"/data/cef/{DATASET}/{DATASET}_original"
        self.dataset_dirs = ["en-fr", "en-de", "en-es"]

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            stop=[]
        )

    def load_source_data(self) -> Dataset:
        logger.info("Loading source data from Hugging Face format...")
        first_dataset_path = os.path.join(self.data_path, self.dataset_dirs[0])

        try:
            dataset = load_from_disk(first_dataset_path)
            logger.info(f"Successfully loaded dataset from {first_dataset_path}")
            if 'src' not in dataset.features:
                possible_src_cols = ['source', 'en', 'english', 'text', 'input']
                src_col = next((col for col in possible_src_cols if col in dataset.features), None)
                if src_col is None:
                    raise ValueError("Could not find source text column.")
                dataset = dataset.rename_column(src_col, 'src')
            if 'id' not in dataset.features:
                dataset = dataset.add_column('id', list(range(len(dataset))))
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def create_translation_prompt(self, tokenizer, text: str, target_language: str) -> str:
        text = truncate_text(text)
        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate the following English text to {target_language}. Provide only the translation. Respond starting directly with the translated sentence, without thinking steps."},
            {"role": "user", "content": text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("Prompt being sent:\n", prompt)
        return prompt

    def translate_batch(self, llm: LLM, tokenizer, texts: List[str], target_language: str) -> List[str]:
        prompts = [self.create_translation_prompt(tokenizer, text, target_language) for text in texts]
        logger.info(f"Translating batch of {len(texts)} texts to {target_language}")
        outputs = llm.generate(prompts, self.sampling_params)

        translations = []
        for i, output in enumerate(outputs):
            print(f"\n--- Prompt {i} ---\n{prompts[i]}")
            print("Raw output:", output)
            if output.outputs and output.outputs[0].text:
                text = output.outputs[0].text.strip()
                print("Generated:", text)
                translations.append(text.split('\n')[0].strip())
            else:
                print("\u26a0\ufe0f Empty or malformed output:", output)
                translations.append("")
        return translations

    def process_model(self, model_key: str, source_dataset: Dataset, batch_size: int = 32):
        model_path = self.models[model_key]
        logger.info(f"Loading model: {model_path}")

        try:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.8,
                max_model_len=8192,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        source_texts = source_dataset['src']
        translated_datasets = {}

        base_path = os.path.join(self.data_path, f"{DATASET}_{model_key}_all_languages")
        os.makedirs(base_path, exist_ok=True)

        for lang_code, lang_name in self.target_languages.items():
            logger.info(f"Translating to {lang_name} using {model_key}")
            all_translations = []

            for i in tqdm(range(0, len(source_texts), batch_size), desc=f"Translating to {lang_name}"):
                batch_texts = source_texts[i:i + batch_size]
                try:
                    batch_translations = self.translate_batch(llm, tokenizer, batch_texts, lang_name)
                    all_translations.extend(batch_translations)
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size}: {str(e)}")
                    all_translations.extend([""] * len(batch_texts))

            output_data = {
                'id': source_dataset['id'],
                'src': source_dataset['src'],
                'trg': all_translations
            }
            output_dataset = Dataset.from_dict(output_data)
            translated_datasets[lang_code] = output_dataset

            # ✅ Save under: /data/cef/europarl/europarl_MODEL_all_languages/en-fr
            lang_path = os.path.join(base_path, lang_code)
            os.makedirs(lang_path, exist_ok=True)
            output_dataset.save_to_disk(lang_path)
            logger.info(f"Saved translation to: {lang_path}")

        # ✅ Save the full DatasetDict with keys "en-fr", "en-es", "en-de"
        dataset_dict = DatasetDict(translated_datasets)
        dataset_dict.save_to_disk(base_path)
        logger.info(f"Saved DatasetDict with splits to: {base_path}")

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, batch_size: int = 32):
        logger.info("Starting translation pipeline...")
        try:
            source_dataset = self.load_source_data()
        except Exception as e:
            logger.error(f"Failed to load source data: {str(e)}")
            return

        for model_key in self.models.keys():
            try:
                logger.info(f"Processing model: {model_key}")
                self.process_model(model_key, source_dataset, batch_size)
                logger.info(f"Completed processing for {model_key}")
            except Exception as e:
                logger.error(f"Error processing model {model_key}: {str(e)}")

        logger.info("Translation pipeline completed!")

def main():
    pipeline = TranslationPipeline()
    batch_size = 16
    try:
        pipeline.run(batch_size=batch_size)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()



    

