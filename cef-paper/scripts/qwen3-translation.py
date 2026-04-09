#!/usr/bin/env python3
"""
Translation Script using OpenAI-compatible vLLM HTTP API (Qwen3-0.6B on worker-5)
"""

import os
import gc
import torch
import logging
import requests
from pathlib import Path
from typing import List, Dict
from datasets import Dataset, DatasetDict, load_from_disk
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
        self.api_url = "http://worker-5:8000/v1/chat/completions"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

        self.target_languages = {
            "en-fr": "French",
            "en-es": "Spanish", 
            "en-de": "German"
        }

        self.data_path = f"/data/cef/{DATASET}/{DATASET}_original"
        self.dataset_dirs = ["en-fr", "en-de", "en-es"]

        self.request_payload = {
            "model": "Qwen/Qwen3-0.6B",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024
        }

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

    def create_prompt(self, text: str, target_language: str) -> List[Dict[str, str]]:
        text = truncate_text(text)
        return [
            {"role": "system", "content": f"You are a professional translator. Translate the following English text to {target_language}. Provide only the translation. Respond starting directly with the translated sentence, without thinking steps."},
            {"role": "user", "content": text}
        ]

    def translate_batch(self, texts: List[str], target_language: str) -> List[str]:
        translations = []

        for i, text in enumerate(texts):
            messages = self.create_prompt(text, target_language)
            payload = {
                **self.request_payload,
                "messages": messages
            }

            try:
                response = requests.post(self.api_url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                translated_text = result["choices"][0]["message"]["content"].strip().split("\n")[0]
                translations.append(translated_text)
            except Exception as e:
                logger.error(f"Error in translation request {i}: {str(e)}")
                translations.append("")

        return translations

    def process_model(self, source_dataset: Dataset, batch_size: int = 16):
        logger.info(f"Using HTTP API at {self.api_url}")
        source_texts = source_dataset['src']
        translated_datasets = {}

        base_path = os.path.join(self.data_path, f"{DATASET}_qwen3-0.6b_all_languages")
        os.makedirs(base_path, exist_ok=True)

        for lang_code, lang_name in self.target_languages.items():
            logger.info(f"Translating to {lang_name} via API")
            all_translations = []

            for i in tqdm(range(0, len(source_texts), batch_size), desc=f"Translating to {lang_name}"):
                batch_texts = source_texts[i:i + batch_size]
                try:
                    batch_translations = self.translate_batch(batch_texts, lang_name)
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

            lang_path = os.path.join(base_path, lang_code)
            os.makedirs(lang_path, exist_ok=True)
            output_dataset.save_to_disk(lang_path)
            logger.info(f"Saved translation to: {lang_path}")

        dataset_dict = DatasetDict(translated_datasets)
        dataset_dict.save_to_disk(base_path)
        logger.info(f"Saved DatasetDict with splits to: {base_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, batch_size: int = 16):
        logger.info("Starting translation pipeline using HTTP API...")
        try:
            source_dataset = self.load_source_data()
        except Exception as e:
            logger.error(f"Failed to load source data: {str(e)}")
            return

        try:
            self.process_model(source_dataset, batch_size)
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")

        logger.info("Translation pipeline completed!")

def main():
    pipeline = TranslationPipeline()
    try:
        pipeline.run(batch_size=16)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
