#!/usr/bin/env python3
"""
A script to translate multiple language pairs using parallel processing.

This script loads the ntrex dataset and translates each language pair,
replacing the target column with new translations and saving each language
pair as a separate dataset.
"""

import os
import gc
import torch
import logging
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Holds all configuration parameters for the translation pipeline."""
    # API and Model Settings
    API_URL = "http://worker-2:8892/v1/chat/completions"
    MODEL_NAME = "openai/gpt-oss-20b"
    SAVE_MODEL_NAME = "qwen3-4b"
    MAX_INPUT_CHARS = 8192
    REQUEST_TIMEOUT = 120  # seconds
    DEBUG = False
    ROUND_TRIP = True
    # Data Settings
    BASE_DATA_PATH = Path(f"/data/cef/ntrex/ntrex_lr_{SAVE_MODEL_NAME}")
    OUTPUT_BASE_PATH = Path("/data/cef/ntrex/ntrex_roundtrip_")
    
    # Processing Settings
    NUM_PROCESSES = 16  # Number of parallel processes for .map()

    @property 
    def output_path(self):
        """Generate output path with model name."""
        return str(self.OUTPUT_BASE_PATH) + str(self.SAVE_MODEL_NAME)


class ParallelTranslator:
    """
    A class designed to work with datasets.map() for parallel translation.
    It handles API requests and prompt creation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.round_trip = config.ROUND_TRIP
        self.api_payload_template = {
            "model": self.config.MODEL_NAME,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 8192
        }

    def _clean_text(self, text: str) -> str:
        """Remove special characters."""
        if not isinstance(text, str):
            logger.warning(f"Non-string input received: {type(text)}. Returning empty string.")
            return ""
        return text.replace("\n", " ")

    def _create_prompt(self, source_language: str, target_language: str, text: str) -> List[Dict[str, str]]:
        """Creates the prompt for the API call."""
        truncated_text = self._clean_text(text)
        return [
            {"role": "system", "content": f"You are a professional translator. Translate the following {source_language} text to {target_language}. Provide only the translation. Do not include explanations or apologies."},
            {"role": "user", "content": truncated_text}
        ]

    def _make_api_request(self, messages: List[Dict[str, str]], attempt: int = 1) -> str:
        """Makes a single, robust API request with retry logic."""
        payload = {**self.api_payload_template, "messages": messages}
        payload["temperature"] = 0.1 * attempt
        try:
            response = requests.post(self.config.API_URL, json=payload, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                return content
            else:
                logger.warning(f"PID {os.getpid()}: Unexpected API response format: {result}")
                raise ValueError(f"Unexpected API response format: {result}")
        except Exception as e:
            logger.error(f"PID {os.getpid()}: API request failed: {e}")
            if attempt < 6:
                time.sleep(2 * attempt)
                return self._make_api_request(messages, attempt + 1)
            else:
                logger.error(f"PID {os.getpid()}: Max retries reached. Returning error string.")
                return "ERROR: MAX_RETRIES"

    def translate_example(self, example: Dict[str, Any], source_lang_name: str, target_lang_name: str) -> Dict[str, Any]:
        """
        Translates a single example. This is the function passed to .map().
        """
        if self.round_trip:
            source_text = example['trg']
        else:
            source_text = example['src']
        prompt = self._create_prompt(source_lang_name, target_lang_name, source_text)
        translation = self._make_api_request(prompt)
        
        # Replace the 'trg' column with the translation
        example['trg'] = translation
        return example


def get_language_names():
    """Returns a mapping of language codes to language names."""
    return {
        'en': 'English',
        'fr': 'French', 
        'es': 'Spanish',
        'de': 'German',
        'ar': 'Arabic',
        'jp': 'Japanese',
        'tir': 'Tigrinya',
        'eus': 'Basque',
        'dzo': 'Dzongkha',
        'mri': 'Maori',
        'khm': 'Khmer'
    }


def run_translation_pipeline(config: Config):
    """Orchestrates the translation of all language pairs in the ntrex dataset."""
    
    # --- 1. Load Dataset ---
    logger.info(f"Loading dataset from: {config.BASE_DATA_PATH}")
    try:
        dataset_dict = load_from_disk(str(config.BASE_DATA_PATH))
        logger.info(f"Loaded dataset with language pairs: {list(dataset_dict.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {config.BASE_DATA_PATH}: {e}")
        return

    # --- 2. Initialize Translator ---
    translator = ParallelTranslator(config)
    language_names = get_language_names()
    
    # --- 3. Process Each Language Pair ---
    for lang_pair in dataset_dict:
        logger.info(f"--- Processing language pair: {lang_pair} ---")
        
        # Parse language codes
        if '-' in lang_pair:
            source_code, target_code = lang_pair.split('-')
        else:
            logger.warning(f"Unexpected language pair format: {lang_pair}. Skipping.")
            continue
        
        if config.ROUND_TRIP:
            source_code, target_code = target_code, source_code
            
        source_lang_name = language_names.get(source_code, source_code)
        target_lang_name = language_names.get(target_code, target_code)
        
        logger.info(f"Translating {source_lang_name} -> {target_lang_name}")
        logger.info(f"Dataset size: {len(dataset_dict[lang_pair])} examples")
        
        # --- 4. Translate ---
        if config.DEBUG:
            dataset_dict[lang_pair] = dataset_dict[lang_pair].select(range(10))
        dataset_dict[lang_pair] = dataset_dict[lang_pair].map(
            translator.translate_example,
            num_proc=config.NUM_PROCESSES,
            fn_kwargs={
                'source_lang_name': source_lang_name,
                'target_lang_name': target_lang_name,
            },
            desc=f"Translating {lang_pair}"
        )
        
        logger.info(f"Translated dataset for {lang_pair}")
        
    # --- 5. Save Results ---
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved translated dataset to {output_dir}")
    
    # --- 6. Clean up memory
    logger.info("Translation pipeline completed!")


if __name__ == "__main__":
    run_translation_pipeline(Config()) 