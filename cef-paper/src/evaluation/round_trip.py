#!/usr/bin/env python3
"""
A script to perform iterative round-trip translations to study "semantic drift".

This version is optimized for speed by parallelizing API calls using
the Hugging Face datasets.map() function.
"""

import os
import gc
import torch
import logging
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Holds all configuration parameters for the translation pipeline."""
    # API and Model Settings
    API_URL = "http://worker-10:8000/v1/chat/completions"
    MODEL_NAME = "/models_llm/Qwen2.5-72B-Instruct"
    MAX_INPUT_CHARS = 8192
    REQUEST_TIMEOUT = 120  # seconds

    # Data Settings
    BASE_DATA_PATH = Path("/data/cef/europarl/filtered_1381/cleaned_data")
    OUTPUT_PATH = Path("/data/cef/europarl/filtered_1381/cleaned_data/europarl_qwen2.5-72b_all_languages_new_translation_results")
    
    # Experiment Settings
    SOURCE_LANG_CODE = "en"
    TARGET_LANG_CODE = "de"
    SOURCE_LANG_NAME = "English"
    TARGET_LANG_NAME = "German"
    INITIAL_DATASET_DIR = f"{SOURCE_LANG_CODE}-{TARGET_LANG_CODE}"
    
    # Processing Settings
    NUM_ITERATIONS = 1
    NUM_PROCESSES = 16 # Number of parallel processes for .map()

class ParallelTranslator:
    """
    A class designed to work with datasets.map() for parallel translation.
    It handles API requests and prompt creation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.api_payload_template = {
            "model": self.config.MODEL_NAME,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 8192
        }

    def _clean_text(self, text: str) -> str:
        """Remove special charactuers."""
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
        try:
            response = requests.post(self.config.API_URL, json=payload, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                return content
            else:
                logger.warning(f"PID {os.getpid()}: Unexpected API response format: {result}")
                return "ERROR: UNEXPECTED_FORMAT"
        except requests.exceptions.RequestException as e:
            logger.error(f"PID {os.getpid()}: API request failed: {e}")
            if attempt < 3:
                time.sleep(2 * attempt)
                return self._make_api_request(messages, attempt + 1)
            else:
                logger.error(f"PID {os.getpid()}: Max retries reached. Returning error string.")
                return "ERROR: MAX_RETRIES"

    def translate_batch(self, batch: Dict[str, List[Any]], source_col: str, source_lang_name: str, target_lang_name: str) -> Dict[str, List[str]]:
        """
        Processes a batch of examples. This is the function passed to .map().
        It receives a dictionary of lists and must return a dictionary of lists.
        """
        translations = []
        source_texts = batch[source_col]
        for text in source_texts:
            prompt = self._create_prompt(source_lang_name, target_lang_name, text)
            translated_text = self._make_api_request(prompt)
            translations.append(translated_text)
        return {"translation": translations}


def run_semantic_drift_experiment(config: Config):
    """Orchestrates the entire round-trip translation experiment."""
    
    # --- 1. Load and Prepare Initial Dataset ---
    dataset_path = config.BASE_DATA_PATH / config.INITIAL_DATASET_DIR
    logger.info(f"Loading initial dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(str(dataset_path))
        if 'src' in dataset.column_names:
             dataset = dataset.rename_column('src', 'original_source')
        elif config.SOURCE_LANG_CODE in dataset.column_names:
             dataset = dataset.rename_column(config.SOURCE_LANG_CODE, 'original_source')
        else: 
             raise ValueError(f"Could not find source column ('src' or '{config.SOURCE_LANG_CODE}') in dataset.")
    except Exception as e:
        logger.error(f"Failed to load or prepare dataset from {dataset_path}: {e}")
        return

    # --- 2. Run Iterative Translation ---
    translator = ParallelTranslator(config)
    current_source_col = 'original_source'
    
    for i in range(1, config.NUM_ITERATIONS + 1):
        logger.info(f"--- Starting Iteration {i}/{config.NUM_ITERATIONS} ---")

        # --- Step A: Translate Source -> Target (e.g., en -> es) ---
        logger.info(f"Translating from '{current_source_col}' ({config.SOURCE_LANG_NAME} -> {config.TARGET_LANG_NAME})")
        target_col_name = f"{config.TARGET_LANG_CODE}_iter_{i}"
        
        translation_result = dataset.map(
            translator.translate_batch,
            batched=True,
            batch_size=8,
            num_proc=config.NUM_PROCESSES,
            fn_kwargs={
                'source_col': current_source_col,
                'source_lang_name': config.SOURCE_LANG_NAME,
                'target_lang_name': config.TARGET_LANG_NAME
            }
        )
        dataset = dataset.add_column(target_col_name, translation_result['translation'])
        current_source_col = target_col_name

        # # --- Step B: Translate Target -> Source (e.g., es -> en) ---
        # logger.info(f"Translating from '{current_source_col}' ({config.TARGET_LANG_NAME} -> {config.SOURCE_LANG_NAME})")
        # target_col_name = f"{config.SOURCE_LANG_CODE}_iter_{i}"

        # translation_result = dataset.map(
        #     translator.translate_batch,
        #     batched=True,
        #     batch_size=8,
        #     num_proc=config.NUM_PROCESSES,
        #     fn_kwargs={
        #         'source_col': current_source_col,
        #         'source_lang_name': config.TARGET_LANG_NAME,
        #         'target_lang_name': config.SOURCE_LANG_NAME
        #     }
        # )
        # dataset = dataset.add_column(target_col_name, translation_result['translation'])
        # current_source_col = target_col_name
        
        # --- 3. Save Incremental Progress ---
        output_dir = config.OUTPUT_PATH / f"semantic_drift_{config.SOURCE_LANG_CODE}_{config.TARGET_LANG_CODE}_iter_{i}"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))
        logger.info(f"Saved intermediate results for iteration {i} to {output_dir}")

    logger.info("Semantic drift experiment completed!")
    final_output_dir = config.OUTPUT_PATH / f"semantic_drift_{config.SOURCE_LANG_CODE}_{config.TARGET_LANG_CODE}_final"
    dataset.save_to_disk(str(final_output_dir))
    logger.info(f"Final dataset saved to: {final_output_dir}")
    
    # Clean up memory
    del dataset
    gc.collect()

if __name__ == "__main__":
    run_semantic_drift_experiment(Config())