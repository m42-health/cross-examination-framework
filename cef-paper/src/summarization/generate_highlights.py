#!/usr/bin/env python3
"""
A script to generate highlights/summaries for the CNN/DailyMail dataset using parallel processing.

This script loads the cnndm dataset and generates highlights from the articles (src),
storing them in a new column and saving the dataset.
"""

import os
import gc
import torch
import logging
import time
import requests
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Holds all configuration parameters for the highlights generation pipeline."""
    # API and Model Settings
    # Available models and workers:
    # - worker-9:8892 → Qwen/Qwen3-1.7B (qwen3-1.7b)
    # - worker-3:8892 → Qwen/Qwen3-4B (qwen3-4b)
    # - worker-0:8892 → Qwen/Qwen3-235B-A22B-Instruct-2507 (qwen3-235b)
    # - worker-4:8892 → openai/gpt-oss-20b (gpt-oss-20b)
    # - worker-8:8892 → meta-llama/Llama-3.1-8B-Instruct (llama3.1-8b)
    # - worker-6:8892 → meta-llama/Llama-3.2-3B-Instruct (llama3.2-3b)
    API_URL = "http://worker-8:8892/v1/chat/completions"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_MODEL_NAME = "llama3.1-8b"
    REQUEST_TIMEOUT = 180  # seconds
    DEBUG = False
    
    # Data Settings
    BASE_DATA_PATH = Path("/data/cef/cnndm/cnndm_original")
    OUTPUT_BASE_PATH = Path("/data/cef/cnndm/cnndm_")
    
    # Prompt Settings
    PROMPT_PATH = "/home/yathagata/cef-translation/src/cef_framework/prompts/summarization.yaml"
    
    # Processing Settings
    NUM_PROCESSES = 10  # Number of parallel processes for .map()

    @property 
    def output_path(self):
        """Generate output path with model name."""
        return str(self.OUTPUT_BASE_PATH) + self.SAVE_MODEL_NAME


class HighlightsGenerator:
    """
    A class designed to work with datasets.map() for parallel highlights generation.
    It handles API requests and prompt creation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.api_payload_template = {
            "model": self.config.MODEL_NAME,
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 8192
        }
        
        # Load prompts
        with open(config.PROMPT_PATH) as f:
            prompts = yaml.safe_load(f)
        self.system_prompt = prompts["system_instruction_set"]["basic_v0"]
        self.highlights_prompt = prompts["user_instruction_set"]["gen_highlights_v0"]

    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        return text.strip()

    def _create_prompt(self, article: str) -> List[Dict[str, str]]:
        """Creates the prompt for the API call."""
        cleaned_article = self._clean_text(article)
        user_prompt = self.highlights_prompt.format(text=cleaned_article)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
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

    def generate_highlights(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates highlights for a single example. This is the function passed to .map().
        """
        source_text = example['src']
        prompt = self._create_prompt(source_text)
        highlights = self._make_api_request(prompt)
        
        # Add generated highlights as a new column
        example['trg'] = highlights
        return example


def run_highlights_pipeline(config: Config):
    """Orchestrates the highlights generation for the CNN/DM dataset."""
    
    # --- 1. Load Dataset ---
    logger.info(f"Loading dataset from: {config.BASE_DATA_PATH}")
    try:
        dataset_dict = load_from_disk(str(config.BASE_DATA_PATH))
        logger.info(f"Loaded dataset with splits: {list(dataset_dict.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {config.BASE_DATA_PATH}: {e}")
        return

    # --- 2. Initialize Generator ---
    generator = HighlightsGenerator(config)
    
    logger.info(f"Using model: {config.MODEL_NAME}")
    logger.info(f"API URL: {config.API_URL}")
    
    # --- 3. Process Each Split ---
    for split in dataset_dict:
        logger.info(f"--- Processing split: {split} ---")
        logger.info(f"Dataset size: {len(dataset_dict[split])} examples")
        
        # --- 4. Generate Highlights ---
        if config.DEBUG:
            dataset_dict[split] = dataset_dict[split].select(range(5))
            logger.info(f"DEBUG mode: Processing only 5 examples")
        
        dataset_dict[split] = dataset_dict[split].map(
            generator.generate_highlights,
            num_proc=config.NUM_PROCESSES,
            desc=f"Generating highlights for {split}"
        )
        
        logger.info(f"Generated highlights for {split}")
        
    # --- 5. Save Results ---
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved dataset with generated highlights to {output_dir}")
    
    # --- 6. Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Highlights generation pipeline completed!")


if __name__ == "__main__":
    run_highlights_pipeline(Config())
