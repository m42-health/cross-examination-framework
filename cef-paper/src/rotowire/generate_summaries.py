#!/usr/bin/env python3
"""
Generate NBA game summaries from boxscore data for the RotoWire dataset.

This script loads the boxscore dataset and generates game summaries,
storing them in the trg column and saving the dataset.
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
    """Holds all configuration parameters for the summary generation pipeline."""
    # API and Model Settings
    # Available generation models and workers:
    # - worker-3:8892 → Qwen/Qwen3-1.7B (qwen3-1.7b)
    # - worker-0:8892 → Qwen/Qwen3-4B (qwen3-4b)
    # - worker-9:8892 → meta-llama/Llama-3.2-3B-Instruct (llama3.2-3b)
    # - worker-13:8892 → meta-llama/Llama-3.1-8B-Instruct (llama3.1-8b)
    # - worker-9:8892 → openai/gpt-oss-20b (gpt-oss-20b) (one model TBD)

    API_URL = "http://worker-9:8892/v1/chat/completions"
    MODEL_NAME = "openai/gpt-oss-20b"
    SAVE_MODEL_NAME = "gpt-oss-20b"
    REQUEST_TIMEOUT = 300  # seconds
    DEBUG = False
    
    # Data Settings
    BASE_DATA_PATH = Path("/data/cef/boxscore/boxscore_original")
    OUTPUT_BASE_PATH = Path("/data/cef/boxscore/boxscore_")
    
    # Prompt Settings
    PROMPT_PATH = "/home/yathagata/cef-translation/src/cef_framework/prompts/boxscore.yaml"
    PROMPT_NAME = "gen_summary_v0"  # or "gen_summary_detailed_v0"
    
    # Processing Settings
    NUM_PROCESSES = 10  # Number of parallel processes for .map()

    @property 
    def output_path(self):
        """Generate output path with model name."""
        return str(self.OUTPUT_BASE_PATH) + self.SAVE_MODEL_NAME


class SummaryGenerator:
    """
    A class designed to work with datasets.map() for parallel summary generation.
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
        self.summary_prompt = prompts["user_instruction_set"][config.PROMPT_NAME]

    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if not isinstance(text, str):
            logger.warning(f"Non-string input received: {type(text)}. Returning empty string.")
            return ""
        return text.strip()

    def _create_prompt(self, boxscore: str) -> List[Dict[str, str]]:
        """Creates the prompt for the API call."""
        cleaned_boxscore = self._clean_text(boxscore)
        user_prompt = self.summary_prompt.format(text=cleaned_boxscore)
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

    def generate_summary(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a game summary for a single example. This is the function passed to .map().
        """
        boxscore = example['src']
        prompt = self._create_prompt(boxscore)
        summary = self._make_api_request(prompt)
        
        # Replace trg with generated summary
        example['trg'] = summary
        return example


def run_summary_generation_pipeline(config: Config):
    """Orchestrates the summary generation for the boxscore dataset."""
    
    # --- 1. Load Dataset ---
    logger.info(f"Loading dataset from: {config.BASE_DATA_PATH}")
    try:
        dataset_dict = load_from_disk(str(config.BASE_DATA_PATH))
        logger.info(f"Loaded dataset with splits: {list(dataset_dict.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {config.BASE_DATA_PATH}: {e}")
        return

    # --- 2. Initialize Generator ---
    generator = SummaryGenerator(config)
    
    logger.info(f"Using model: {config.MODEL_NAME}")
    logger.info(f"API URL: {config.API_URL}")
    logger.info(f"Prompt: {config.PROMPT_NAME}")
    
    # --- 3. Process Each Split ---
    for split in dataset_dict:
        logger.info(f"--- Processing split: {split} ---")
        logger.info(f"Dataset size: {len(dataset_dict[split])} examples")
        
        # --- 4. Generate Summaries ---
        if config.DEBUG:
            dataset_dict[split] = dataset_dict[split].select(range(5))
            logger.info(f"DEBUG mode: Processing only 5 examples")
        
        dataset_dict[split] = dataset_dict[split].map(
            generator.generate_summary,
            num_proc=config.NUM_PROCESSES,
            desc=f"Generating summaries for {split}"
        )
        
        logger.info(f"Generated summaries for {split}")
        
    # --- 5. Save Results ---
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved dataset with generated summaries to {output_dir}")
    
    # --- 6. Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Summary generation pipeline completed!")


if __name__ == "__main__":
    run_summary_generation_pipeline(Config())

