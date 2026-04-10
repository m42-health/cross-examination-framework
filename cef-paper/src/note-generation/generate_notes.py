#!/usr/bin/env python3
"""
Generate clinical notes from doctor-patient conversations for the ACI benchmark.

This script loads the ACI dataset and generates clinical notes from conversations (src),
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
    """Holds all configuration parameters for the note generation pipeline."""
    # API and Model Settings
    # Available models and workers:
    # - worker-9:8892 → Qwen/Qwen3-1.7B (qwen3-1.7b)
    # - worker-3:8892 → Qwen/Qwen3-4B (qwen3-4b)
    # - worker-4:8892 → openai/gpt-oss-20b (gpt-oss-20b)
    # - worker-8:8892 → meta-llama/Llama-3.1-8B-Instruct (llama3.1-8b)
    # - worker-6:8892 → meta-llama/Llama-3.2-3B-Instruct (llama3.2-3b)
    API_URL = "http://worker-6:8892/v1/chat/completions"
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    SAVE_MODEL_NAME = "llama3.2-3b"
    REQUEST_TIMEOUT = 300  # seconds (longer for clinical notes)
    DEBUG = False
    
    # Data Settings
    BASE_DATA_PATH = Path("/data/cef/aci/aci_original")
    OUTPUT_BASE_PATH = Path("/data/cef/aci/aci_")
    
    # Prompt Settings
    PROMPT_PATH = "/home/yathagata/cef-translation/src/cef_framework/prompts/note_generation.yaml"
    PROMPT_NAME = "gen_note_v0"  # or "gen_note_structured_v0"
    
    # Processing Settings
    NUM_PROCESSES = 10  # Number of parallel processes for .map()

    @property 
    def output_path(self):
        """Generate output path with model name."""
        return str(self.OUTPUT_BASE_PATH) + self.SAVE_MODEL_NAME


class NoteGenerator:
    """
    A class designed to work with datasets.map() for parallel note generation.
    It handles API requests and prompt creation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.api_payload_template = {
            "model": self.config.MODEL_NAME,
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 16384  # Clinical notes can be long
        }
        
        # Load prompts
        with open(config.PROMPT_PATH) as f:
            prompts = yaml.safe_load(f)
        self.system_prompt = prompts["system_instruction_set"]["basic_v0"]
        self.note_prompt = prompts["user_instruction_set"][config.PROMPT_NAME]

    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if not isinstance(text, str):
            logger.warning(f"Non-string input received: {type(text)}. Returning empty string.")
            return ""
        return text.strip()

    def _create_prompt(self, conversation: str) -> List[Dict[str, str]]:
        """Creates the prompt for the API call."""
        cleaned_conversation = self._clean_text(conversation)
        user_prompt = self.note_prompt.format(text=cleaned_conversation)
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

    def generate_note(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a clinical note for a single example. This is the function passed to .map().
        """
        conversation = example['src']
        prompt = self._create_prompt(conversation)
        note = self._make_api_request(prompt)
        
        # Replace trg with generated note
        example['trg'] = note
        return example


def run_note_generation_pipeline(config: Config):
    """Orchestrates the note generation for the ACI dataset."""
    
    # --- 1. Load Dataset ---
    logger.info(f"Loading dataset from: {config.BASE_DATA_PATH}")
    try:
        dataset_dict = load_from_disk(str(config.BASE_DATA_PATH))
        logger.info(f"Loaded dataset with splits: {list(dataset_dict.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {config.BASE_DATA_PATH}: {e}")
        return

    # --- 2. Initialize Generator ---
    generator = NoteGenerator(config)
    
    logger.info(f"Using model: {config.MODEL_NAME}")
    logger.info(f"API URL: {config.API_URL}")
    logger.info(f"Prompt: {config.PROMPT_NAME}")
    
    # --- 3. Process Each Split ---
    for split in dataset_dict:
        logger.info(f"--- Processing split: {split} ---")
        logger.info(f"Dataset size: {len(dataset_dict[split])} examples")
        
        # --- 4. Generate Notes ---
        if config.DEBUG:
            dataset_dict[split] = dataset_dict[split].select(range(5))
            logger.info(f"DEBUG mode: Processing only 5 examples")
        
        dataset_dict[split] = dataset_dict[split].map(
            generator.generate_note,
            num_proc=config.NUM_PROCESSES,
            desc=f"Generating notes for {split}"
        )
        
        logger.info(f"Generated notes for {split}")
        
    # --- 5. Save Results ---
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved dataset with generated notes to {output_dir}")
    
    # --- 6. Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Note generation pipeline completed!")


if __name__ == "__main__":
    run_note_generation_pipeline(Config())

