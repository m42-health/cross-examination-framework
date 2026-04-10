#!/usr/bin/env python3
"""
A script to translate multiple language pairs using NLLB model with multi-GPU parallel processing.

This script loads the ntrex dataset and translates each language pair using Facebook's
NLLB model across multiple GPUs, replacing the target column with new translations and 
saving each language pair as a separate dataset.
"""

import os
import gc
import torch
import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import DatasetDict, load_from_disk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import multiprocessing as mp
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Set multiprocessing start method for CUDA compatibility
if torch.cuda.is_available():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global lock for model loading to prevent race conditions
model_loading_lock = threading.Lock()

# --- Configuration ---
class Config:
    """Holds all configuration parameters for the NLLB translation pipeline."""
    # Model Settings
    MODEL_NAME = "facebook/nllb-200-3.3B"
    SAVE_MODEL_NAME = "nllb-3.3B"
    MAX_LENGTH = 2000
    DEBUG = False
    USE_LIST = True
    # Data Settings
    BASE_DATA_PATH = Path("/data/cef/ntrex/ntrex_original")
    OUTPUT_BASE_PATH = Path("/data/cef/ntrex/ntrex_")
        
    ROUND_TRIP = True
    
    # Multi-GPU Processing Settings
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
    MAX_WORKERS = None  # Will be set to NUM_GPUS
    BATCH_SIZE = 8  # Reduced batch size to avoid memory issues
    NUM_PROCESSES = 1  # Use 1 process per GPU to avoid CUDA multiprocessing issues
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory management
    MAX_GPU_MEMORY_FRACTION = 0.85  # Leave some memory for other processes

    def __post_init__(self):
        """Initialize computed properties after object creation."""
        if self.MAX_WORKERS is None:
            # Use fewer workers than GPUs to avoid memory conflicts
            self.MAX_WORKERS = min(self.NUM_GPUS, 6)
        
        # Set GPU memory fraction for each GPU
        if torch.cuda.is_available():
            for i in range(self.NUM_GPUS):
                try:
                    torch.cuda.set_per_process_memory_fraction(
                        self.MAX_GPU_MEMORY_FRACTION, device=i
                    )
                except Exception as e:
                    logger.warning(f"Could not set memory fraction for GPU {i}: {e}")
        
        logger.info(f"Detected {self.NUM_GPUS} GPUs, using {self.MAX_WORKERS} workers")
        logger.info(f"Batch size: {self.BATCH_SIZE}, Processes per GPU: {self.NUM_PROCESSES}")

    @property 
    def output_path(self):
        """Generate output path with model name."""
        return str(self.OUTPUT_BASE_PATH) + str(self.SAVE_MODEL_NAME)

def get_nllb_lang_code(lang_code: str) -> str:
    """Convert language codes to NLLB format."""
    # Mapping from common language codes to NLLB language codes
    lang_mapping = {
        'en': 'eng_Latn',
        'fr': 'fra_Latn',
        'es': 'spa_Latn', 
        'de': 'deu_Latn',
        'ar': 'arb_Arab',
        'jp': 'jpn_Jpan',
        'tir': 'tir_Ethi',
        'eus': 'eus_Latn',
        'dzo': 'dzo_Tibt',
        'mri': 'mri_Latn',
        'khm': 'khm_Khmr',
    }
    return lang_mapping.get(lang_code, lang_code)


class MultiGPUNLLBTranslator:
    """
    A class designed for multi-GPU NLLB translation with optimized batching.
    Each instance is bound to a specific GPU device.
    """
    def __init__(self, config: Config, source_lang: str, target_lang: str, device_id: int):
        self.config = config
        self.device_id = device_id
        self.pipe = None
        self.src_lang = source_lang
        self.tgt_lang = target_lang
        self.round_trip = config.ROUND_TRIP
        self.use_list = config.USE_LIST
        self._load_model()

    def _load_model(self):
        """Load the NLLB model and tokenizer on specific GPU."""
        logger.info(f"Loading NLLB model on GPU {self.device_id}: {self.config.MODEL_NAME}")
        
        with model_loading_lock:  # Prevent concurrent downloads
            try:
                # Set the device for this process
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.device_id)
                
                # Load model and tokenizer separately to avoid meta tensor issues
                logger.info(f"[GPU {self.device_id}] Loading model weights...")
                
                if torch.cuda.is_available():
                    # For CUDA, use device_map and don't specify device in pipeline
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.config.MODEL_NAME,
                        torch_dtype=torch.float16,
                        device_map={"": self.device_id},
                        low_cpu_mem_usage=True,
                        local_files_only=False
                    )
                    device_arg = None  # Don't set device when using device_map
                else:
                    # For CPU, load normally
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.config.MODEL_NAME,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        local_files_only=False
                    )
                    device_arg = -1
                
                logger.info(f"[GPU {self.device_id}] Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.MODEL_NAME,
                    local_files_only=False
                )
                
                # Create pipeline with pre-loaded model and tokenizer
                logger.info(f"[GPU {self.device_id}] Creating translation pipeline...")
                pipeline_kwargs = {
                    "task": "translation",
                    "model": model,
                    "tokenizer": tokenizer,
                    "src_lang": self.src_lang,
                    "tgt_lang": self.tgt_lang,
                    "max_length": self.config.MAX_LENGTH,
                    "batch_size": self.config.BATCH_SIZE
                }
                
                # Only add device if not using device_map
                if device_arg is not None:
                    pipeline_kwargs["device"] = device_arg
                
                self.pipe = pipeline(**pipeline_kwargs)
                
                logger.info(f"NLLB model loaded successfully on GPU {self.device_id}")
                
            except Exception as e:
                logger.error(f"Failed to load NLLB model on GPU {self.device_id}: {e}")
                logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                raise

    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for translation."""
        if not isinstance(text, str):
            logger.warning(f"Non-string input received: {type(text)}. Returning empty string.")
            return ""
        # Remove excessive whitespace and newlines
        return text.replace("\n", " ").strip()

    def translate_batch(self, text_list: List[str]) -> List[str]:
        """Translate a batch of texts using NLLB with optimized batching."""
        try:
            # Clean all texts
            cleaned_texts = [self._clean_text(text) for text in text_list]
            
            # Filter out empty texts but keep track of their positions
            non_empty_texts = []
            non_empty_indices = []
            for i, text in enumerate(cleaned_texts):
                if text:
                    non_empty_texts.append(text)
                    non_empty_indices.append(i)
            
            if not non_empty_texts:
                return [""] * len(text_list)
            
            # Perform batch translation
            results = self.pipe(non_empty_texts, max_length=self.config.MAX_LENGTH)
            
            # Handle different result formats
            translations = []
            for result in results:
                if isinstance(result, dict):
                    translations.append(result.get('translation_text', '').strip())
                elif isinstance(result, list) and len(result) > 0:
                    translations.append(result[0].get('translation_text', '').strip())
                else:
                    logger.warning(f"Unexpected translation result format: {type(result)}")
                    translations.append("")
            
            # Reconstruct full result list with empty strings for originally empty texts
            full_results = [""] * len(text_list)
            for i, translation in zip(non_empty_indices, translations):
                full_results[i] = translation
                
            return full_results
            
        except Exception as e:
            logger.error(f"Batch translation failed on GPU {self.device_id}: {e}")
            return ["ERROR: TRANSLATION_FAILED"] * len(text_list)

    def translate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates a single example with batch processing.
        """
        # Determine source text based on round trip and format preferences
        if self.round_trip:
            source_text_list = example['trg_list'] if self.use_list else [example['trg']]
        else:
            source_text_list = example['src_list'] if self.use_list else [example['src']]
            
        translation_list = self.translate_batch(source_text_list)
        
        # Replace the 'trg' column with the translation
        if self.use_list:
            return {
                'trg_list': translation_list,
                "trg": " ".join(translation_list)
            }
        else:
            return {
                'trg': translation_list[0]
            }

    def cleanup(self):
        """Clean up GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def process_language_pair(args: Tuple[str, Any, Config, int]) -> Tuple[str, bool, str]:
    """
    Process a single language pair on a specific GPU.
    Returns (lang_pair, success, error_message)
    """
    lang_pair, dataset, config, device_id = args
    
    try:
        logger.info(f"[GPU {device_id}] Processing language pair: {lang_pair}")
        
        # Parse language codes
        if '-' in lang_pair:
            source_code, target_code = lang_pair.split('-')
        else:
            return lang_pair, False, f"Unexpected language pair format: {lang_pair}"
        
        if config.ROUND_TRIP:
            source_lang_code = get_nllb_lang_code(target_code)
            target_lang_code = get_nllb_lang_code(source_code)
        else:
            source_lang_code = get_nllb_lang_code(source_code)
            target_lang_code = get_nllb_lang_code(target_code)
        
        # Initialize translator for this GPU
        translator = MultiGPUNLLBTranslator(config, source_lang=source_lang_code, 
                                          target_lang=target_lang_code, device_id=device_id)
        
        # Process dataset
        if config.DEBUG:
            dataset = dataset.select(range(10))
        
        logger.info(f"[GPU {device_id}] Translating {len(dataset)} examples for {lang_pair}")
        
        # Use dataset.map without multiprocessing to avoid CUDA conflicts
        # The parallelism comes from multiple GPUs processing different language pairs
        translated_dataset = dataset.map(
            translator.translate_example,
            num_proc=1,  # Always use single process to avoid CUDA multiprocessing issues
            desc=f"[GPU {device_id}] Translating {source_lang_code}-{target_lang_code}",
            batched=False
        )
        
        # Clean up translator
        translator.cleanup()
        
        logger.info(f"[GPU {device_id}] Completed translation for {lang_pair}")
        return lang_pair, True, translated_dataset
        
    except Exception as e:
        error_msg = f"Failed to translate {lang_pair} on GPU {device_id}: {e}"
        logger.error(error_msg)
        return lang_pair, False, error_msg


def get_language_names():
    """Returns a mapping of language codes to language names."""
    return {
        'en': 'English',
        'fr': 'French', 
        'es': 'Spanish',
        'de': 'German',
        'ar': 'Arabic',
        'jp': 'Japanese',
    }


def run_multi_gpu_nllb_translation_pipeline(config: Config):
    """Orchestrates the translation of all language pairs using multiple GPUs."""
    
    # Initialize config computed properties
    config.__post_init__()
    
    # --- 1. Load Dataset ---
    logger.info(f"Loading dataset from: {config.BASE_DATA_PATH}")
    try:
        dataset_dict = load_from_disk(str(config.BASE_DATA_PATH))
        logger.info(f"Loaded dataset with language pairs: {list(dataset_dict.keys())}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {config.BASE_DATA_PATH}: {e}")
        return

    # --- 2. Prepare Language Pairs for Processing ---
    language_names = get_language_names()
    lang_pairs = list(dataset_dict.keys())
    
    logger.info(f"Processing {len(lang_pairs)} language pairs across {config.MAX_WORKERS} GPUs")
    
    # --- 3. Process Language Pairs in Parallel ---
    results = {}
    failed_pairs = []
    
    # Create arguments for parallel processing
    # Distribute language pairs across available GPUs
    processing_args = []
    for i, lang_pair in enumerate(lang_pairs):
        device_id = i % config.NUM_GPUS
        processing_args.append((lang_pair, dataset_dict[lang_pair], config, device_id))
    
    # Process with ThreadPoolExecutor to handle multiple GPUs
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # Submit all translation jobs
        future_to_langpair = {
            executor.submit(process_language_pair, args): args[0] 
            for args in processing_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_langpair):
            lang_pair = future_to_langpair[future]
            try:
                lang_pair_result, success, data_or_error = future.result()
                if success:
                    results[lang_pair_result] = data_or_error
                    logger.info(f"✓ Successfully translated {lang_pair_result}")
                else:
                    failed_pairs.append(lang_pair_result)
                    logger.error(f"✗ Failed to translate {lang_pair_result}: {data_or_error}")
            except Exception as e:
                failed_pairs.append(lang_pair)
                logger.error(f"✗ Exception during translation of {lang_pair}: {e}")
    
    # --- 4. Update Dataset Dict with Results ---
    for lang_pair, translated_dataset in results.items():
        dataset_dict[lang_pair] = translated_dataset
    
    # --- 5. Save Results ---
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved translated dataset to {output_dir}")
    
    # --- 6. Summary ---
    logger.info(f"Translation pipeline completed!")
    logger.info(f"Successfully translated: {len(results)} language pairs")
    if failed_pairs:
        logger.warning(f"Failed translations: {failed_pairs}")
    
    # Final cleanup
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    config = Config()
    logger.info(f"Using {config.NUM_GPUS} GPUs")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    run_multi_gpu_nllb_translation_pipeline(config)




