import os
from datasets import load_from_disk
from transformers import AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

# Ground truth paths
gt_paths = {
    "en-ar": "/data/cef/NTREX/en-ar",
    "en-jp": "/data/cef/NTREX/en-jp"
}

# Model output base paths
model_paths = {
    "Llama3.2-1B": "/data/cef/NTREX/ntrex_Llama3.2-1B-Instruct_all_languages",
    "Helsinki": "/data/cef/NTREX/ntrex_helsinki_translation/ar_jp_languages",
    "Qwen2.5-72B": "/data/cef/NTREX/ntrex_qwen2.5-72b_all_languages",
    "Qwen3-0.6B": "/data/cef/NTREX/ntrex_qwen3-0.6b_all_languages",
}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Evaluation config
LANGS = ["en-ar", "en-jp"]
PERCENTAGE_THRESHOLD = 0.1  # 10% threshold for incomplete translations

def get_token_length(text):
    return len(tokenizer.tokenize(text)) if text else 0

def compare_tokens(gt_ds, model_ds, lang):
    full_count = 0
    not_full_count = 0
    full_example = None
    not_full_example = None
    
    for i in range(len(gt_ds)):
        src_text = gt_ds[i]["src"]  # Keep source for reference
        gt_text = gt_ds[i]["trg"]
        model_text = model_ds[i]["trg"] if "trg" in model_ds[i] else ""
        
        src_tokens = get_token_length(src_text)
        gt_tokens = get_token_length(gt_text)
        model_tokens = get_token_length(model_text)
        
        # Calculate if model translation is 10% shorter than ground truth target
        if gt_tokens > 0:  # Avoid division by zero
            length_ratio = model_tokens / gt_tokens
            is_incomplete = length_ratio < (1 - PERCENTAGE_THRESHOLD)
        else:
            is_incomplete = model_tokens == 0
        
        if is_incomplete:
            not_full_count += 1
            if not_full_example is None:
                not_full_example = {
                    "index": i,
                    "src_text": src_text,
                    "gt_text": gt_text,
                    "model_text": model_text,
                    "src_tokens": src_tokens,
                    "gt_tokens": gt_tokens,
                    "model_tokens": model_tokens,
                    "length_ratio": length_ratio if gt_tokens > 0 else 0,
                    "percentage_shorter": (1 - length_ratio) * 100 if gt_tokens > 0 else 100
                }
        else:
            full_count += 1
            if full_example is None:
                full_example = {
                    "index": i,
                    "src_text": src_text,
                    "gt_text": gt_text,
                    "model_text": model_text,
                    "src_tokens": src_tokens,
                    "gt_tokens": gt_tokens,
                    "model_tokens": model_tokens,
                    "length_ratio": length_ratio if gt_tokens > 0 else 0,
                    "percentage_shorter": (1 - length_ratio) * 100 if gt_tokens > 0 else 0
                }
    
    total = full_count + not_full_count
    full_pct = round((full_count / total) * 100, 2) if total > 0 else 0
    not_full_pct = round((not_full_count / total) * 100, 2) if total > 0 else 0
    
    return full_pct, not_full_pct, full_count, not_full_count, full_example, not_full_example

def print_example(example, title):
    logger.info(f"\n🔹 {title} (Index: {example['index']}, {example['percentage_shorter']:.1f}% shorter than ground truth)")
    logger.info(f"[Source] ({example['src_tokens']} tokens): {example['src_text']}")
    logger.info(f"[Ground Truth] ({example['gt_tokens']} tokens): {example['gt_text']}")
    logger.info(f"[Model Output] ({example['model_tokens']} tokens): {example['model_text']}")
    logger.info(f"Length ratio (model/ground_truth): {example['length_ratio']:.2f}")

def main():
    for model_name, model_path in model_paths.items():
        logger.info(f"\n=== Model: {model_name} ===")
        
        for lang in LANGS:
            try:
                gt_ds = load_from_disk(gt_paths[lang])
                model_lang_path = os.path.join(model_path, lang)
                model_ds = load_from_disk(model_lang_path)
                
                full_pct, not_full_pct, full_count, not_full_count, full_example, not_full_example = compare_tokens(
                    gt_ds, model_ds, lang
                )
                
                logger.info(
                    f"\n{lang.upper()} | Complete: {full_pct}% ({full_count}) | Incomplete (>10% shorter than GT): {not_full_pct}% ({not_full_count})"
                )
                
                if full_example:
                    print_example(full_example, "✅ Example Complete Translation")
                if not_full_example:
                    print_example(not_full_example, "❌ Example Incomplete Translation")
                    
            except Exception as e:
                logger.error(f"Failed to load or compare {lang.upper()} for {model_name}: {e}")

if __name__ == "__main__":
    main()