#!/bin/bash
# Activate conda environment before running Python
# conda activate medic

# %%
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient
from datasets import load_from_disk
from collections import defaultdict
from pathlib import Path
from fire import Fire
from tqdm import tqdm
import time

# %%
# Check if the API key is available
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
endpoint = "https://api.cognitive.microsofttranslator.com/"
AZURE_TRANSLATOR_REGION = "uaenorth"


DEBUG = True
BASE_DATA_PATH = Path("/data/cef/ntrex/ntrex_lr_azure")
OUTPUT_BASE_PATH = Path("/data/cef/ntrex/ntrex_roundtrip_azure")
ROUNDTRIP = True

try:
    client = TextTranslationClient(
        credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY),
        region=AZURE_TRANSLATOR_REGION
    )

    # Translate text using the correct API method
    input_text = ["This is a sentence that is translated into a Polish language."]
    response = client.translate(
        body=input_text,
        from_language="en",
        to_language=["de", "fr"]
    )
    for translation in response:
        if translation.translations:
            for translated_text in translation.translations:
                print(f"Translated text: {translated_text.text}")
        
except Exception as e:
    print(f"Translation error: {e}")

# %%
def get_azure_lang_code(lang_code: str, reverse: bool = False) -> str:
    """Convert language codes to Azure format."""
    # Mapping from common language codes to NLLB language codes
    lang_mapping = {
        'en': 'en',
        'fr': 'fr',
        'es': 'es', 
        'de': 'de',
        'ar': 'ar',
        'jp': 'ja',
        'tir': 'ti',
        'mri': 'mi',
    }
    if reverse:
        for k, v in lang_mapping.items():
            if v == lang_code:
                return k
        return lang_code
    else:
        return lang_mapping[lang_code] if lang_code in lang_mapping else ""


# %%
def translate_with_retry(client, body, from_language, to_language):
    """Translate text with retry."""
    responses = {}
    for lang in to_language:
        for _ in range(3):
            try:
                response = client.translate(body=body, from_language=from_language, to_language=[lang])
                for translation in response:
                    if translation.translations:
                        for translated_text in translation.translations:
                            responses[lang] = translated_text.text
                break
            except Exception as e:
                print(f"Translation error: {e}")
                time.sleep(30)
    for lang in to_language:
        if lang not in responses:
            responses[lang] = "Translation failed"
    return responses

def run_azure_translator():
    """Run the Azure translator."""
    client = TextTranslationClient(
        credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY),
        region=AZURE_TRANSLATOR_REGION
    )
    
    dataset_dict = load_from_disk(str(BASE_DATA_PATH)) 
    lang_pairs = list(dataset_dict.keys())
    target_langs = [get_azure_lang_code(lang_pair.split('-')[1]) for lang_pair in lang_pairs]
    target_langs = [lang for lang in target_langs if lang != ""]

    if DEBUG:
        for lang_pair in lang_pairs:
            dataset_dict[lang_pair] = dataset_dict[lang_pair].select(range(10))
    
    
    
    if not ROUNDTRIP:
        translations = defaultdict(list)
        first_pair = lang_pairs[0]
        for example in tqdm(dataset_dict[first_pair]):
            responses = translate_with_retry(
                client,
                body=[example['src']],
                from_language="en",
                to_language=target_langs
            )
            for target_lang, translation in responses.items():
                tl = get_azure_lang_code(target_lang, reverse=True)
                translations[f"en-{tl}"].append(translation)
    else:
        translations = defaultdict(list)
        for lang_pair in dataset_dict.keys():
            if get_azure_lang_code(lang_pair.split('-')[1]) in target_langs:
                for example in tqdm(dataset_dict[lang_pair]):
                    responses = translate_with_retry(
                        client,
                        body=[example['trg']],
                        from_language=get_azure_lang_code(lang_pair.split('-')[1]),
                        to_language=["en"]
                    )
                    for target_lang, translation in responses.items():
                        translations[f"{lang_pair}"].append(translation)
    
    breakpoint()
    for lang_pair, lang_translations in translations.items():
        assert len(lang_translations) == len(dataset_dict[lang_pair]), f"Length mismatch for {lang_pair}"
        dataset_dict[lang_pair] = dataset_dict[lang_pair].map(
            lambda x, idx: {
                "trg": lang_translations[idx]
            },
            with_indices=True,
        )
        
    dataset_dict.save_to_disk(str(OUTPUT_BASE_PATH))

# %%
if __name__ == "__main__":
    Fire(run_azure_translator)