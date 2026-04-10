# %%
import os
from google.cloud import translate_v3 as translate
from datasets import load_from_disk
from collections import defaultdict
from pathlib import Path
from fire import Fire
from tqdm import tqdm
# %%
PROJECT_ID = "perspective-api-289610"
PARENT = f"projects/{PROJECT_ID}"
client = translate.TranslationServiceClient()
DEBUG = False
BASE_DATA_PATH = Path("/data/cef/ntrex/ntrex_lr_google")
OUTPUT_BASE_PATH = Path("/data/cef/ntrex/ntrex_roundtrip_google")
ROUNDTRIP = True

# %%

def convert_local_to_google_code(lang_code: str, reverse: bool = False):
    lang_mapping = {
        'jp': 'ja',
        'dzo': 'dz',
        'tir': 'ti',
        'mri': 'mi',
    }
    if reverse:
        lang_mapping = {v: k for k, v in lang_mapping.items()}
    return lang_mapping[lang_code] if lang_code in lang_mapping else lang_code
    

# %%
def translate_text(text: str, target_language_code: str, source_language_code="en") -> translate.Translation:
    for _ in range(3):
        try:
            response = client.translate_text(
                parent=PARENT,
                contents=[text],
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
            return response.translations[0]
        except Exception as e:
            print(f"Translation error: {e}")
    return None

# %%
def run_google_translator():
    """Run the Google translator."""
    global client
    
    dataset_dict = load_from_disk(str(BASE_DATA_PATH)) 
    lang_pairs = list(dataset_dict.keys())
    target_langs = [convert_local_to_google_code(lang_pair.split('-')[1]) for lang_pair in lang_pairs]
    unsupported_langs = ['eus', 'khm']
    target_langs = [lang for lang in target_langs if lang != "" and lang not in unsupported_langs]
    print(target_langs)
    
    if DEBUG:
        for lang_pair in lang_pairs:
            dataset_dict[lang_pair] = dataset_dict[lang_pair].select(range(10))

    if not ROUNDTRIP:
        translations = defaultdict(list)
        first_pair = lang_pairs[0]
        for example in tqdm(dataset_dict[first_pair]):
            for target_lang in target_langs:
                translated = translate_text(
                    example['src'],
                    target_language_code=target_lang,
                    source_language_code="en"
                )
                if translated:
                    tl = convert_local_to_google_code(target_lang, reverse=True)
                    translations[f"en-{tl}"].append(translated.translated_text)
                else:
                    tl = convert_local_to_google_code(target_lang, reverse=True)
                    translations[f"en-{tl}"].append("Translation failed")
    else:
        translations = defaultdict(list)
        for lang_pair in dataset_dict.keys():
            if convert_local_to_google_code(lang_pair.split('-')[1]) in target_langs:
                for example in tqdm(dataset_dict[lang_pair]):
                    translated = translate_text(
                        example['trg'],
                        target_language_code="en",
                        source_language_code=convert_local_to_google_code(lang_pair.split('-')[1])
                    )
                    if translated:
                        translations[f"{lang_pair}"].append(translated.translated_text)
                    else:
                        translations[f"{lang_pair}"].append("Translation failed")
    
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
    Fire(run_google_translator)