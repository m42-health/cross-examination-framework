# %%
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np

# %%
ds = load_dataset("facebook/flores", "all", trust_remote_code=True)

# %%
ds_combined = concatenate_datasets([ds["devtest"], ds["dev"]])
# %%
columns_to_keep = ['id', 'URL', 'domain', 'topic', 'sentence_eng_Latn', 'sentence_fra_Latn', 'sentence_spa_Latn', 'sentence_deu_Latn']
# %%
ds_combined = ds_combined.select_columns(columns_to_keep)
# %%
# Group by URL and concatenate sentences
grouped_data = {}
for item in ds_combined:
    url = item['URL']
    if url not in grouped_data:
        grouped_data[url] = {
            'id': f"grouped_{len(grouped_data)}",
            'URL': url,
            'domain': item['domain'],
            'topic': item['topic'],
            'sentence_eng_Latn': '',
            'sentence_fra_Latn': '',
            'sentence_spa_Latn': '',
            'sentence_deu_Latn': ''
        }
    # Concatenate sentences directly as strings
    grouped_data[url]['sentence_eng_Latn'] += item['sentence_eng_Latn'] + ' '
    grouped_data[url]['sentence_fra_Latn'] += item['sentence_fra_Latn'] + ' '
    grouped_data[url]['sentence_spa_Latn'] += item['sentence_spa_Latn'] + ' '
    grouped_data[url]['sentence_deu_Latn'] += item['sentence_deu_Latn'] + ' '

# Create new dataset from grouped data
ds_grouped = Dataset.from_list(list(grouped_data.values()))

# %%
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
# %%
# Tokenize and calculate token counts for each language column using ds.map
languages = {
    "en": "sentence_eng_Latn",
    "fr": "sentence_fra_Latn",
    "es": "sentence_spa_Latn",
    "de": "sentence_deu_Latn"
}

ds_dict = DatasetDict()

for lang in ["fr", "es", "de"]:
    # Create a copy of the dataset with only English and the target language
    columns_to_keep = ['id', 'URL', 'domain', 'topic', 'sentence_eng_Latn', languages[lang]]
    
    ds_grouped_filtered = ds_grouped.select_columns(columns_to_keep)
    ds_grouped_filtered = ds_grouped_filtered.rename_column('sentence_eng_Latn', 'src')
    ds_grouped_filtered = ds_grouped_filtered.rename_column(languages[lang], 'trg')
    ds_dict[f"en-{lang}"] = ds_grouped_filtered

# %%
ds_dict.save_to_disk("/data/cef/flores")
# %%
