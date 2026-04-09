# %%
"""
Convert FRANK dataset to HuggingFace Dataset format.
Filters for CNN/DM only and applies majority voting for sentence annotations.
"""

import json
from pathlib import Path
from collections import Counter
from datasets import Dataset, DatasetDict

# %%
# Paths to FRANK data files
FRANK_DATA_DIR = Path("/home/yathagata/cef-translation/data/frank/data")

# Load all data files
with open(FRANK_DATA_DIR / "human_annotations.json", "r") as f:
    human_annotations = json.load(f)

with open(FRANK_DATA_DIR / "human_annotations_sentence.json", "r") as f:
    human_annotations_sentence = json.load(f)

with open(FRANK_DATA_DIR / "benchmark_data.json", "r") as f:
    benchmark_data = json.load(f)

print(f"Loaded {len(human_annotations)} human annotations")
print(f"Loaded {len(human_annotations_sentence)} sentence annotations")
print(f"Loaded {len(benchmark_data)} benchmark samples")

# %%
# Create lookup dictionaries for merging
# Key: (hash, model_name)

# Sentence annotations lookup
sentence_annotations_lookup = {}
for item in human_annotations_sentence:
    key = (item["hash"], item["model_name"])
    sentence_annotations_lookup[key] = item

# Human annotations (scores) lookup
scores_lookup = {}
for item in human_annotations:
    key = (item["hash"], item["model_name"])
    scores_lookup[key] = item

print(f"Created lookup for {len(sentence_annotations_lookup)} sentence annotations")
print(f"Created lookup for {len(scores_lookup)} score annotations")

# %%
def apply_majority_voting(
    sentences: list[str], 
    sentence_annotations: list[dict]
) -> dict[str, list[str]]:
    """
    Apply majority voting to sentence annotations.
    
    For each sentence, only include an error if it was identified by 
    at least 50% of annotators (i.e., at least 2 out of 3).
    
    Args:
        sentences: List of sentence strings from the summary.
        sentence_annotations: List of dicts, one per sentence.
            Each dict has annotator_0, annotator_1, annotator_2 keys
            with lists of error codes.
    
    Returns:
        Dict mapping sentence text to list of majority-voted errors.
    """
    result = {}
    
    for idx, sent_annotation in enumerate(sentence_annotations):
        # Get the sentence text (use index if sentences list is shorter)
        if idx < len(sentences):
            sentence_text = sentences[idx]
        else:
            sentence_text = f"[sentence_{idx}]"
        
        # Collect all errors from all annotators
        all_errors = []
        num_annotators = 0
        
        for key in ["annotator_0", "annotator_1", "annotator_2"]:
            if key in sent_annotation:
                all_errors.extend(sent_annotation[key])
                num_annotators += 1
        
        # Count occurrences of each error
        error_counts = Counter(all_errors)
        
        # Keep errors that appear in >= 50% of annotators
        threshold = num_annotators / 2
        majority_errors = [
            error for error, count in error_counts.items() 
            if count >= threshold
        ]
        
        # Sort for consistency
        majority_errors = sorted(majority_errors)
        
        # If no errors passed majority voting, mark as NoE
        if not majority_errors:
            majority_errors = ["NoE"]
        # If NoE is in the list along with actual errors, remove NoE
        elif "NoE" in majority_errors and len(majority_errors) > 1:
            majority_errors = [e for e in majority_errors if e != "NoE"]
        
        result[sentence_text] = majority_errors
    
    return result

# %%
# Test majority voting function
test_sentences = ["The FBI helped the killer access iPhones."]
test_annotation = [
    {
        "annotator_0": ["EntE"],
        "annotator_1": ["EntE", "RelE"],
        "annotator_2": ["RelE", "EntE", "CircE"]
    }
]
print("Test majority voting:")
print(f"Input sentences: {test_sentences}")
print(f"Input annotations: {test_annotation}")
print(f"Output: {apply_majority_voting(test_sentences, test_annotation)}")
# Expected: {"The FBI helped...": ["EntE", "RelE"]} since EntE=3 (100%), RelE=2 (67%)

# %%
# Build the combined dataset - filter for CNN/DM only
combined_data = []

# Score columns to include
score_columns = [
    "Factuality",
    "Semantic_Frame_Errors",
    "Discourse_Errors",
    "Content_Verifiability_Errors",
    "RelE",
    "EntE",
    "CircE",
    "OutE",
    "GramE",
    "CorefE",
    "LinkE",
    "Other",
    "Flip_Semantic_Frame_Errors",
    "Flip_Discourse_Errors",
    "Flip_Content_Verifiability_Errors",
    "Flip_RelE",
    "Flip_EntE",
    "Flip_CircE",
    "Flip_OutE",
    "Flip_GramE",
    "Flip_CorefE",
    "Flip_LinkE",
    "Flip_Other",
]

for item in benchmark_data:
    key = (item["hash"], item["model_name"])
    
    # Get scores from human_annotations
    scores = scores_lookup.get(key)
    if scores is None:
        continue
    
    # Filter for CNN/DM only
    if scores.get("dataset") != "cnndm":
        continue
    
    # Get sentence annotations
    sent_ann = sentence_annotations_lookup.get(key)
    if sent_ann is None:
        continue
    
    # Get sentences and their annotations
    sentences = sent_ann.get("summary_sentences", [])
    raw_annotations = sent_ann.get("summary_sentences_annotations", [])
    
    # Apply majority voting to sentence annotations
    majority_voted_annotations = apply_majority_voting(sentences, raw_annotations)
    
    # Build the row
    # NOTE: sentence_annotation is stored as JSON string because HuggingFace
    # Datasets require dict columns to have consistent keys across all rows.
    # Use json.loads(row["sentence_annotation"]) to parse it back to a dict.
    row = {
        "id": item["hash"],
        "model": item["model_name"],
        "src": item["article"],
        "trg": item["summary"],
        "reference": item.get("reference", ""),
        "split": scores.get("split", "test"),
        "sentence_annotation": json.dumps(majority_voted_annotations),
    }
    
    # Add all score columns
    for col in score_columns:
        row[col] = scores.get(col, None)
    
    combined_data.append(row)

print(f"Combined dataset size (CNN/DM only): {len(combined_data)}")

# %%
# Show sample data
print("\nSample row:")
sample = combined_data[0]
for k, v in sample.items():
    if k == "src":
        print(f"  {k}: {v[:100]}...")
    elif k == "summary":
        print(f"  {k}: {v[:100]}...")
    elif k == "sentence_annotation":
        print(f"  {k}:")
        # Parse JSON string back to dict for display
        annotations = json.loads(v)
        for sent, errors in annotations.items():
            print(f"    - \"{sent[:50]}...\" -> {errors}")
    else:
        print(f"  {k}: {v}")

# %%
# Split into train (validation split) and test (test split)
# FRANK uses "valid" and "test" in their split field

train_data = [row for row in combined_data if row["split"] == "valid"]
test_data = [row for row in combined_data if row["split"] == "test"]

print(f"\nTrain (validation) set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# %%
# Remove the 'split' column as it's no longer needed after splitting
for row in train_data:
    del row["split"]
for row in test_data:
    del row["split"]

# %%
# Create HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
})

print("\nHuggingFace DatasetDict:")
print(dataset_dict)

# %%
# Show dataset info
del dataset_dict["train"]
# %%
dataset_dict.save_to_disk("/data/cef/frank/frank_original")

# %%
ds = load_from_disk("/data/cef/frank/with_cef")
# %%
def map_answers(samp):
    cef_details = samp["cef_details"]
    questions_from_src_new = []
    questions_from_trg_new = []
    for q,a in zip(cef_details["questions_from_src"], cef_details["coverage_answers"]):
        questions_from_src_new.append({
            "question": q["question"],
            "original": q["answer"],
            "predicted": a
        })
    for q,a in zip(cef_details["questions_from_trg"], cef_details["conformity_answers"]):
        questions_from_trg_new.append({
            "question": q["question"],
            "answer": q["answer"],
            "predicted": a
        })
    return {
        "cef_details": {
            "questions_from_src": questions_from_src_new,
            "questions_from_trg": questions_from_trg_new,
            "coverage_answers": cef_details["coverage_answers"],
            "conformity_answers": cef_details["conformity_answers"],
            "consistency_answers": cef_details["consistency_answers"]
        }
    }

# %%
def remove_noe_annotations(samp):
    sentence_annotation = json.loads(samp["sentence_annotation"])
    # Remove "NoE" if other items are present; skip only sentences that are exactly ["NoE"]
    new_errors_list = {}
    for sent, errors in sentence_annotation.items():
        if isinstance(errors, list):
            # Skip sentences that are only ["NoE"]
            if len(errors) == 1 and errors[0] == "NoE":
                continue  # skip this sentence entirely
            # If "NoE" is present with other errors, remove it
            new_errors = [e for e in errors if e != "NoE"]
            new_errors_list[sent] = new_errors
        else:
            new_errors_list[sent] = errors
    return {
        "sentence_annotation": json.dumps(new_errors_list)
    }
# %%
ds = ds.map(map_answers, num_proc=16)
# %%
ds = ds.map(remove_noe_annotations, num_proc=16)
# %%
ds.save_to_disk("/data/cef/frank/frank_original_with_cef")
# %%
ds = load_from_disk("/data/cef/frank/frank_original_with_cef")
# %%
