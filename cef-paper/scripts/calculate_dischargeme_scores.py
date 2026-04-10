# %%
from datasets import load_from_disk, DatasetDict
from pathlib import Path
import re
import json
import numpy as np

# %%
# Define base paths
results_cef_path = Path("/data/cef/dischargeme/results_cef")
results_di_cef_path = Path("/data/cef/dischargeme/results_di_cef")

# %%
def load_all_model_results(base_path: Path) -> DatasetDict:
    """
    Load all model results from a directory into a DatasetDict.
    Each model's test split is stored with the model name as key.
    """
    model_datasets = {}
    
    for folder in base_path.iterdir():
        # Skip debug folder and non-directories
        if not folder.is_dir() or folder.name == "debug":
            continue
        
        # Extract model name (everything before the UUID)
        # Pattern: model_name_uuid where uuid is 8-4-4-4-12 hex format
        match = re.match(r"(.+)_[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", folder.name)
        if match:
            model_name = match.group(1)
        else:
            # If no UUID pattern found, use the folder name as-is
            model_name = folder.name
        
        results_ds_path = folder / "results_ds"
        if results_ds_path.exists():
            try:
                ds = load_from_disk(str(results_ds_path))
                # Extract the test split from the DatasetDict
                if isinstance(ds, dict) and "test" in ds:
                    model_datasets[model_name] = ds["test"]
                else:
                    model_datasets[model_name] = ds
                print(f"Loaded: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    
    return DatasetDict(model_datasets)

# %%
# Load results from results_cef
print("Loading results from results_cef...")
results_cef = load_all_model_results(results_cef_path)
print(f"\nLoaded {len(results_cef)} models from results_cef")
print(f"Models: {list(results_cef.keys())}")

# %%
# Load results from results_di_cef
print("\nLoading results from results_di_cef...")
results_di_cef = load_all_model_results(results_di_cef_path)
print(f"\nLoaded {len(results_di_cef)} models from results_di_cef")
print(f"Models: {list(results_di_cef.keys())}")

# %%
# Model name to model_id mapping
MODEL_ID_MAP = {
    "DeepSeek-V3.1": "deepseek-ai/DeepSeek-V3.1",
    "Llama3-OpenBioLLM-70B": "aaditya/Llama3-OpenBioLLM-70B",
    "MediPhi-Instruct": "microsoft/MediPhi-Instruct",
    "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Qwen3-235B-A22B-Thinking-2507": "Qwen/Qwen3-235B-A22B",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "llama3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "m42-health_Llama3-Med42-70B": "m42-health/Llama3-Med42-70B",
    "m42-health_Llama3-Med42-8B": "m42-health/Llama3-Med42-8B",
    "meta-llama_Llama-4-Maverick-17B-128E-Instruct": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "microsoft_Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "microsoft_phi-4": "microsoft/phi-4",
    "mistralai_Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai_Mistral-Large-3-675B-Instruct-2512": "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "moonshotai_Kimi-K2-Thinking": "moonshotai/Kimi-K2-Thinking",
    "nvidia_Llama-3.1-Nemotron-70B-Instruct-HF": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "qwen3-8B": "Qwen/Qwen3-8B",
}

def calculate_model_scores(dataset_dict: DatasetDict) -> list:
    """
    Calculate average scores for each model and return in the required format.
    Filters out samples with None values and prints stats.
    """
    models_results = []
    
    print("\n" + "="*60)
    print("Sample filtering stats:")
    print("="*60)
    
    for model_name, dataset in dataset_dict.items():
        # Get model_id from mapping, or use model_name as fallback
        model_id = MODEL_ID_MAP.get(model_name, model_name)
        
        # Extract cef_scores from all samples
        cef_scores_list = dataset["cef_scores"]
        total_samples = len(cef_scores_list)
        
        # Filter out samples with None values in any score
        valid_scores = []
        for s in cef_scores_list:
            if (s is not None and 
                s.get("overall_score") is not None and
                s.get("conciseness_score") is not None and
                s.get("conformity_score") is not None and
                s.get("consistency_score") is not None and
                s.get("coverage_score") is not None):
                valid_scores.append(s)
        
        removed_samples = total_samples - len(valid_scores)
        print(f"{model_name}: {len(valid_scores)}/{total_samples} valid samples ({removed_samples} removed, {removed_samples/total_samples*100:.2f}%)")
        
        if len(valid_scores) == 0:
            print(f"  WARNING: No valid samples for {model_name}, skipping...")
            continue
        
        # Calculate averages (multiply by 100 to convert to percentage)
        overall_scores = [s["overall_score"] for s in valid_scores]
        conciseness_scores = [s["conciseness_score"] for s in valid_scores]
        conformity_scores = [s["conformity_score"] for s in valid_scores]
        consistency_scores = [s["consistency_score"] for s in valid_scores]
        coverage_scores = [s["coverage_score"] for s in valid_scores]
        
        model_result = {
            "model_id": model_id,
            "overall_score": np.mean(overall_scores) * 100,
            "extra_scores": [
                {"metric": "brief", "value": np.mean(conciseness_scores) * 100},
                {"metric": "coverage", "value": np.mean(coverage_scores) * 100},
                {"metric": "conform", "value": np.mean(conformity_scores) * 100},
                {"metric": "fact", "value": np.mean(consistency_scores) * 100},
            ]
        }
        models_results.append(model_result)
    
    print("="*60 + "\n")
    return models_results

# %%
# Calculate scores for both tasks
bhc_models = calculate_model_scores(results_cef)
di_models = calculate_model_scores(results_di_cef)

# Create the final JSON structure
output_json = {
    "tasks": [
        {
            "task_id": "bhc",
            "models": bhc_models
        },
        {
            "task_id": "di",
            "models": di_models
        }
    ]
}

# %%
# Save to file
output_path = "/home/yathagata/cef-translation/scripts/dischargeme_cef_results.json"
with open(output_path, "w") as f:
    json.dump(output_json, f, indent=2)

print(f"Saved results to {output_path}")

# %%
# Preview the output
print(json.dumps(output_json, indent=2))

# %%
