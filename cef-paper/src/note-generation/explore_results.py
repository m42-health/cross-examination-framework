# %%
import os
import json
from datasets import load_from_disk
import pandas as pd
from pathlib import Path
import numpy as np

# %%
# Base path for ACI data
BASE_PATH = "/data/cef/aci"

# %%
# List all datasets in the directory
all_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
print("All directories in /data/cef/aci/:")
for d in sorted(all_dirs):
    print(f"  - {d}")

# %%
# Load all generated note datasets
model_datasets = {}
model_names = ["original", "llama3.1-8b", "llama3.2-3b", "qwen3-1.7b", "qwen3-4b", "gpt-oss-20b"]

for model in model_names:
    ds_path = os.path.join(BASE_PATH, f"aci_{model}")
    if os.path.exists(ds_path):
        try:
            ds = load_from_disk(ds_path)
            model_datasets[model] = ds
            print(f"Loaded {model}: {ds}")
        except Exception as e:
            print(f"Failed to load {model}: {e}")

# %%
# Display sample from each model
print("\n" + "=" * 80)
print("SAMPLE CLINICAL NOTES FROM EACH MODEL")
print("=" * 80)

for model, ds in model_datasets.items():
    print(f"\n--- {model} ---")
    sample = ds["test"][0]
    print(f"ID: {sample['id']}")
    print(f"Conversation (first 500 chars): {sample['src'][:500]}...")
    print(f"Clinical Note (first 800 chars): {sample['trg'][:800]}...")
    print()

# %%
# Load all CEF results
results_cef_path = os.path.join(BASE_PATH, "results_cef")
cef_results = {}

if os.path.exists(results_cef_path):
    result_dirs = [d for d in os.listdir(results_cef_path) if os.path.isdir(os.path.join(results_cef_path, d))]
    print(f"\nFound {len(result_dirs)} result directories in results_cef/")
    
    for result_dir in result_dirs:
        result_path = os.path.join(results_cef_path, result_dir)
        
        # Try to load the report
        report_path = os.path.join(result_path, "cef_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            cef_results[result_dir] = {
                "report": report,
                "path": result_path
            }
            print(f"  - {result_dir}: Loaded CEF report")
        
        # Try to load the dataset
        ds_path = os.path.join(result_path, "results_ds")
        if os.path.exists(ds_path):
            try:
                ds = load_from_disk(ds_path)
                if result_dir in cef_results:
                    cef_results[result_dir]["dataset"] = ds
                else:
                    cef_results[result_dir] = {"dataset": ds, "path": result_path}
                print(f"  - {result_dir}: Loaded results dataset")
            except Exception as e:
                print(f"  - {result_dir}: Failed to load dataset: {e}")

# %%
# Display CEF results summary
print("\n" + "=" * 80)
print("CEF EVALUATION RESULTS SUMMARY")
print("=" * 80)

results_data = []
for result_id, result in cef_results.items():
    if "report" in result:
        report = result["report"]
        data_path = report.get("cef_params", {}).get("data_path", "Unknown")
        
        # Extract model name from data path
        model_name = data_path.split("_")[-1] if "_" in data_path else data_path
        
        for split, scores in report.get("scores", {}).items():
            row = {
                "result_id": result_id[:8] + "...",
                "model": model_name,
                "split": split,
            }
            
            if "coverage_score" in scores:
                row["coverage"] = scores["coverage_score"].get("mean", -1)
            if "conformity_score" in scores:
                row["conformity"] = scores["conformity_score"].get("mean", -1)
            if "consistency_score" in scores:
                row["consistency"] = scores["consistency_score"].get("mean", -1)
            if "conciseness" in scores:
                row["conciseness"] = scores["conciseness"].get("mean", -1)
            
            results_data.append(row)

if results_data:
    df_results = pd.DataFrame(results_data)
    print("\nCEF Scores by Model:")
    display(df_results)
else:
    print("No CEF results found")

# %%
# Create consolidated results dictionary
print("\n" + "=" * 80)
print("CONSOLIDATED RESULTS BY MODEL")
print("=" * 80)

consolidated_results = {}

for result_id, result in cef_results.items():
    if "report" not in result:
        continue
    
    report = result["report"]
    data_path = report.get("cef_params", {}).get("data_path", "")
    
    # Extract model name from data path
    model_name = data_path.split("_")[-1] if "_" in data_path else os.path.basename(data_path)
    
    # Skip if model already exists
    if model_name in consolidated_results:
        continue
    
    # Build scores dict with confidence intervals
    scores_with_ci = {}
    for split, scores in report.get("scores", {}).items():
        scores_with_ci[split] = {}
        
        for score_name in ["coverage_score", "conformity_score", "consistency_score"]:
            if score_name in scores:
                score_data = scores[score_name]
                scores_with_ci[split][score_name] = {
                    "mean": score_data.get("mean"),
                    "ci": score_data.get("confidence_interval"),
                    "n": score_data.get("sample_size")
                }
        
        if "conciseness" in scores:
            scores_with_ci[split]["conciseness"] = {
                "mean": scores["conciseness"].get("mean"),
                "std": scores["conciseness"].get("std"),
                "n": scores["conciseness"].get("sample_size")
            }
    
    consolidated_results[model_name] = {
        "dataset": result.get("dataset"),
        "scores": scores_with_ci,
        "result_id": result_id,
        "path": result.get("path")
    }

# Display the consolidated results
print(f"\nModels with CEF results: {list(consolidated_results.keys())}")

for model_name, data in consolidated_results.items():
    print(f"\n--- {model_name} ---")
    print(f"  Result ID: {data['result_id']}")
    print(f"  Path: {data['path']}")
    print(f"  Dataset: {data['dataset']}")

# %%
# Recalculate scores based on correct definitions
print("\n" + "=" * 80)
print("RECALCULATED SCORES (Correct Definitions)")
print("=" * 80)
print("""
Definitions for Clinical Note Generation:
- Coverage: 100 - %(Q_conversation|note == "IDK") 
  → Questions from conversation, answered using note. Measures info retention.
  
- Conformity: 100 - %(Q_conversation|note == "NO")
  → Questions from conversation, answered using note. Measures contradictions.
  
- Consistency: 100 - %(Q_note|conversation == "IDK")
  → Questions from note, answered using conversation. Measures hallucination.

- Conciseness: len(note) / len(conversation) * 100
  → Ratio of note length to conversation length.
""")

def recalculate_scores(dataset, split="test"):
    """Recalculate CEF scores based on correct definitions."""
    coverage_scores = []
    conformity_scores = []
    consistency_scores = []
    conciseness_scores = []
    
    for example in dataset[split]:
        cef_details = example.get("cef_details", {})
        
        # Q_src|trg answers (coverage and conformity)
        q_src_trg_answers = cef_details.get("coverage_answers", [])
        # Q_trg|src answers (consistency)
        q_trg_src_answers = cef_details.get("conformity_answers", [])
        
        # Coverage: 100 - %(IDK) from Q_src|trg
        if q_src_trg_answers:
            idk_count = sum(1 for a in q_src_trg_answers if a == "IDK")
            coverage = 100 - (idk_count / len(q_src_trg_answers) * 100)
            coverage_scores.append(coverage)
        
        # Conformity: 100 - %(NO) from Q_src|trg
        if q_src_trg_answers:
            no_count = sum(1 for a in q_src_trg_answers if a == "NO")
            conformity = 100 - (no_count / len(q_src_trg_answers) * 100)
            conformity_scores.append(conformity)
        
        # Consistency: 100 - %(IDK) from Q_trg|src
        if q_trg_src_answers:
            idk_count = sum(1 for a in q_trg_src_answers if a == "IDK")
            consistency = 100 - (idk_count / len(q_trg_src_answers) * 100)
            consistency_scores.append(consistency)
        
        # Conciseness
        src = example.get("src", "")
        trg = example.get("trg", "")
        if src and trg and len(src) > 0:
            conciseness = (len(trg) / len(src)) * 100
            conciseness_scores.append(conciseness)
    
    return {
        "coverage": {
            "mean": round(sum(coverage_scores) / len(coverage_scores), 2) if coverage_scores else None,
            "scores": coverage_scores
        },
        "conformity": {
            "mean": round(sum(conformity_scores) / len(conformity_scores), 2) if conformity_scores else None,
            "scores": conformity_scores
        },
        "consistency": {
            "mean": round(sum(consistency_scores) / len(consistency_scores), 2) if consistency_scores else None,
            "scores": consistency_scores
        },
        "conciseness": {
            "mean": round(sum(conciseness_scores) / len(conciseness_scores), 2) if conciseness_scores else None,
            "scores": conciseness_scores
        }
    }

# Recalculate for all models
recalculated_results = {}

for model_name, data in consolidated_results.items():
    if data["dataset"] is None:
        print(f"Skipping {model_name}: No dataset")
        continue
    
    scores = recalculate_scores(data["dataset"])
    recalculated_results[model_name] = scores
    
    print(f"\n--- {model_name} ---")
    print(f"  Coverage:    {scores['coverage']['mean']}%")
    print(f"  Conformity:  {scores['conformity']['mean']}%")
    print(f"  Consistency: {scores['consistency']['mean']}%")
    print(f"  Conciseness: {scores['conciseness']['mean']}%")

# %%
# Bootstrap confidence intervals
def bootstrap_ci(scores, n_iterations=1000, confidence=0.95):
    """Calculate bootstrap confidence interval."""
    if not scores:
        return None, None
    
    scores = np.array(scores)
    bootstrap_means = []
    
    for _ in range(n_iterations):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    sorted_means = np.sort(bootstrap_means)
    lower_idx = int((1 - confidence) / 2 * len(sorted_means))
    upper_idx = int((1 + confidence) / 2 * len(sorted_means))
    
    mean = np.mean(scores)
    ci_lower = sorted_means[lower_idx]
    ci_upper = sorted_means[upper_idx]
    
    return mean, f"+{round(ci_upper - mean, 2)}/-{round(mean - ci_lower, 2)}"

recalculated_with_ci = {}
for model_name, scores in recalculated_results.items():
    recalculated_with_ci[model_name] = {}
    for score_name in ["coverage", "conformity", "consistency", "conciseness"]:
        mean, ci = bootstrap_ci(scores[score_name]["scores"])
        recalculated_with_ci[model_name][score_name] = {
            "mean": round(mean, 2) if mean else None,
            "ci": ci,
            "n": len(scores[score_name]["scores"])
        }

print("\n" + "=" * 80)
print("RECALCULATED SCORES WITH 95% CI")
print("=" * 80)

for model_name, scores in recalculated_with_ci.items():
    print(f"\n--- {model_name} ---")
    for score_name, score_data in scores.items():
        print(f"  {score_name}: {score_data['mean']}% ({score_data['ci']}), n={score_data['n']}")

# %%
# Final comparison table
final_comparison_data = []
for model_name, scores in recalculated_with_ci.items():
    row = {"model": model_name}
    for score_name, score_data in scores.items():
        row[score_name] = f"{score_data['mean']} ({score_data['ci']})"
        row[f"{score_name}_mean"] = score_data['mean']
    final_comparison_data.append(row)

if final_comparison_data:
    df_final = pd.DataFrame(final_comparison_data)
    df_final = df_final.sort_values("model")
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE (Recalculated with CI)")
    print("=" * 80)
    display(df_final[["model", "coverage", "conformity", "consistency", "conciseness"]])

# %%
# Show sample with detailed CEF information
print("\n" + "=" * 80)
print("SAMPLE WITH DETAILED CEF INFORMATION")
print("=" * 80)

if consolidated_results:
    # Get first model with dataset
    for model_name, data in consolidated_results.items():
        if data["dataset"] is not None:
            ds = data["dataset"]
            sample = ds["test"][0]
            
            print(f"\nModel: {model_name}")
            print(f"ID: {sample.get('id', 'N/A')}")
            print(f"\nConversation (first 500 chars):")
            print(sample["src"][:500] + "...")
            print(f"\nGenerated Note (first 800 chars):")
            print(sample["trg"][:800] + "...")
            
            cef_details = sample.get("cef_details", {})
            
            print(f"\nQuestions from Conversation ({len(cef_details.get('questions_from_src', []))} questions):")
            for i, q in enumerate(cef_details.get("questions_from_src", [])[:3]):
                ans = cef_details.get("coverage_answers", [])[i] if i < len(cef_details.get("coverage_answers", [])) else "N/A"
                print(f"  {i+1}. {q['question']} → {ans}")
            
            print(f"\nQuestions from Note ({len(cef_details.get('questions_from_trg', []))} questions):")
            for i, q in enumerate(cef_details.get("questions_from_trg", [])[:3]):
                ans = cef_details.get("conformity_answers", [])[i] if i < len(cef_details.get("conformity_answers", [])) else "N/A"
                print(f"  {i+1}. {q['question']} → {ans}")
            
            break

# %%
print("\n✓ All results loaded!")
print(f"  - consolidated_results: {list(consolidated_results.keys())}")
print(f"  - recalculated_results: {list(recalculated_results.keys())}")
print(f"  - recalculated_with_ci: {list(recalculated_with_ci.keys())}")

# %%

