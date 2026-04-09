# %%
import os
import json
from datasets import load_from_disk
import pandas as pd
from pathlib import Path

# %%
# Base path for CNN/DM data
BASE_PATH = "/data/cef/cnndm"

# %%
# List all datasets in the directory
all_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
print("All directories in /data/cef/cnndm/:")
for d in sorted(all_dirs):
    print(f"  - {d}")

# %%
# Load all generated summary datasets
model_datasets = {}
model_names = ["original", "llama3.1-8b", "llama3.2-3b", "qwen3-1.7b", "qwen3-4b", "gpt-oss-20b"]

for model in model_names:
    ds_path = os.path.join(BASE_PATH, f"cnndm_{model}")
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
print("SAMPLE SUMMARIES FROM EACH MODEL")
print("=" * 80)

for model, ds in model_datasets.items():
    print(f"\n--- {model} ---")
    sample = ds["test"][0]
    print(f"ID: {sample['id']}")
    print(f"Source (first 300 chars): {sample['src'][:300]}...")
    print(f"Summary: {sample['trg'][:500]}...")
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
            
            results_data.append(row)

if results_data:
    df_results = pd.DataFrame(results_data)
    print("\nCEF Scores by Model:")
    display(df_results)
else:
    print("No CEF results found")

# %%
# Load and explore a specific results dataset
print("\n" + "=" * 80)
print("DETAILED RESULTS FROM LATEST EVALUATION")
print("=" * 80)

# Get the most recent result (by directory name, assuming UUIDs are somewhat time-ordered)
if cef_results:
    # Try to find one with a dataset
    results_with_ds = {k: v for k, v in cef_results.items() if "dataset" in v}
    
    if results_with_ds:
        latest_result_id = list(results_with_ds.keys())[-1]
        latest_result = results_with_ds[latest_result_id]
        
        print(f"\nResult ID: {latest_result_id}")
        print(f"Path: {latest_result['path']}")
        
        if "report" in latest_result:
            print(f"\nReport:")
            print(json.dumps(latest_result["report"], indent=2))
        
        if "dataset" in latest_result:
            ds = latest_result["dataset"]
            print(f"\nDataset: {ds}")
            
            # Show sample with CEF scores
            print("\nSample with CEF scores:")
            sample = ds["test"][0]
            for key, value in sample.items():
                if key in ["src", "trg"]:
                    print(f"  {key}: {str(value)[:200]}...")
                elif key == "cef_details":
                    print(f"  {key}:")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, list) and len(v) > 0:
                                print(f"    {k}: {v[:2]}... ({len(v)} items)")
                            else:
                                print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

# %%
# Load question datasets
print("\n" + "=" * 80)
print("QUESTION GENERATION DATASETS")
print("=" * 80)

question_datasets = {}
judge_models = [
    "meta-llama+++Llama-3.3-70B-Instruct",
    "Qwen+++Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai+++DeepSeek-V3",
    "openai+++gpt-oss-120b"
]

for judge in judge_models:
    ds_path = os.path.join(BASE_PATH, f"cnndm_questions_{judge}")
    if os.path.exists(ds_path):
        try:
            ds = load_from_disk(ds_path)
            question_datasets[judge] = ds
            print(f"Loaded questions from {judge}: {ds}")
        except Exception as e:
            print(f"Failed to load questions from {judge}: {e}")

# %%
# Display sample questions
if question_datasets:
    print("\n" + "=" * 80)
    print("SAMPLE QUESTIONS FROM EACH JUDGE MODEL")
    print("=" * 80)
    
    for judge, ds in question_datasets.items():
        print(f"\n--- {judge} ---")
        sample = ds["test"][0]
        
        if "question_list_src" in sample and sample["question_list_src"]:
            print("Questions from article (src):")
            for i, q in enumerate(sample["question_list_src"][:3]):
                print(f"  {i+1}. {q['question']}")
        
        if "question_list_trg" in sample and sample["question_list_trg"]:
            print("Questions from summary (trg):")
            for i, q in enumerate(sample["question_list_trg"][:3]):
                print(f"  {i+1}. {q['question']}")

# %%
# Load answer datasets
print("\n" + "=" * 80)
print("ANSWER DATASETS")
print("=" * 80)

answer_datasets = {}
for judge in judge_models:
    ds_path = os.path.join(BASE_PATH, f"cnndm_answers_{judge}")
    if os.path.exists(ds_path):
        try:
            ds = load_from_disk(ds_path)
            answer_datasets[judge] = ds
            print(f"Loaded answers from {judge}: {ds}")
            print(f"  Columns: {ds['test'].column_names}")
        except Exception as e:
            print(f"Failed to load answers from {judge}: {e}")

# %%
# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Model datasets loaded: {len(model_datasets)}")
print(f"CEF result directories: {len(cef_results)}")
print(f"Question datasets loaded: {len(question_datasets)}")
print(f"Answer datasets loaded: {len(answer_datasets)}")

# %%
# Create consolidated results dictionary
# Keys: model name, Values: {"dataset": results_ds, "scores": {coverage, conformity, consistency with CI}}

print("\n" + "=" * 80)
print("CONSOLIDATED RESULTS BY MODEL")
print("=" * 80)

consolidated_results = {}

for result_id, result in cef_results.items():
    if "report" not in result:
        continue
    
    report = result["report"]
    data_path = report.get("cef_params", {}).get("data_path", "")
    
    # Extract model name from data path (e.g., /data/cef/cnndm/cnndm_llama3.1-8b -> llama3.1-8b)
    model_name = data_path.split("_")[-1] if "_" in data_path else os.path.basename(data_path)
    
    # Skip if model already exists (keep the first/latest one)
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
    print(f"  Scores:")
    for split, scores in data["scores"].items():
        print(f"    {split}:")
        for score_name, score_data in scores.items():
            print(f"      {score_name}: {score_data['mean']} ({score_data['ci']}), n={score_data['n']}")

# %%
# Create a comparison DataFrame
comparison_data = []
for model_name, data in consolidated_results.items():
    for split, scores in data["scores"].items():
        row = {"model": model_name, "split": split}
        for score_name, score_data in scores.items():
            short_name = score_name.replace("_score", "")
            row[short_name] = f"{score_data['mean']} ({score_data['ci']})"
            row[f"{short_name}_mean"] = score_data['mean']
        comparison_data.append(row)

if comparison_data:
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    display(df_comparison[["model", "split", "coverage", "conformity", "consistency"]])

# %%
# The final consolidated_results dict is ready for use
# Structure:
# consolidated_results = {
#     "model_name": {
#         "dataset": Dataset,  # The results dataset with cef_scores
#         "scores": {
#             "test": {
#                 "coverage_score": {"mean": 85.5, "ci": "+1.2/-1.1", "n": 1000},
#                 "conformity_score": {"mean": 92.3, "ci": "+0.8/-0.9", "n": 1000},
#                 "consistency_score": {"mean": 95.1, "ci": "+0.5/-0.6", "n": 1000}
#             }
#         },
#         "result_id": "uuid...",
#         "path": "/data/cef/cnndm/results_cef/uuid..."
#     },
#     ...
# }

print("\n✓ consolidated_results dict is ready!")
print(f"  Access with: consolidated_results['model_name']['dataset'] or consolidated_results['model_name']['scores']")

# %%
# Recalculate scores based on correct definitions
# Coverage: 100 - %(Q_src|trg == "IDK") - Questions from src, answered using trg, count IDK
# Conformity: 100 - %(Q_src|trg == "NO") - Questions from src, answered using trg, count NO  
# Consistency: 100 - %(Q_trg|src == "IDK") - Questions from trg, answered using src, count IDK

print("\n" + "=" * 80)
print("RECALCULATED SCORES (Correct Definitions)")
print("=" * 80)
print("""
Definitions:
- Coverage: 100 - %(Q_src|trg == "IDK") 
  → Questions from article, answered using summary. Measures info retention.
  
- Conformity: 100 - %(Q_src|trg == "NO")
  → Questions from article, answered using summary. Measures contradictions.
  
- Consistency: 100 - %(Q_trg|src == "IDK")
  → Questions from summary, answered using article. Measures hallucination.

- Conciseness: len(summary) / len(article) * 100
  → Ratio of summary length to article length. Lower = more concise.
""")

def recalculate_scores(dataset, split="test"):
    """
    Recalculate CEF scores based on correct definitions.
    
    In cef_details:
    - coverage_answers: Q_src answered using trg (for Coverage and Conformity)
    - conformity_answers: Q_trg answered using src (for Consistency - measures hallucination)
    """
    coverage_scores = []
    conformity_scores = []
    consistency_scores = []
    conciseness_scores = []
    
    for example in dataset[split]:
        cef_details = example.get("cef_details", {})
        
        # Q_src|trg answers (used for Coverage and Conformity)
        q_src_trg_answers = cef_details.get("coverage_answers", [])
        
        # Q_trg|src answers (used for Consistency)
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
        
        # Conciseness: len(summary) / len(article) * 100
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
# Create comparison DataFrame with recalculated scores
recalc_comparison_data = []
for model_name, scores in recalculated_results.items():
    row = {
        "model": model_name,
        "coverage": scores["coverage"]["mean"],
        "conformity": scores["conformity"]["mean"],
        "consistency": scores["consistency"]["mean"],
        "conciseness": scores["conciseness"]["mean"]
    }
    recalc_comparison_data.append(row)

if recalc_comparison_data:
    df_recalc = pd.DataFrame(recalc_comparison_data)
    df_recalc = df_recalc.sort_values("model")
    print("\n" + "=" * 80)
    print("RECALCULATED SCORES COMPARISON TABLE")
    print("=" * 80)
    display(df_recalc)

# %%
# Calculate bootstrap confidence intervals for recalculated scores
import numpy as np

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
# Final comparison table with CI
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
