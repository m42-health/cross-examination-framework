# %%
"""
Explore and analyze results from the RotoWire Boxscore-to-Summary evaluation.

This notebook loads:
1. Generated summaries from different models
2. CEF evaluation reports
3. Question/Answer datasets from robustness study

And provides consolidated results with recalculated CEF scores.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
from typing import Dict, List, Any, Tuple

# For Jupyter display
try:
    from IPython.display import display
except ImportError:
    display = print

# %%
# Configuration
BASE_PATH = Path("/data/cef/boxscore")

# Models that were used for generation
# Available generation models:
# - worker-3:8892 → Qwen/Qwen3-1.7B (qwen3-1.7b)
# - worker-0:8892 → Qwen/Qwen3-4B (qwen3-4b)
# - worker-9:8892 → meta-llama/Llama-3.2-3B-Instruct (llama3.2-3b)
# - worker-13:8892 → meta-llama/Llama-3.1-8B-Instruct (llama3.1-8b)
# (one model TBD)
generation_models = [
    "qwen3-1.7b",
    "qwen3-4b",
    "llama3.2-3b",
    "llama3.1-8b",
]

# %%
def load_model_results(model_name: str) -> Dict[str, Any]:
    """Load all results for a given model."""
    results = {
        "model": model_name,
        "dataset": None,
        "cef_report": None
    }
    
    # Load generated dataset (or CEF-evaluated dataset)
    cef_path = BASE_PATH / f"boxscore_{model_name}_cef"
    gen_path = BASE_PATH / f"boxscore_{model_name}"
    
    if cef_path.exists():
        try:
            results["dataset"] = load_from_disk(str(cef_path))
            print(f"✓ Loaded CEF-evaluated dataset for {model_name}")
        except Exception as e:
            print(f"✗ Failed to load CEF dataset for {model_name}: {e}")
    elif gen_path.exists():
        try:
            results["dataset"] = load_from_disk(str(gen_path))
            print(f"✓ Loaded generated dataset for {model_name}")
        except Exception as e:
            print(f"✗ Failed to load generated dataset for {model_name}: {e}")
    else:
        print(f"✗ No dataset found for {model_name}")
    
    # Load CEF report if exists
    report_path = cef_path / "evaluation_report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                results["cef_report"] = json.load(f)
            print(f"✓ Loaded CEF report for {model_name}")
        except Exception as e:
            print(f"✗ Failed to load CEF report for {model_name}: {e}")
    
    return results


def load_original_dataset() -> Any:
    """Load the original boxscore dataset with reference summaries."""
    original_path = BASE_PATH / "boxscore_original"
    if original_path.exists():
        try:
            ds = load_from_disk(str(original_path))
            print(f"✓ Loaded original dataset")
            return ds
        except Exception as e:
            print(f"✗ Failed to load original dataset: {e}")
    return None


# %%
# Load all results
print("="*60)
print("Loading datasets...")
print("="*60)

original_ds = load_original_dataset()

consolidated_results = {}
for model in generation_models:
    results = load_model_results(model)
    if results["dataset"] is not None or results["cef_report"] is not None:
        consolidated_results[model] = results

print(f"\nLoaded results for {len(consolidated_results)} models")

# %%
def compute_bootstrap_ci(values: List[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval."""
    values = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(values) == 0:
        return (0.0, 0.0)
    boot_means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n_bootstrap)]
    return np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def recalculate_scores(dataset, original_dataset=None) -> Dict[str, Any]:
    """
    Recalculate CEF scores from cef_details in the dataset.
    
    For Boxscore-to-Summary:
    - Coverage: 100 - %(IDK) from coverage_answers (Q_boxscore|summary)
    - Conformity: 100 - %(NO) from coverage_answers (Q_boxscore|summary)
    - Faithfulness: 100 - %(NO) from faithfulness_answers (Q_summary|boxscore)
    - Consistency: 100 - %(IDK) from consistency_answers (Q_summary|summary)
    - Conciseness: len(summary) / len(boxscore) * 100
    """
    coverage_scores = []
    conformity_scores = []
    faithfulness_scores = []
    consistency_scores = []
    conciseness_scores = []
    
    for example in dataset["test"]:
        cef_details = example.get("cef_details", {})
        src_len = len(example.get("src", ""))
        trg_len = len(example.get("trg", ""))
        
        # Conciseness
        if src_len > 0:
            conciseness_scores.append((trg_len / src_len) * 100)
        else:
            conciseness_scores.append(0)
        
        # Coverage and Conformity: Q from boxscore (src), answered by summary (trg)
        coverage_answers = cef_details.get("coverage_answers", [])
        if coverage_answers:
            idk_count = coverage_answers.count("IDK")
            no_count = coverage_answers.count("NO")
            
            coverage = 100 - (idk_count / len(coverage_answers) * 100)
            conformity = 100 - (no_count / len(coverage_answers) * 100)
            coverage_scores.append(coverage)
            conformity_scores.append(conformity)
        
        # Faithfulness: Q from summary (trg), answered by boxscore (src)
        faithfulness_answers = cef_details.get("faithfulness_answers", [])
        if faithfulness_answers:
            no_count = faithfulness_answers.count("NO")
            faithfulness = 100 - (no_count / len(faithfulness_answers) * 100)
            faithfulness_scores.append(faithfulness)
        
        # Consistency: Q from summary (trg), answered by summary (trg)
        consistency_answers = cef_details.get("consistency_answers", [])
        if consistency_answers:
            idk_count = consistency_answers.count("IDK")
            consistency = 100 - (idk_count / len(consistency_answers) * 100)
            consistency_scores.append(consistency)
    
    results = {}
    
    if coverage_scores:
        mean = np.mean(coverage_scores)
        ci_low, ci_high = compute_bootstrap_ci(coverage_scores)
        results["coverage"] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
    
    if conformity_scores:
        mean = np.mean(conformity_scores)
        ci_low, ci_high = compute_bootstrap_ci(conformity_scores)
        results["conformity"] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
    
    if faithfulness_scores:
        mean = np.mean(faithfulness_scores)
        ci_low, ci_high = compute_bootstrap_ci(faithfulness_scores)
        results["faithfulness"] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
    
    if consistency_scores:
        mean = np.mean(consistency_scores)
        ci_low, ci_high = compute_bootstrap_ci(consistency_scores)
        results["consistency"] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
    
    if conciseness_scores:
        mean = np.mean(conciseness_scores)
        ci_low, ci_high = compute_bootstrap_ci(conciseness_scores)
        results["conciseness"] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
    
    return results


# %%
# Recalculate scores for all models
print("\n" + "="*60)
print("Recalculating CEF Scores")
print("="*60)

recalculated_results = {}
for model, data in consolidated_results.items():
    if data["dataset"] is not None and "test" in data["dataset"]:
        # Check if cef_details exists
        if "cef_details" in data["dataset"]["test"].column_names:
            recalculated_results[model] = recalculate_scores(data["dataset"], original_ds)
            print(f"✓ Recalculated scores for {model}")
        else:
            print(f"✗ No cef_details found for {model}")
    else:
        print(f"✗ No test split for {model}")

# %%
# Display recalculated results
print("\n" + "="*60)
print("RECALCULATED CEF SCORES")
print("="*60)

for model, scores in recalculated_results.items():
    print(f"\n--- {model} ---")
    for metric, values in scores.items():
        print(f"  {metric.capitalize():15s}: {values['mean']:6.2f} (95% CI: [{values['ci_low']:.2f}, {values['ci_high']:.2f}])")

# %%
# Create summary DataFrame
rows = []
for model, scores in recalculated_results.items():
    row = {"Model": model}
    for metric, values in scores.items():
        row[f"{metric.capitalize()}"] = f"{values['mean']:.2f}"
        row[f"{metric.capitalize()} CI"] = f"[{values['ci_low']:.2f}, {values['ci_high']:.2f}]"
    rows.append(row)

df_final = pd.DataFrame(rows)
print("\n" + "="*60)
print("Summary Table")
print("="*60)
display(df_final)

# %%
# Also add scores with CI in compact format
recalculated_with_ci = {}
for model, scores in recalculated_results.items():
    recalculated_with_ci[model] = {}
    for metric, values in scores.items():
        recalculated_with_ci[model][metric] = f"{values['mean']:.2f} [{values['ci_low']:.2f}, {values['ci_high']:.2f}]"

print("\n" + "="*60)
print("Scores with CI (compact format)")
print("="*60)
for model, scores in recalculated_with_ci.items():
    print(f"\n{model}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value}")

# %%
# Final consolidated results dictionary
# Keys: model name
# Values: dict with "dataset" and "scores_with_ci"
print("\n" + "="*60)
print("Consolidated Results Dictionary Structure")
print("="*60)

for model in consolidated_results:
    consolidated_results[model]["recalculated_scores"] = recalculated_results.get(model, {})
    consolidated_results[model]["scores_with_ci"] = recalculated_with_ci.get(model, {})

print("consolidated_results = {")
for model in consolidated_results:
    print(f"  '{model}': {{")
    print(f"    'dataset': <DatasetDict>,")
    print(f"    'cef_report': <dict or None>,")
    print(f"    'recalculated_scores': {consolidated_results[model].get('recalculated_scores', {})}),")
    print(f"    'scores_with_ci': {consolidated_results[model].get('scores_with_ci', {})}),")
    print(f"  }},")
print("}")

# %%
# Interpretation guide for boxscore-to-summary
print("\n" + "="*60)
print("INTERPRETATION GUIDE: Boxscore-to-Summary")
print("="*60)
print("""
CEF Metrics for NBA Game Summary Generation:

1. COVERAGE (100 - %IDK from Q_boxscore→summary)
   - "How much of the boxscore data is mentioned in the summary?"
   - High score = summary covers most important stats and facts
   - Low score = summary misses key information from boxscore

2. CONFORMITY (100 - %NO from Q_boxscore→summary)
   - "Is the summary consistent with the boxscore data?"
   - High score = summary doesn't contradict boxscore stats
   - Low score = summary contains contradictory information

3. FAITHFULNESS (100 - %NO from Q_summary→boxscore)
   - "Are the claims in the summary verifiable from boxscore?"
   - High score = summary is factually grounded
   - Low score = summary contains hallucinated information

4. CONSISTENCY (100 - %IDK from Q_summary→summary)
   - "Is the summary internally coherent?"
   - High score = summary is self-consistent
   - Low score = summary has internal contradictions

5. CONCISENESS (len(summary) / len(boxscore) * 100)
   - "How compact is the summary relative to the boxscore?"
   - Lower score = more concise summary
   - Higher score = more verbose summary

Ideal Model Characteristics:
- High Coverage: Captures key game facts
- High Conformity: Doesn't contradict the data
- High Faithfulness: No hallucinated claims
- High Consistency: Coherent narrative
- Appropriate Conciseness: Good signal-to-noise ratio
""")

# %%
# Example: Access specific model data
if consolidated_results:
    example_model = list(consolidated_results.keys())[0]
    print(f"\nExample: Accessing data for {example_model}")
    print(f"  Dataset: {consolidated_results[example_model]['dataset']}")
    print(f"  Scores: {consolidated_results[example_model].get('scores_with_ci', 'Not calculated')}")
    
    # Show a sample if dataset exists
    if consolidated_results[example_model]['dataset'] is not None:
        ds = consolidated_results[example_model]['dataset']
        if "test" in ds and len(ds["test"]) > 0:
            sample = ds["test"][0]
            print(f"\n  Sample src (boxscore) length: {len(sample.get('src', ''))}")
            print(f"  Sample trg (summary) length: {len(sample.get('trg', ''))}")
            if "cef_details" in sample:
                print(f"  CEF details keys: {list(sample['cef_details'].keys())}")

# %%

