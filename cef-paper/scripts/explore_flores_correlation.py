# %%
from datasets import load_from_disk
import json
from datasets import Dataset, DatasetDict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
import os
from datetime import datetime

def load_reports_from_dates(results_dir, dates, params_column = "cef_params"):
    """
    Load reports from specified dates in the results directory.
    
    Args:
        results_dir (str): Path to the results directory
        dates (list): List of date strings in format "Aug 22", "Aug 25", etc.
    
    Returns:
        tuple: (cef_params_rows, reports) where:
            - cef_params_rows: List of dictionaries with flattened cef_params
            - reports: Dictionary mapping folder paths to report data
    """
    print(f"\n{'='*40}\nFolders from {dates} in {results_dir}:\n{'='*40}")

    # Get all folders and their stats
    folders = []
    for item in os.listdir(results_dir):
        full_path = os.path.join(results_dir, item)
        if os.path.isdir(full_path):  # Check if it's a directory
            stats = os.stat(full_path)
            # Convert timestamp to readable format
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%b %d")
            if mod_time in dates:
                folders.append((full_path, stats))

    # Print folders sorted by modification time

    # Read all reports from today's folders
    reports = {}
    cef_params_rows = []
    for folder, stats in sorted(folders, key=lambda x: x[1].st_mtime):
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{mod_time}  {os.path.basename(folder)}/")
        report_path = os.path.join(folder, "report.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
                reports[folder] = report
                params = report[params_column]
                if params_column == "cef_params":
                    cef_params_rows.append({
                        'Folder': folder,
                        "Mod Time": mod_time,
                        'Data Path': params['data_path'],
                        'Judge Model': params['judge_model'],
                        'Num Questions': params['num_questions_to_generate'],
                        'Token per Question': params['token_per_question'],
                        'Bootstrap Iterations': params['bootstrap_iterations'],
                        'Bootstrap Fraction': params['bootstrap_fraction']
                    })
                else:
                    cef_params_rows.append({
                        "Folder": folder,
                        "Mod Time": mod_time,
                        "Data Path": params["data_path"],
                    })
        else:
            print(f"Warning: No report.json found in {folder}")

    print(f"\nLoaded {len(reports)} reports from specified dates")
    return cef_params_rows, reports

# Usage
# %%
results_dir = "/data/cef/ntrex/results_cef_new"
today = ["Aug 22", "Aug 25", "Aug 29", "Sep 01", "Sep 11"]  # Hardcoded date for July 25th

cef_params_rows, reports = load_reports_from_dates(results_dir, today)
params_df = pd.DataFrame(cef_params_rows)

target_models = ["qwen3-1.7b", "gemma3-4b", "gpt-oss-20b", "nllb-3.3B", "azure"]
print("Filtering reports where num_questions = 10")
filtered_reports = []
filtered_datasets = []
for i, params in enumerate(cef_params_rows):
    model_name = params["Data Path"].split("ntrex_")[-1] if "ntrex_" in params["Data Path"] else ""
    if params['Num Questions'] == 10 and model_name in target_models and "debug" not in params["Folder"]:
        ds = load_from_disk(params["Folder"] + "/results_ds")
        filtered_datasets.append(ds)
        filtered_reports.append(reports[params["Folder"]])
        
cef_reports = filtered_reports
cef_datasets = filtered_datasets

print("len(cef_reports):", len(cef_reports))


# %%
results_dir = "/data/cef/ntrex/results_cef_new_with_reference"
today = ["Aug 22", "Aug 25", "Aug 29", "Sep 01", "Sep 11", "Sep 22"]  # Hardcoded date for July 25th

cef_params_rows, reports = load_reports_from_dates(results_dir, today)
params_df = pd.DataFrame(cef_params_rows)

target_models = ["qwen3-1.7b", "gemma3-4b", "gpt-oss-20b", "nllb-3.3B", "azure"]
print("Filtering reports where num_questions = 10")
filtered_reports = []
filtered_datasets = []
for i, params in enumerate(cef_params_rows):
    model_name = params["Data Path"].split("ntrex_")[-1] if "ntrex_" in params["Data Path"] else ""
    if params['Num Questions'] == 10 and model_name in target_models and "debug" not in params["Folder"]:
        ds = load_from_disk(params["Folder"] + "/results_ds")
        filtered_datasets.append(ds)
        filtered_reports.append(reports[params["Folder"]])
        
cef_ref_reports = filtered_reports
cef_ref_datasets = filtered_datasets

print("len(cef_ref_reports):", len(cef_ref_reports))


# %%
results_dir = "/data/cef/ntrex/results_trad_new"
today = ["Aug 22", "Aug 25", "Aug 29", "Sep 01", "Sep 11", "Sep 22"]  # Hardcoded date for July 25th

cef_params_rows, reports = load_reports_from_dates(results_dir, today, params_column = "traditional_params")
params_df = pd.DataFrame(cef_params_rows)

target_models = ["qwen3-1.7b", "gemma3-4b", "gpt-oss-20b", "nllb-3.3B", "azure", "google"]
print("Filtering reports where num_questions = 10")
filtered_reports = []
filtered_datasets = []
for i, params in enumerate(cef_params_rows):
    model_name = params["Data Path"].split("ntrex_")[-1] if "ntrex_" in params["Data Path"] else ""
    if model_name in target_models and "debug" not in params["Folder"]:
        ds = load_from_disk(params["Folder"] + "/results_ds")
        filtered_datasets.append(ds)
        filtered_reports.append(reports[params["Folder"]])
        
trad_reports = filtered_reports
trad_datasets = filtered_datasets

print("len(trad_reports):", len(trad_reports))

# %%
# Create a comprehensive dataframe from the reports
rows = []

# Process CEF reference reports
for report in cef_ref_reports:
    data_path = report['cef_params']['data_path']
    scores_dict = report['scores']
    
    for lang_pair, scores in scores_dict.items():
        if scores:  # Only process non-empty score dictionaries
            row = {
                'data_path': data_path,
                'language_pair': lang_pair,
                'conformity_score': scores.get('conformity_score', {}).get('mean', None),
                'consistency_score': scores.get('consistency_score', {}).get('mean', None),
                'coverage_score': scores.get('coverage_score', {}).get('mean', None),
                'bleu_score': None,
                'chrf_score': None,
                'bertscore_score': None,
                'comet_score': None
            }
            rows.append(row)

# Process traditional reports and merge with CEF data
for report in trad_reports:
    data_path = report['traditional_params']['data_path']
    scores_dict = report['scores']
    
    for lang_pair, scores in scores_dict.items():
        if scores:  # Only process non-empty score dictionaries
            # Find existing row with same data_path and language_pair
            existing_row = None
            for row in rows:
                if row['data_path'] == data_path and row['language_pair'] == lang_pair:
                    existing_row = row
                    break
            
            if existing_row:
                # Update existing row with traditional metrics
                existing_row['bleu_score'] = scores.get('bleu_score', {}).get('mean', None)
                existing_row['chrf_score'] = scores.get('chrf_score', {}).get('mean', None)
                existing_row['bertscore_score'] = scores.get('bertscore_score', {}).get('mean', None)
                existing_row['comet_score'] = scores.get('comet_score', {}).get('mean', None)
            else:
                # Create new row if no CEF data exists
                row = {
                    'data_path': data_path,
                    'language_pair': lang_pair,
                    'conformity_score': None,
                    'consistency_score': None,
                    'coverage_score': None,
                    'bleu_score': scores.get('bleu_score', {}).get('mean', None),
                    'chrf_score': scores.get('chrf_score', {}).get('mean', None),
                    'bertscore_score': scores.get('bertscore_score', {}).get('mean', None),
                    'comet_score': scores.get('comet_score', {}).get('mean', None)
                }
                rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)
# Extract model name from data_path and create new Model_Name column
df['Model_Name'] = df['data_path'].str.replace('/data/cef/ntrex/ntrex_', '', regex=False)

# Drop the original data_path column
df = df.drop('data_path', axis=1)

# Reorder columns to put Model_Name first
columns = ['Model_Name'] + [col for col in df.columns if col != 'Model_Name']
df = df[columns]

# Sort DataFrame by Model_Name and language_pair
df = df.sort_values(['Model_Name', 'language_pair']).reset_index(drop=True)

# Filter out rows with all None values for scores
df = df.dropna(subset=['conformity_score', 'consistency_score', 'coverage_score', 
                       'bleu_score', 'chrf_score', 'bertscore_score', 'comet_score'], 
               how='any')

display(df)

# %%



# %%
latex_table = df.to_latex(
    index=False,
    float_format="%.3f",
    na_rep="--",
    caption="Correlation between CEF scores and traditional metrics across models and language pairs",
    label="tab:cef_traditional_correlation",
    column_format="l" + "c" * (len(df.columns) - 1),
    escape=False
)

print("LaTeX Table:")
print(latex_table)

# %%
cef_dataset = DatasetDict()
cef_ref_dataset = DatasetDict()
trad_dataset = DatasetDict()
model_name = "qwen3-1.7b"
for dataset, report in zip(cef_datasets, cef_reports):
    if report['cef_params']["data_path"] == f"/data/cef/ntrex/ntrex_{model_name}":
        for split in dataset:
            if "cef_scores" in dataset[split].features and "cef_details" in dataset[split].features and len(dataset[split]) == 123:
                cef_dataset[split] = dataset[split]
                
for dataset, report in zip(cef_ref_datasets, cef_ref_reports):
    if report['cef_params']["data_path"] == f"/data/cef/ntrex/ntrex_{model_name}":
        for split in dataset:
            print(split)
            print(dataset[split])
            if "cef_scores" in dataset[split].features and "cef_details" in dataset[split].features and len(dataset[split]) == 123:
                cef_ref_dataset[split] = dataset[split]
                
for dataset, report in zip(trad_datasets, trad_reports):
    if report['traditional_params']["data_path"] == f"/data/cef/ntrex/ntrex_{model_name}":
        for split in dataset:
            trad_dataset[split] = dataset[split]

# %%
# Combine all datasets into a single dataset, mapping by id
combined_dataset = DatasetDict()

# Get all language pairs that exist across datasets
all_lang_pairs = set()
all_lang_pairs.update(cef_dataset.keys())
all_lang_pairs.update(cef_ref_dataset.keys())
all_lang_pairs.update(trad_dataset.keys())

for lang_pair in all_lang_pairs:
    # Get datasets for this language pair if they exist
    cef_ds = cef_dataset.get(lang_pair)
    cef_ref_ds = cef_ref_dataset.get(lang_pair)
    trad_ds = trad_dataset.get(lang_pair)
    
    # Create mapping dictionaries by id for each dataset
    cef_data = {}
    cef_ref_data = {}
    trad_data = {}
    
    if cef_ds is not None:
        for row in cef_ds:
            cef_data[row['id']] = {
                'cef_scores': row['cef_scores'],
                'cef_details': row['cef_details']
            }
    
    if cef_ref_ds is not None:
        for row in cef_ref_ds:
            cef_ref_data[row['id']] = {
                'cef_ref_scores': row['cef_scores'],
                'cef_ref_details': row['cef_details']
            }
    
    if trad_ds is not None:
        for row in trad_ds:
            trad_data[row['id']] = {
                'bleu': row['bleu'],
                'chrf': row['chrf'],
                'bertscore': row['bertscore'],
                'comet': row['comet'],
                'reference_translation': row['reference_translation']
            }
    
    # Get all unique ids across all datasets for this language pair
    all_ids = set()
    all_ids.update(cef_data.keys())
    all_ids.update(cef_ref_data.keys())
    all_ids.update(trad_data.keys())
    
    # Build combined rows
    combined_rows = []
    for id_val in sorted(all_ids):
        # Start with basic info from any available dataset
        row = {'id': id_val}
        
        # Add src and trg from any available dataset
        if cef_ds is not None and id_val in [r['id'] for r in cef_ds]:
            cef_row = next(r for r in cef_ds if r['id'] == id_val)
            row.update({'src': cef_row['src'], 'trg': cef_row['trg']})
        elif cef_ref_ds is not None and id_val in [r['id'] for r in cef_ref_ds]:
            cef_ref_row = next(r for r in cef_ref_ds if r['id'] == id_val)
            row.update({'src': cef_ref_row['src'], 'trg': cef_ref_row['trg']})
        elif trad_ds is not None and id_val in [r['id'] for r in trad_ds]:
            trad_row = next(r for r in trad_ds if r['id'] == id_val)
            row.update({'src': trad_row['src'], 'trg': trad_row['trg']})
        
        # Add CEF data if available
        if id_val in cef_data:
            row.update(cef_data[id_val])
        else:
            row.update({'cef_scores': None, 'cef_details': None})
        
        # Add CEF reference data if available
        if id_val in cef_ref_data:
            row.update(cef_ref_data[id_val])
        else:
            row.update({'cef_ref_scores': None, 'cef_ref_details': None})
        
        # Add traditional metrics data if available
        if id_val in trad_data:
            row.update(trad_data[id_val])
        else:
            row.update({'bleu': None, 'chrf': None, 'bertscore': None, 'comet': None, 'reference_translation': None})
        
        combined_rows.append(row)
    
    # Create dataset for this language pair
    if combined_rows:
        from datasets import Dataset
        combined_dataset[lang_pair] = Dataset.from_list(combined_rows)

print("Combined dataset:")
print(combined_dataset)
#%%


# %%
# Combine all language pairs into a single dataset
all_rows = []
for lang_pair, dataset in combined_dataset.items():
    for row in dataset:
        # Add language pair information to each row
        row_with_lang = dict(row)
        row_with_lang['lang_pair'] = lang_pair
        all_rows.append(row_with_lang)

# Create a single combined dataset
from datasets import Dataset
single_combined_dataset = Dataset.from_list(all_rows)

print("Single combined dataset:")
print(single_combined_dataset)

# %%
# Correlation analysis
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Extract data for correlation analysis
correlation_data = []
for row in single_combined_dataset:
    if row['cef_scores'] is not None and row['cef_ref_scores'] is not None:
        # Extract CEF scores (excluding conciseness)
        cef_scores = {k: v for k, v in row['cef_scores'].items() if k != 'conciseness_score'}
        cef_ref_scores = {k: v for k, v in row['cef_ref_scores'].items() if k != 'conciseness_score'}
        
        # Extract traditional metrics
        trad_metrics = {
            'bleu': row['bleu'],
            'chrf': row['chrf'],
            'bertscore': row['bertscore'],
            'comet': row['comet']
        }
        
        # Only include rows with all data available
        if all(v is not None for v in trad_metrics.values()):
            correlation_data.append({
                **{f'cef_{k}': v for k, v in cef_scores.items()},
                **{f'cef_ref_{k}': v for k, v in cef_ref_scores.items()},
                **trad_metrics,
                'lang_pair': row['lang_pair'],
                'src': row['src'],
                'trg': row['trg'],
                'reference_translation': row['reference_translation'],
                'cef_details': row['cef_details'],
                'cef_ref_details': row['cef_ref_details']
            })

# Convert to DataFrame
df = pd.DataFrame(correlation_data)

# %%
# Remove rows where any CEF score is 0 or negative
print(f"Original dataset size: {len(df)}")

# Get all CEF columns (both cef_ and cef_ref_ prefixed)
cef_cols = [col for col in df.columns if col.startswith('cef_') and "details" not in col]

# Create mask for rows where all CEF scores are positive
positive_mask = (df[cef_cols] > 0).all(axis=1)

# Filter the dataframe
df = df[positive_mask].copy()

print(f"After removing rows with zero or negative CEF scores: {len(df)}")



# %%
# Save the processed dataframe to CSV
output_path = f"/data/cef/combined_scores_{model_name}.csv"
df.to_csv(output_path, index=False)
print(f"Saved combined scores dataset to: {output_path}")
print(f"Dataset shape: {df.shape}")


# %%
# Calculate correlations between CEF and CEF_REF scores
print("=== Correlation between CEF and CEF_REF scores ===")

# Define the three main CEF dimensions
cef_dimensions = ['conformity_score', 'consistency_score', 'coverage_score']

cef_ref_correlations = {}
for dim in cef_dimensions:
    cef_col = f'cef_{dim}'
    cef_ref_col = f'cef_ref_{dim}'
    
    if cef_col in df.columns and cef_ref_col in df.columns:
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(df[cef_col], df[cef_ref_col])
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(df[cef_col], df[cef_ref_col])
        
        cef_ref_correlations[dim] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
        
        print(f"\n{dim}:")
        print(f"  Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"  Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")


# %%
# Create scatter plots comparing CEF scores with CEF_REF scores
print("\n=== Scatter plots: CEF vs CEF_REF scores ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, dim in enumerate(cef_dimensions):
    cef_col = f'cef_{dim}'
    cef_ref_col = f'cef_ref_{dim}'
    
    if cef_col in df.columns and cef_ref_col in df.columns:
        # Create scatter plot
        axes[i].scatter(df[cef_col] * 100 + np.random.uniform(-3, 3, len(df[cef_col])), df[cef_ref_col] * 100 + np.random.uniform(-3, 3, len(df[cef_ref_col])), alpha=0.6, s=20, color='#2E8B57')
        
        # Add diagonal line for perfect correlation
        min_val = min(df[cef_col].min(), df[cef_ref_col].min()) * 100
        max_val = max(df[cef_col].max(), df[cef_ref_col].max()) * 100
        axes[i].plot([10, 100], [10, 100], 'r--', alpha=0.9, linewidth=2)
        
        # Set labels and title with increased font sizes
        axes[i].set_xlabel(f'{dim.replace("_score", "").title()}', fontsize=16)
        axes[i].set_ylabel(f'{dim.replace("_score", "").title()} with Reference', fontsize=16)
        axes[i].set_title(f'{dim.replace("_score", "").title()} Score Comparison', fontsize=16)
        axes[i].set_xlim(0, 105)
        axes[i].set_ylim(0, 105)
        
        # Increase tick label font size
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        
        # Add correlation coefficient to plot with larger font
        if dim in cef_ref_correlations:
            r = cef_ref_correlations[dim]['pearson_r']
            axes[i].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)

plt.tight_layout()
plt.savefig(f'plots/scatter_correlation_cef.png', dpi=600, bbox_inches='tight')
plt.show()


# %%
# Calculate correlations between CEF scores and traditional metrics
print("\n=== Comprehensive 7x7 Correlation Study ===")

# Define all metrics for the correlation study
trad_cols = ['bleu', 'chrf', 'bertscore', 'comet']
cef_cols = [f'cef_ref_{dim}' for dim in cef_dimensions]
all_metrics = trad_cols + cef_cols

# Create comprehensive correlation matrices for different correlation types
correlation_methods = ['pearson', 'spearman', 'kendall', 'distance', 'shepherd']
correlation_matrices = {}
p_value_matrices = {}

for method in correlation_methods:
    correlation_matrices[method] = pd.DataFrame(index=all_metrics, columns=all_metrics)
    p_value_matrices[method] = pd.DataFrame(index=all_metrics, columns=all_metrics)

print("Calculating all pairwise correlations for multiple methods...")

# Import additional correlation methods
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, kendalltau

def distance_correlation(x, y):
    """Calculate distance correlation between two variables"""
    def _distance_covariance(x, y):
        n = len(x)
        a = np.sqrt((x[:, None] - x[None, :]) ** 2)
        b = np.sqrt((y[:, None] - y[None, :]) ** 2)
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        return np.sqrt((A * B).sum() / n**2)
    
    def _distance_variance(x):
        return _distance_covariance(x, x)
    
    dcov_xy = _distance_covariance(x, y)
    dvar_x = _distance_variance(x)
    dvar_y = _distance_variance(y)
    
    if dvar_x > 0 and dvar_y > 0:
        return dcov_xy / np.sqrt(dvar_x * dvar_y)
    else:
        return 0

def shepherd_pi_correlation(x, y):
    """Calculate Shepherd's Pi correlation"""
    n = len(x)
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    tied_both = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            x_diff = x[i] - x[j]
            y_diff = y[i] - y[j]
            
            if x_diff == 0 and y_diff == 0:
                tied_both += 1
            elif x_diff == 0:
                tied_x += 1
            elif y_diff == 0:
                tied_y += 1
            elif (x_diff > 0 and y_diff > 0) or (x_diff < 0 and y_diff < 0):
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = n * (n - 1) / 2
    if total_pairs > tied_x and total_pairs > tied_y:
        return (concordant - discordant) / np.sqrt((total_pairs - tied_x) * (total_pairs - tied_y))
    else:
        return 0

for i, metric1 in enumerate(all_metrics):
    if metric1 in df.columns:
        for j, metric2 in enumerate(all_metrics):
            if metric2 in df.columns:
                if i == j:
                    # Perfect correlation with itself
                    for method in correlation_methods:
                        correlation_matrices[method].loc[metric1, metric2] = 1.0
                        p_value_matrices[method].loc[metric1, metric2] = 0.0
                else:
                    # Calculate different correlation methods
                    x = df[metric1].values
                    y = df[metric2].values
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) > 1:
                        # Pearson correlation
                        pearson_r, pearson_p = pearsonr(x_clean, y_clean)
                        correlation_matrices['pearson'].loc[metric1, metric2] = pearson_r
                        p_value_matrices['pearson'].loc[metric1, metric2] = pearson_p
                        
                        # Spearman correlation
                        spearman_r, spearman_p = spearmanr(x_clean, y_clean)
                        correlation_matrices['spearman'].loc[metric1, metric2] = spearman_r
                        p_value_matrices['spearman'].loc[metric1, metric2] = spearman_p
                        
                        # Kendall tau correlation
                        kendall_r, kendall_p = kendalltau(x_clean, y_clean)
                        correlation_matrices['kendall'].loc[metric1, metric2] = kendall_r
                        p_value_matrices['kendall'].loc[metric1, metric2] = kendall_p
                        
                        # Distance correlation
                        dist_corr = distance_correlation(x_clean, y_clean)
                        correlation_matrices['distance'].loc[metric1, metric2] = dist_corr
                        p_value_matrices['distance'].loc[metric1, metric2] = np.nan  # No p-value for distance correlation
                        
                        # Shepherd's Pi correlation
                        shepherd_corr = shepherd_pi_correlation(x_clean, y_clean)
                        correlation_matrices['shepherd'].loc[metric1, metric2] = shepherd_corr
                        p_value_matrices['shepherd'].loc[metric1, metric2] = np.nan  # No p-value for Shepherd's Pi

# Convert to numeric for proper handling
for method in correlation_methods:
    correlation_matrices[method] = correlation_matrices[method].astype(float)
    p_value_matrices[method] = p_value_matrices[method].astype(float)

# Print detailed correlation results for all methods
print("\n=== Detailed Correlation Results (All Methods) ===")

# Traditional metrics vs CEF scores
for method in correlation_methods:
    print(f"\n{'='*50}")
    print(f"{method.upper()} CORRELATIONS")
    print(f"{'='*50}")
    
    print("\nTraditional Metrics vs CEF Scores:")
    for trad_metric in trad_cols:
        if trad_metric in df.columns:
            print(f"\n{trad_metric.upper()}:")
            for dim in cef_dimensions:
                cef_col = f'cef_ref_{dim}'
                if cef_col in df.columns:
                    corr_val = correlation_matrices[method].loc[trad_metric, cef_col]
                    if method in ['pearson', 'spearman', 'kendall']:
                        p_val = p_value_matrices[method].loc[trad_metric, cef_col]
                        print(f"  vs {dim}: r={corr_val:.4f}, p={p_val:.4f}")
                    else:
                        print(f"  vs {dim}: r={corr_val:.4f}")

    # Inter-traditional metrics correlations
    print(f"\nTraditional Metrics Inter-correlations ({method}):")
    for i, metric1 in enumerate(trad_cols):
        if metric1 in df.columns:
            print(f"\n{metric1.upper()}:")
            for metric2 in trad_cols[i+1:]:
                if metric2 in df.columns:
                    corr_val = correlation_matrices[method].loc[metric1, metric2]
                    if method in ['pearson', 'spearman', 'kendall']:
                        p_val = p_value_matrices[method].loc[metric1, metric2]
                        print(f"  vs {metric2}: r={corr_val:.4f}, p={p_val:.4f}")
                    else:
                        print(f"  vs {metric2}: r={corr_val:.4f}")

    # Inter-CEF scores correlations
    print(f"\nCEF Scores Inter-correlations ({method}):")
    for i, dim1 in enumerate(cef_dimensions):
        cef_col1 = f'cef_ref_{dim1}'
        if cef_col1 in df.columns:
            print(f"\n{dim1}:")
            for dim2 in cef_dimensions[i+1:]:
                cef_col2 = f'cef_ref_{dim2}'
                if cef_col2 in df.columns:
                    corr_val = correlation_matrices[method].loc[cef_col1, cef_col2]
                    if method in ['pearson', 'spearman', 'kendall']:
                        p_val = p_value_matrices[method].loc[cef_col1, cef_col2]
                        print(f"  vs {dim2}: r={corr_val:.4f}, p={p_val:.4f}")
                    else:
                        print(f"  vs {dim2}: r={corr_val:.4f}")

# %%
# Create comprehensive 7x7 heatmap visualizations for all correlation methods
print(f"\n=== Creating 7x7 Correlation Heatmaps for All Methods ===")

# Create a visualization-ready correlation matrix with proper ordering
viz_metrics = trad_cols + cef_cols

# Update subplots layout to accommodate 5 methods
fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()

method_titles = {
    'pearson': 'Pearson Correlation',
    'spearman': 'Spearman Correlation',
    'kendall': 'Kendall Tau Correlation', 
    'distance': 'Distance Correlation',
    'shepherd': "Shepherd's Pi Correlation"
}

for idx, method in enumerate(correlation_methods):
    viz_correlation_matrix = correlation_matrices[method].loc[viz_metrics, viz_metrics]
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(viz_correlation_matrix, dtype=bool), k=1)
    
    # Create heatmap
    sns.heatmap(viz_correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0, 
                square=True,
                mask=mask,
                vmin=-1, 
                vmax=1,
                fmt='.3f',
                cbar_kws={'label': f'{method_titles[method]} Coefficient'},
                linewidths=0.5,
                ax=axes[idx])
    
    axes[idx].set_title(f'7x7 {method_titles[method]} Matrix: Traditional Metrics vs CEF Scores', 
              fontsize=12, fontweight='bold', pad=20)
    axes[idx].set_xlabel('Metrics', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Metrics', fontsize=10, fontweight='bold')
    
    # Rotate labels for better readability
    axes[idx].tick_params(axis='x', rotation=45, labelsize=8)
    axes[idx].tick_params(axis='y', rotation=0, labelsize=8)

# Hide the last subplot since we only have 5 methods
axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

# Also create full matrices (without masking) for complete view
fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()

for idx, method in enumerate(correlation_methods):
    viz_correlation_matrix = correlation_matrices[method].loc[viz_metrics, viz_metrics]
    
    # Create heatmap without mask
    sns.heatmap(viz_correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0, 
                square=True,
                vmin=-1, 
                vmax=1,
                fmt='.3f',
                cbar=False,
                linewidths=0.5,
                ax=axes[idx])
    
    axes[idx].set_title(f'Complete 7x7 {method_titles[method]} Matrix: Traditional Metrics vs CEF Scores', 
              fontsize=12, fontweight='bold', pad=20)
    axes[idx].set_xlabel('Metrics', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Metrics', fontsize=10, fontweight='bold')
    
    # Rotate labels for better readability
    axes[idx].tick_params(axis='x', rotation=45, labelsize=8)
    axes[idx].tick_params(axis='y', rotation=0, labelsize=8)

# Hide the last subplot since we only have 5 methods
axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

# %%
# Create a single complete 7x7 Pearson correlation heatmap
print(f"\n=== Creating Complete 7x7 Pearson Correlation Heatmap ===")

# Create a visualization-ready correlation matrix with proper ordering
viz_metrics = trad_cols + cef_cols

# Get the Pearson correlation matrix
viz_correlation_matrix = correlation_matrices['pearson'].loc[viz_metrics, viz_metrics]

# Replace metric names for visualization
metric_name_mapping = {
    'cef_ref_conformity_score': 'Conformity',
    'cef_ref_consistency_score': 'Consistency', 
    'cef_ref_coverage_score': 'Coverage',
    'bleu': 'BLEU',
    'chrf': 'CHRF',
    'bertscore': 'BertScore',
    'comet': 'COMET'
}

# Apply name mapping to both index and columns
viz_correlation_matrix = viz_correlation_matrix.rename(index=metric_name_mapping, columns=metric_name_mapping)

# Clip values: no correlation below -0.1
viz_correlation_matrix = viz_correlation_matrix.clip(lower=-0.1)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Use 'YlGn' colormap: dark at low (near -0.1), light green at high (1)
cmap = sns.color_palette("viridis", as_cmap=True)

# Create heatmap
im = sns.heatmap(
    viz_correlation_matrix,
    annot=True,
    cmap=cmap,
    vmin=-0.1,  # Only show from -0.1 upward
    vmax=1.0,
    center=0.5,  # Center of color scale
    square=True,
    fmt='.3f',
    cbar=False,
    linewidths=0.5,
    ax=ax,
    annot_kws={'size': 14}
)

# Rotate labels
ax.tick_params(axis='x', rotation=45, labelsize=16)
ax.tick_params(axis='y', rotation=0, labelsize=16)

# Optional: Add title

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# Create correlation matrices for visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# CEF vs CEF_REF correlation matrix
cef_ref_matrix = []
for dim in cef_dimensions:
    if dim in cef_ref_correlations:
        cef_ref_matrix.append(cef_ref_correlations[dim]['pearson_r'])

if cef_ref_matrix:
    cef_ref_df = pd.DataFrame({
        'CEF_REF': cef_ref_matrix
    }, index=[dim.replace('_score', '') for dim in cef_dimensions])
    
    sns.heatmap(cef_ref_df, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('CEF vs CEF_REF Correlations (Pearson)')
    axes[0].set_ylabel('CEF Dimensions')

# CEF vs Traditional metrics correlation matrix
cef_trad_matrix = []
for dim in cef_dimensions:
    if dim in cef_trad_correlations:
        row = []
        for trad_metric in trad_cols:
            if trad_metric in cef_trad_correlations[dim]:
                row.append(cef_trad_correlations[dim][trad_metric]['pearson_r'])
            else:
                row.append(0)
        cef_trad_matrix.append(row)

if cef_trad_matrix:
    cef_trad_df = pd.DataFrame(cef_trad_matrix, 
                               columns=trad_cols,
                               index=[dim.replace('_score', '') for dim in cef_dimensions])
    
    sns.heatmap(cef_trad_df, annot=True, cmap='coolwarm', center=0,
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('CEF vs Traditional Metrics Correlations (Pearson)')
    axes[1].set_ylabel('CEF Dimensions')
    axes[1].set_xlabel('Traditional Metrics')

plt.tight_layout()
plt.show()


# %%



# Print each column of row 107 one by one
print("All columns for df.iloc[107] (printed individually):")
for col in df.columns:
    print(f"{col}: {df.iloc[107][col]}")


