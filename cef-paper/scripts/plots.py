# %%
from datasets import load_from_disk, Dataset, DatasetDict
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

# %% Fig 1A

# Load data
results_dir = "/data/cef/ntrex/results_cef_new"
today = ["Aug 22", "Aug 25", "Aug 29", "Sep 01", "Sep 11", "Sep 22"]

# Get folders from specified dates
folders = []
for item in os.listdir(results_dir):
    full_path = os.path.join(results_dir, item)
    if os.path.isdir(full_path):
        stats = os.stat(full_path)
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%b %d")
        if mod_time in today:
            folders.append((full_path, stats))

# Load reports
reports = []
for folder, _ in sorted(folders, key=lambda x: x[1].st_mtime):
    report_path = os.path.join(folder, "report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
            reports.append(report)

# Filter reports with 10 questions
filtered_reports = [report for report in reports if report['cef_params']['num_questions_to_generate'] == 10]

# Configuration
FIXED_JUDGE = "deepseek-ai/DeepSeek-V3"
target_models = ["qwen3-1.7b", "gemma3-4b", "gpt-oss-20b", "nllb-3.3B", "azure", "google"]
lang_pairs = ["en-fr", "en-de", "en-es", "en-jp", "en-ar"]

# Collect scores for heatmap
scores = {}
for lang_pair in lang_pairs:
    for report in filtered_reports:
        if (lang_pair in report['scores'] and 
            report['scores'][lang_pair] != {} and
            report['cef_params']['judge_model'] == FIXED_JUDGE):
            scores_data = report['scores'][lang_pair]
            data_path = report['cef_params']['data_path']
            if "nolist" in data_path:
                model = data_path.split('/')[-1].split('_')[-1] + "_nolist"
            else:
                model = data_path.split('/')[-1].split('_')[-1]
            
            if model in target_models:
                if model not in scores:
                    scores[model] = {}
                scores[model][lang_pair] = {
                    'Conformity': scores_data['conformity_score']['mean'],
                    'Consistency': scores_data['consistency_score']['mean'],
                    'Coverage': scores_data['coverage_score']['mean']
                }

# Create heatmap visualization
language_pairs = ["en-fr", "en-es", "en-ar", "en-jp"]
metrics = ["Conformity", "Consistency", "Coverage"]
columns = [f"{lp} {m}" for lp in language_pairs for m in metrics]
 
df_rows = []
for model, langs in scores.items():
    row = {}
    for lp in language_pairs:
        for m in metrics:
            row[f"{lp} {m}"] = langs.get(lp, {}).get(m, None)
    df_rows.append(pd.Series(row, name=model))
 
df = pd.DataFrame(df_rows)

# Replace model names for display
df.index = df.index.str.replace('google', 'Google Translate')
df.index = df.index.str.replace('azure', 'Azure Translate')
 
# Plot zoomed heatmap
fig, ax = plt.subplots(figsize=(14, 5))
 
data = df.values.astype(float)
 
# Restrict scale to emphasize high-end variation
vmin, vmax = 85, 100
 
im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Score", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Add vertical dividers between language pairs
num_metrics = len(metrics)
for i in range(1, len(language_pairs)):
    x_pos = i * num_metrics - 0.5
    ax.axvline(x=x_pos, color='black', linewidth=2)
 
# Axis ticks
ax.set_yticks(np.arange(df.shape[0]))
ax.set_yticklabels(df.index, fontsize=12)
 
# Set bottom x-axis labels (metrics)
ax.set_xticks(np.arange(df.shape[1]))
ax.set_xticklabels([col.split(' ')[1] for col in columns], rotation=45, ha="right", fontsize=12)

# Add top x-axis for language pairs
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
# Center the language labels over their respective metric groups
lang_positions = [(i * num_metrics + (num_metrics - 1) / 2) for i in range(len(language_pairs))]
ax2.set_xticks(lang_positions)
ax2.set_xticklabels(language_pairs, fontsize=12)
ax2.tick_params(axis='x', which='major', pad=1)
  
# Annotate cells
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        val = df.iat[i, j]
        if pd.notnull(val):
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                fontsize=11,
                color="white" if val < 92 else "black"
            )
 
plt.tight_layout(pad=1.0)
plt.savefig('plots/cef_scores.pdf', dpi=600, format='pdf', bbox_inches='tight')
plt.show()

# %% Fig 1B

# Load the pre-saved combined scores data
model_name = "qwen3-1.7b"
df = pd.read_csv(f"/data/cef/combined_scores_{model_name}.csv")

# Define the three main CEF dimensions
cef_dimensions = ['conformity_score', 'consistency_score', 'coverage_score']

# Calculate correlations between CEF and CEF_REF scores
cef_ref_correlations = {}
for dim in cef_dimensions:
    cef_col = f'cef_{dim}'
    cef_ref_col = f'cef_ref_{dim}'
    
    if cef_col in df.columns and cef_ref_col in df.columns:
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(df[cef_col], df[cef_ref_col])
        
        cef_ref_correlations[dim] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p
        }

# Create scatter plots comparing CEF scores with CEF_REF scores
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, dim in enumerate(cef_dimensions):
    cef_col = f'cef_{dim}'
    cef_ref_col = f'cef_ref_{dim}'
    
    if cef_col in df.columns and cef_ref_col in df.columns:
        # Create scatter plot
        axes[i].scatter(df[cef_col] * 100 + np.random.uniform(-3, 3, len(df[cef_col])), 
                       df[cef_ref_col] * 100 + np.random.uniform(-3, 3, len(df[cef_ref_col])), 
                       alpha=0.6, s=20, color='#2E8B57')
        
        # Add diagonal line for perfect correlation
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
plt.savefig('plots/scatter_correlation_cef.pdf', dpi=600, format='pdf', bbox_inches='tight')
plt.show()

# %% Fig 2

# Try to load the pre-saved combined scores data, otherwise use sample data
model_name = "qwen3-1.7b"
csv_path = f"/data/cef/combined_scores_{model_name}.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    print(f"CSV file {csv_path} not found. Creating sample data for demonstration.")
    # Create sample data with the expected structure
    np.random.seed(42)
    n_samples = 500
    
    # Generate correlated sample data
    base_conformity = np.random.beta(2, 2, n_samples)
    base_consistency = np.random.beta(2, 2, n_samples) 
    base_coverage = np.random.beta(2, 2, n_samples)
    base_bleu = np.random.beta(2, 3, n_samples)
    base_chrf = np.random.beta(2, 3, n_samples)
    base_bertscore = np.random.beta(3, 2, n_samples)
    base_comet = np.random.beta(3, 2, n_samples)
    
    df = pd.DataFrame({
        'cef_ref_conformity_score': base_conformity,
        'cef_ref_consistency_score': base_consistency,
        'cef_ref_coverage_score': base_coverage,
        'bleu': base_bleu,
        'chrf': base_chrf,
        'bertscore': base_bertscore,
        'comet': base_comet,
    })

# Define metrics for the correlation study
trad_cols = ['bleu', 'chrf', 'bertscore', 'comet']
cef_dimensions = ['conformity_score', 'consistency_score', 'coverage_score']
cef_cols = [f'cef_ref_{dim}' for dim in cef_dimensions]
all_metrics = trad_cols + cef_cols

# Create Pearson correlation matrix
correlation_matrix = pd.DataFrame(index=all_metrics, columns=all_metrics)

# Calculate pairwise correlations
for metric1 in all_metrics:
    if metric1 in df.columns:
        for metric2 in all_metrics:
            if metric2 in df.columns:
                if metric1 == metric2:
                    # Perfect correlation with itself
                    correlation_matrix.loc[metric1, metric2] = 1.0
                else:
                    # Calculate Pearson correlation
                    x = df[metric1].values
                    y = df[metric2].values
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) > 1:
                        pearson_r, _ = pearsonr(x_clean, y_clean)
                        correlation_matrix.loc[metric1, metric2] = pearson_r

# Convert to numeric for proper handling
correlation_matrix = correlation_matrix.astype(float)

# Create visualization-ready correlation matrix with proper ordering
viz_metrics = trad_cols + cef_cols
viz_correlation_matrix = correlation_matrix.loc[viz_metrics, viz_metrics]

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

# Create the plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Use viridis colormap
cmap = sns.color_palette("viridis", as_cmap=True)

# Create heatmap
sns.heatmap(
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

# Adjust layout
plt.tight_layout()

# Save as PDF
plt.savefig('plots/heatmap_correlation_cef_trad.pdf', dpi=600, format='pdf', bbox_inches='tight')
plt.show()
print("Heatmap saved to plots/heatmap_correlation_cef_trad.pdf")
# %% Fig 3A and 3B

# Change prompt to "translation2" for Fig 3B
prompt = "translation2"

# Load dataset
ds = load_from_disk(os.path.join("/data/cef/yes_only", f"{prompt}"))

# Extract answers and previous answers for confusion matrix
all_answers = []
all_previous_answers = []

ANSWER_COLUMN = "question_list_trg"

for split_name, split_data in ds.items():
    for example in split_data:
        if ANSWER_COLUMN in example and example[ANSWER_COLUMN] is not None:
            for qa_pair in example[ANSWER_COLUMN]:
                if qa_pair["answer"] is not None and qa_pair["previous_answer"] is not None:
                    all_answers.append(qa_pair["answer"])
                    all_previous_answers.append(qa_pair["previous_answer"])

# Define the labels
labels = ["YES", "NO", "IDK"]

# Create confusion matrix
cm = confusion_matrix(all_previous_answers, all_answers, labels=labels)

# Create the plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Plot confusion matrix
plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis_r', 
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 24}, square=False)

if prompt == "translation":
    plt.title('YES-ONLY prompt', fontsize=26, pad=8)
else:
    plt.title('Mixed-Answer prompt', fontsize=26, pad=8)

plt.xlabel('Re-Evaluated Answer', fontsize=22)
plt.ylabel('Generated Answer', fontsize=22)
plt.tick_params(axis='both', labelsize=20)

# Make colorbar labels bigger
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig(f"plots/{prompt}.pdf", format='pdf', bbox_inches='tight')
plt.show()

print(f"Plot saved to plots/{prompt}.pdf")
print(f"Total question pairs: {len(all_answers)}")
print(f"Confusion Matrix:\n{cm}")
# %%
