# %%

from datasets import load_from_disk
import json
from datasets import Dataset, DatasetDict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# # %%
# DATASET="ntrex"
# results_dir = f"/data/cef/{DATASET}/results_cef"
# RESULTS_DATASET_NAME = "results_ds"

# # %%
# import os
# from pathlib import Path

# # List all result folders
# result_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
# print(f"Found {len(result_folders)} result folders:")
# for folder in result_folders:
#     print(f"  - {folder}")

# # Load all results and filter by latest unique parameters
# import glob
# from datetime import datetime

# # Get all report.json files
# report_files = glob.glob(os.path.join(results_dir, "*/report.json"))

# # Load all reports and their metadata
# reports_data = []
# for report_file in report_files:
#     folder_name = os.path.basename(os.path.dirname(report_file))
    
#     # Get folder creation time
#     folder_path = os.path.dirname(report_file)
#     creation_time = os.path.getctime(folder_path)
    
#     with open(report_file, 'r') as f:
#         report = json.load(f)
    
#     reports_data.append({
#         'folder': folder_name,
#         'creation_time': creation_time,
#         'report': report
#     })

# # Group by CEF parameters (excluding creation time)
# def get_params_key(report):
#     cef_params = report.get('cef_params', {})
#     # Create a key based on parameters that should be the same for duplicates
#     return (
#         cef_params.get('data_path'),
#         cef_params.get('prompt_catalogue_path'),
#         cef_params.get('num_questions_to_generate'),
#         cef_params.get('token_per_question'),
#         cef_params.get('bootstrap_iterations'),
#         cef_params.get('bootstrap_fraction')
#     )

# # Group reports by parameters and keep only the latest one for each group
# unique_reports = {}
# for data in reports_data:
#     params_key = get_params_key(data['report'])
#     if params_key not in unique_reports or data['creation_time'] > unique_reports[params_key]['creation_time']:
#         unique_reports[params_key] = data

# # Load datasets for unique reports
# unique_results = {}
# for params_key, data in unique_reports.items():
#     folder_name = data['folder']
#     dataset_path = os.path.join(results_dir, folder_name, RESULTS_DATASET_NAME)
    
#     if os.path.exists(dataset_path):
#         dataset = load_from_disk(dataset_path)
#         unique_results[params_key] = {
#             'folder': folder_name,
#             'report': data['report'],
#             'dataset': dataset
#         }
#         print(f"Loaded results from {folder_name} (created: {datetime.fromtimestamp(data['creation_time'])})")
#     else:
#         print(f"Warning: Dataset not found for {folder_name}")

# print(f"\nLoaded {len(unique_results)} unique result sets:")
# for params_key, result in unique_results.items():
#     print(f"  - {result['folder']}: {len(result['dataset'])} splits")
#     for split_name, split_data in result['dataset'].items():
#         print(f"    - {split_name}: {len(split_data)} samples")


# # Load and examine one example to understand the structure

# # %%
# base_path1 = f"/data/cef/{DATASET}/results_cef"
# base_path2 = f"/data/cef/{DATASET}/results_trad"
# base_path3 = f"/data/cef/{DATASET}/results_tq"

# ##
# import json
# from collections import defaultdict

# def load_reports_and_datasets(base_paths):
#     """
#     Load all reports and datasets from the given base paths, grouped by datapath.
    
#     Args:
#         base_paths: dict with keys as path names and values as base paths
        
#     Returns:
#         dict: Grouped results by datapath
#     """
#     all_results = defaultdict(lambda: {'reports': [], 'datasets': []})
    
#     for path_name, base_path in base_paths.items():
#         print(f"\nProcessing {path_name} from {base_path}")
        
#         if not os.path.exists(base_path):
#             print(f"Warning: Path {base_path} does not exist")
#             continue
            
#         # Get all folders in the base path, excluding debug folders
#         folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and 'debug' not in f.lower()]
#         print(f"Found {len(folders)} folders (excluding debug)")
        
#         # Collect all reports for this path_name to find latest ones
#         path_reports = []
        
#         for folder in folders:
#             folder_path = os.path.join(base_path, folder)
#             report_path = os.path.join(folder_path, "report.json")
#             dataset_path = os.path.join(folder_path, RESULTS_DATASET_NAME)
            
#             # Load report
#             if os.path.exists(report_path):
#                 try:
#                     with open(report_path, 'r') as f:
#                         report = json.load(f)
                    
#                     # Extract datapath from report params
#                     datapath = None
#                     # Find the parameter key that contains 'params' as substring
#                     param_key = None
#                     for key in report.keys():
#                         if 'params' in key:
#                             param_key = key
#                             break
                    
#                     if param_key and 'data_path' in report[param_key]:
#                         datapath = report[param_key]['data_path']
#                     else:
#                         print(f"Warning: Could not find datapath in report for {folder}")
#                         continue
                    
#                     report["params"] = report[param_key]
#                     del report[param_key]
#                     # Get folder creation time
#                     creation_time = os.path.getctime(folder_path)
                    
#                     path_reports.append({
#                         'path_name': path_name,
#                         'folder': folder,
#                         'report': report,
#                         'datapath': datapath,
#                         'creation_time': creation_time
#                     })
                    
#                 except Exception as e:
#                     print(f"  ✗ Failed to load report for {folder}: {e}")
        
#         # Group by datapath and keep only the latest report for each datapath
#         datapath_latest = {}
#         for report_info in path_reports:
#             datapath = report_info['datapath']
#             if datapath not in datapath_latest or report_info['creation_time'] > datapath_latest[datapath]['creation_time']:
#                 datapath_latest[datapath] = report_info
        
#         # Load datasets for the latest reports
#         for datapath, report_info in datapath_latest.items():
#             folder = report_info['folder']
#             folder_path = os.path.join(base_path, folder)
#             dataset_path = os.path.join(folder_path, RESULTS_DATASET_NAME)
            
#             print(f"Datapath: {datapath} (latest: {folder})")
            
#             # Load dataset
#             if os.path.exists(dataset_path):
#                 try:
#                     dataset = load_from_disk(dataset_path)
#                     all_results[datapath]['reports'].append({
#                         'path_name': report_info['path_name'],
#                         'folder': report_info['folder'],
#                         'report': report_info['report']
#                     })
#                     all_results[datapath]['datasets'].append({
#                         'path_name': report_info['path_name'],
#                         'folder': report_info['folder'],
#                         'dataset': dataset
#                     })
#                     print(f"  ✓ Loaded {folder} (datapath: {datapath})")
#                 except Exception as e:
#                     print(f"  ✗ Failed to load dataset for {folder}: {e}")
#             else:
#                 print(f"  ✗ Dataset not found for {folder}")
    
#     return dict(all_results)

# # Define base paths
# base_paths = {
#     'cef': base_path1,
#     'traditional': base_path2,
#     'translation_quality': base_path3
# }

# # Load all reports and datasets grouped by datapath
# grouped_results = load_reports_and_datasets(base_paths)

# # Print summary

# # %%
# # Print summary of grouped results
# print("Summary of grouped results:")
# for datapath, data in grouped_results.items():
#     print(f"\nDatapath: {datapath}")
#     print(f"  Reports: {len(data['reports'])}")
#     print(f"  Datasets: {len(data['datasets'])}")
#     for dataset_info in data['datasets']:
#         print(f"    - {dataset_info['path_name']}: {dataset_info['folder']}")

# # Function to merge datasets by ID
# def merge_datasets_by_id(datasets_info):
#     """
#     Merge multiple datasets by ID, combining all columns from different evaluation methods.
    
#     Args:
#         datasets_info: List of dicts with 'path_name' and 'dataset' keys
        
#     Returns:
#         dict: Dictionary with language codes as keys and merged datasets as values
#     """
#     # Get all language codes from the first dataset
#     first_dataset = datasets_info[0]['dataset']
#     language_codes = list(first_dataset.keys())
    
#     merged_datasets = {}
    
#     for lang_code in language_codes:
#         print(f"\nMerging datasets for language: {lang_code}")
        
#         # Start with the first dataset
#         merged_df = datasets_info[0]['dataset'][lang_code].to_pandas()
#         print(f"  Base dataset: {datasets_info[0]['path_name']} ({len(merged_df)} rows)")
        
#         # Merge with subsequent datasets
#         for i, dataset_info in enumerate(datasets_info[1:], 1):
#             current_df = dataset_info['dataset'][lang_code].to_pandas()
#             print(f"  Merging with: {dataset_info['path_name']} ({len(current_df)} rows)")
            
#             # Merge on 'id' column, keeping all columns from both datasets
#             merged_df = merged_df.merge(
#                 current_df, 
#                 on='id', 
#                 how='outer',
#                 suffixes=('', f'_{dataset_info["path_name"]}')
#             )
            
#             print(f"  After merge: {len(merged_df)} rows")
        
#         # Convert back to Dataset
#         from datasets import Dataset
#         merged_datasets[lang_code] = Dataset.from_pandas(merged_df)
        
#         # Print column information
#         print(f"  Final columns: {list(merged_df.columns)}")
#         print(f"  Final rows: {len(merged_df)}")
    
#     return merged_datasets

# # Merge datasets for each datapath
# merged_results = {}
# for datapath, data in grouped_results.items():
#     print(f"\n{'='*60}")
#     print(f"Processing datapath: {datapath}")
#     print(f"{'='*60}")
    
#     if len(data['datasets']) > 1:
#         merged_datasets = merge_datasets_by_id(data['datasets'])
#         merged_results[datapath] = merged_datasets
#     else:
#         print("Only one dataset found, no merging needed")
#         # Convert single dataset to the same format
#         first_dataset = data['datasets'][0]['dataset']
#         merged_results[datapath] = {lang: dataset for lang, dataset in first_dataset.items()}

# # Print final summary
# print(f"\n{'='*60}")
# print("FINAL SUMMARY")
# print(f"{'='*60}")
# for datapath, datasets in merged_results.items():
#     print(f"\nDatapath: {datapath}")
#     for lang_code, dataset in datasets.items():
#         print(f"  {lang_code}: {len(dataset)} rows, {len(dataset.column_names)} columns")
#         print(f"    Columns: {dataset.column_names}")

# # Example: Access merged data for the first datapath
# if merged_results:
#     first_datapath = list(merged_results.keys())[0]
#     print(f"\nExample - First datapath ({first_datapath}) datasets:")
#     for lang_code, dataset in merged_results[first_datapath].items():
#         print(f"  {lang_code}: {len(dataset)} samples")
#         if len(dataset) > 0:
#             print(f"    Sample columns: {dataset.column_names[:5]}...")

# # %%
# # Filter datasets to keep only required columns
# required_columns = [
#     'id', 'src', 'trg', 'bleu', 'chrf', 'bertscore', 
#     'reference_translation', 'translation_quality_score', 
#     'translation_quality_analysis', 'cef_scores', 'cef_details'
# ]

# print(f"\n{'='*60}")
# print("FILTERING DATASETS TO REQUIRED COLUMNS")
# print(f"{'='*60}")

# filtered_results = {}
# for datapath, datasets in merged_results.items():
#     print(f"\nDatapath: {datapath}")
#     filtered_results[datapath] = {}
    
#     for lang_code, dataset in datasets.items():
#         print(f"  Processing {lang_code}...")
        
#         # Check which required columns are missing
#         missing_columns = [col for col in required_columns if col not in dataset.column_names]
#         if missing_columns:
#             print(f"    WARNING: Missing columns: {missing_columns}")
        
#         # Filter to only keep required columns that exist
#         available_columns = [col for col in required_columns if col in dataset.column_names]
#         print(f"    Available columns: {available_columns}")
        
#         # Select only the required columns
#         filtered_dataset = dataset.select_columns(available_columns)
#         filtered_results[datapath][lang_code] = filtered_dataset
        
#         print(f"    Final: {len(filtered_dataset)} rows, {len(filtered_dataset.column_names)} columns")

# # Update merged_results with filtered data
# merged_results = filtered_results

# # Print final summary after filtering
# print(f"\n{'='*60}")
# print("FINAL SUMMARY AFTER FILTERING")
# print(f"{'='*60}")
# for datapath, datasets in merged_results.items():
#     print(f"\nDatapath: {datapath}")
#     for lang_code, dataset in datasets.items():
#         print(f"  {lang_code}: {len(dataset)} rows, {len(dataset.column_names)} columns")
#         print(f"    Columns: {dataset.column_names}")






# # Divide translation quality scores by 100 in datasets
# # %%
# # Divide translation quality scores by 100 in reports
# for datapath, data in grouped_results.items():
#     for report in data['reports']:
#         scores = report['report']['scores']
#         for lang_pair, lang_scores in scores.items():
#             if 'translation_quality_score' in lang_scores:
#                 # Update the mean score
#                 lang_scores['translation_quality_score']['mean'] /= 100
#                 # Update confidence interval if it exists
#                 if 'confidence_interval' in lang_scores['translation_quality_score']:
#                     ci = lang_scores['translation_quality_score']['confidence_interval']
#                     if isinstance(ci, (list, tuple)) and len(ci) == 2:
#                         lang_scores['translation_quality_score']['confidence_interval'] = [ci[0]/100, ci[1]/100]

# print("Translation quality scores divided by 100 in both datasets and reports")


# # %%
# # Create tables for each language showing all scores across different evaluation methods
# # Function to merge reports by datapath
# def merge_reports_by_datapath(reports_info):
#     """
#     Merge multiple reports by datapath, combining scores from different evaluation methods.
    
#     Args:
#         reports_info: List of dicts with 'path_name' and 'report' keys
        
#     Returns:
#         dict: Merged report with combined scores
#     """
#     if len(reports_info) == 1:
#         return reports_info[0]['report']
    
#     # Start with the first report
#     merged_report = reports_info[0]['report'].copy()
#     print(f"  Base report: {reports_info[0]['path_name']}")
    
#     # Merge with subsequent reports
#     for i, report_info in enumerate(reports_info[1:], 1):
#         current_report = report_info['report']
#         print(f"  Merging with: {report_info['path_name']}")
        
#         # Merge scores
#         if 'scores' in current_report and 'scores' in merged_report:
#             for lang_pair, lang_scores in current_report['scores'].items():
#                 if lang_pair not in merged_report['scores']:
#                     merged_report['scores'][lang_pair] = {}
                
#                 # Merge individual score types
#                 for score_type, score_data in lang_scores.items():
#                     if score_type not in merged_report['scores'][lang_pair]:
#                         merged_report['scores'][lang_pair][score_type] = score_data
#                     else:
#                         # If score type already exists, keep the one from the first report
#                         print(f"    WARNING: Score type {score_type} already exists for {lang_pair}, keeping original")
    
#     return merged_report

# # Merge reports for each datapath
# merged_reports = {}
# for datapath, data in grouped_results.items():
#     print(f"\n{'='*60}")
#     print(f"Merging reports for datapath: {datapath}")
#     print(f"{'='*60}")
    
#     if len(data['reports']) > 1:
#         merged_report = merge_reports_by_datapath(data['reports'])
#         merged_reports[datapath] = merged_report
#     else:
#         print("Only one report found, no merging needed")
#         merged_reports[datapath] = data['reports'][0]['report']

# # Print summary of merged reports
# print(f"\n{'='*60}")
# print("MERGED REPORTS SUMMARY")
# print(f"{'='*60}")
# for datapath, report in merged_reports.items():
#     print(f"\nDatapath: {datapath}")
#     if 'scores' in report:
#         for lang_pair, lang_scores in report['scores'].items():
#             print(f"  {lang_pair}:")
#             for score_type, score_data in lang_scores.items():
#                 if isinstance(score_data, dict) and 'mean' in score_data:
#                     print(f"    {score_type}: {score_data['mean']:.4f}")
#                 else:
#                     print(f"    {score_type}: {score_data}")


# # %%
# # Create tables for each language pair
# print("="*80)
# print("SCORE COMPARISON TABLES BY LANGUAGE PAIR")
# print("="*80)

# # Get all unique language pairs
# all_lang_pairs = set()
# for datapath, report in merged_reports.items():
#     if 'scores' in report:
#         all_lang_pairs.update(report['scores'].keys())

# # Get all unique score types
# all_score_types = set()
# for datapath, report in merged_reports.items():
#     if 'scores' in report:
#         for lang_pair, lang_scores in report['scores'].items():
#             all_score_types.update(lang_scores.keys())

# # Create a table for each language pair
# for lang_pair in sorted(all_lang_pairs):
#     print(f"\n{'-'*80}")
#     print(f"LANGUAGE PAIR: {lang_pair}")
#     print(f"{'-'*80}")
    
#     # Create DataFrame for this language pair
#     data = []
#     datapaths = []
    
#     for datapath, report in merged_reports.items():
#         if 'scores' in report and lang_pair in report['scores']:
#             datapaths.append(datapath)
#             row = {}
#             for score_type in sorted(all_score_types):
#                 if score_type in report['scores'][lang_pair]:
#                     score_data = report['scores'][lang_pair][score_type]
#                     if isinstance(score_data, dict) and 'mean' in score_data:
#                         mean = score_data['mean']
#                         std = score_data.get('confidence_interval', 0)
#                         # Format as mean ± std (confidence interval)
#                         row[score_type] = f"{mean:.2f} || {std}"
#                     else:
#                         row[score_type] = str(score_data)
#                 else:
#                     row[score_type] = "N/A"
#             data.append(row)
    
#     if data:
#         df = pd.DataFrame(data, index=datapaths)
        
#         # Clean up datapath names for display
#         df.index = [dp.split('/')[-1] for dp in datapaths]
        
#         # Display the table
#         display(df)
#     else:
#         print("No data available for this language pair")

#  # %%
# for dataset_path in merged_results.keys():
#     for lang_pair in merged_results[dataset_path].keys():
#         merged_results[dataset_path][lang_pair] = merged_results[dataset_path][lang_pair].map(lambda x: {
#             "coverage": x["cef_scores"]["coverage_score"],
#             "consistency": x["cef_scores"]["consistency_score"],
#             "conformity": x["cef_scores"]["conformity_score"],
#             'overall_cef': x["cef_scores"]["overall_score"]
#         })


# # %%
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Get all dataset paths and language pairs
# dataset_paths = list(merged_results.keys())

# # Get all available language pairs from the data
# all_lang_pairs = set()
# for dataset_path in dataset_paths:
#     all_lang_pairs.update(merged_results[dataset_path].keys())

# # Define score types to plot
# score_types = ['coverage', 'consistency', 'conformity', 'bleu', 'chrf', 'bertscore', 'translation_quality_score']

# # Colors for different datasets
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# # Process each language pair
# for lang_pair in sorted(all_lang_pairs):
#     print(f"\n{'='*80}")
#     print(f"PROCESSING LANGUAGE PAIR: {lang_pair}")
#     print(f"{'='*80}")
    
#     # Create subplots for each score type - 4 columns, 2 rows for higher resolution
#     fig, axes = plt.subplots(4, 2, figsize=(32, 64), dpi=300)
#     fig.suptitle(f'Score Distribution for {lang_pair}', fontsize=28, fontweight='bold')

#     # Flatten axes for easier iteration
#     axes = axes.flatten()

#     for i, score_type in enumerate(score_types):
#         ax = axes[i]
        
#         # Collect all scores for this score type across datasets
#         all_scores = []
#         all_dataset_names = []
#         all_sample_indices = []
        
#         for dataset_path in dataset_paths:
#             if lang_pair in merged_results[dataset_path]:
#                 dataset = merged_results[dataset_path][lang_pair]
#                 if score_type in dataset.column_names:
#                     # Get individual sample scores
#                     score_values = [float(x) for x in dataset[score_type] if x is not None and str(x) != 'nan']
#                     if score_values:
#                         all_scores.extend(score_values)
#                         dataset_name = dataset_path.split('/')[-1]
#                         all_dataset_names.extend([dataset_name] * len(score_values))
#                         all_sample_indices.extend(range(len(score_values)))
        
#         if all_scores:
#             # Create DataFrame for easier plotting
#             import pandas as pd
#             df = pd.DataFrame({
#                 'score': all_scores,
#                 'dataset': all_dataset_names,
#                 'sample_idx': all_sample_indices
#             })
            
#             # Create violin plot with individual points
#             sns.violinplot(data=df, x='dataset', y='score', ax=ax, palette=colors[:len(df['dataset'].unique())])
            
#             # Add individual points
#             sns.stripplot(data=df, x='dataset', y='score', ax=ax, 
#                          color='black', alpha=0.3, size=3, jitter=0.2)
            
#             # Customize the plot
#             ax.set_title(f'{score_type.capitalize()} Score Distribution', fontweight='bold', fontsize=20)
#             ax.set_ylabel('Score', fontsize=18)
#             ax.set_xlabel('Dataset', fontsize=18)
            
#             # Rotate x-axis labels for better readability
#             ax.tick_params(axis='x', rotation=45, labelsize=16)
#             ax.tick_params(axis='y', labelsize=16)
            
#             # Add grid for better readability
#             ax.grid(axis='y', alpha=0.3)
            
#             # Add statistics text
#             stats_text = f"Mean: {np.mean(all_scores):.3f}\nStd: {np.std(all_scores):.3f}"
#             ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
#                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#                    fontsize=16)
        
#         else:
#             ax.text(0.5, 0.5, f'No data for {score_type}', 
#                    ha='center', va='center', transform=ax.transAxes, fontsize=20)

#     # Hide unused subplot if we have less than 8 score types
#     for i in range(len(score_types), len(axes)):
#         axes[i].set_visible(False)

#     plt.tight_layout()
#     plt.show()

#     # Print summary statistics
#     print(f"\n{'='*60}")
#     print(f"SUMMARY STATISTICS FOR {lang_pair}")
#     print(f"{'='*60}")

#     for score_type in score_types:
#         print(f"\n{score_type.upper()} SCORES:")
#         print("-" * 40)
#         for dataset_path in dataset_paths:
#             if lang_pair in merged_results[dataset_path]:
#                 dataset = merged_results[dataset_path][lang_pair]
#                 if score_type in dataset.column_names:
#                     score_values = [float(x) for x in dataset[score_type] if x is not None and str(x) != 'nan']
#                     if score_values:
#                         mean_score = np.mean(score_values)
#                         std_score = np.std(score_values)
#                         min_score = np.min(score_values)
#                         max_score = np.max(score_values)
#                         dataset_name = dataset_path.split('/')[-1]
#                         print(f"{dataset_name:30} | Mean: {mean_score:.3f} ± {std_score:.3f} | Range: [{min_score:.3f}, {max_score:.3f}]")

# # %%
# # Correlation analysis between different scores for all language pairs
# # Get all available language pairs
# dataset_paths = [path for path in merged_results.keys() if f'/data/cef/{DATASET}/{DATASET}_original' and f'/data/cef/{DATASET}/{DATASET}_qwen3-0.6b_all_languages' not in path]
# all_lang_pairs = set()
# for dataset_path in dataset_paths:
#     all_lang_pairs.update(merged_results[dataset_path].keys())

# print(f"Available language pairs: {sorted(all_lang_pairs)}")

# # Define score types based on available columns
# score_types = ['bleu', 'chrf', 'bertscore', 'translation_quality_score', 'coverage', 'consistency', 'conformity']

# # Function to collect correlation data for a given language pair
# def collect_correlation_data(lang_pair):
#     """Collect all scores for correlation analysis for a given language pair."""
#     all_scores_data = {}

#     for dataset_path in dataset_paths:
#         if lang_pair in merged_results[dataset_path]:
#             dataset = merged_results[dataset_path][lang_pair]
            
#             for score_type in score_types:
#                 if score_type in dataset.column_names:
#                     score_values = [float(x) for x in dataset[score_type] if x is not None and str(x) != 'nan']
#                     if score_values:
#                         if score_type not in all_scores_data:
#                             all_scores_data[score_type] = []
#                         all_scores_data[score_type].extend(score_values)
    
#     return all_scores_data

# # Collect correlation data for all language pairs
# lang_pair_data = {}
# for lang_pair in sorted(all_lang_pairs):
#     data = collect_correlation_data(lang_pair)
#     if data:
#         lang_pair_data[lang_pair] = data

# # Create 1x3 plot showing Pearson correlations for all language pairs
# fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# # Get the first 3 language pairs (or all if less than 3)
# lang_pairs_to_plot = list(lang_pair_data.keys())[:3]

# for idx, lang_pair in enumerate(lang_pairs_to_plot):
#     ax = axes[idx]
#     data = lang_pair_data[lang_pair]
    
#     if data:
#         # Convert to DataFrame for correlation analysis
#         import pandas as pd
#         df = pd.DataFrame(data)
        
#         # Calculate Pearson correlation matrix
#         pearson_corr = df.corr(method='pearson')
        
#         # Create correlation heatmap
#         im = ax.imshow(pearson_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
#         ax.set_title(f'Pearson Correlation - {lang_pair}', fontweight='bold', fontsize=14)
#         ax.set_xticks(range(len(pearson_corr.columns)))
#         ax.set_yticks(range(len(pearson_corr.columns)))
#         ax.set_xticklabels(pearson_corr.columns, rotation=45, ha='right', fontsize=10)
#         ax.set_yticklabels(pearson_corr.columns, fontsize=10)
        
#         # Add correlation values as text
#         for i in range(len(pearson_corr.columns)):
#             for j in range(len(pearson_corr.columns)):
#                 text = ax.text(j, i, f'{pearson_corr.iloc[i, j]:.2f}',
#                                ha="center", va="center", color="black", fontweight='bold', fontsize=8)
#     else:
#         ax.text(0.5, 0.5, f'No data for {lang_pair}', 
#                ha='center', va='center', transform=ax.transAxes, fontsize=12)


# plt.tight_layout()
# plt.show()

# # Print summary statistics for all language pairs
# print(f"\n{'='*80}")
# print(f"CORRELATION SUMMARY FOR ALL LANGUAGE PAIRS")
# print(f"{'='*80}")

# for lang_pair in sorted(lang_pair_data.keys()):
#     print(f"\n{lang_pair.upper()}:")
#     print("-" * 40)
    
#     data = lang_pair_data[lang_pair]
#     if data:
#         import pandas as pd
#         from scipy.stats import pearsonr
        
#         df = pd.DataFrame(data)
#         pearson_corr = df.corr(method='pearson')
        
#         # Find strongest correlations
#         strong_correlations = []
#         score_types_list = list(data.keys())
        
#         for i, score1 in enumerate(score_types_list):
#             for j, score2 in enumerate(score_types_list):
#                 if i < j:
#                     corr_value = pearson_corr.iloc[i, j]
#                     if abs(corr_value) > 0.5:  # Show moderate to strong correlations
#                         strong_correlations.append((score1, score2, corr_value))
        
#         if strong_correlations:
#             strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
#             for score1, score2, corr in strong_correlations[:3]:  # Show top 3
#                 print(f"  {score1:15} - {score2:15}: {corr:.3f}")
#         else:
#             print("  No strong correlations found (|r| > 0.5)")
#     else:
#         print("  No valid score data found")

# # %%
# # Create 1x3 plot showing Pearson correlations for all language pairs
# # Find samples with lowest overall CEF scores for each language pair
# lowest_cef_samples = {}

# for lang_pair, dataset in merged_results[f'/data/cef/{DATASET}/{DATASET}_original'].items():
#     # Convert to pandas for easier sorting
#     df = dataset.to_pandas()
    
#     # Sort by overall_cef score (ascending to get lowest first)
#     df_sorted = df.sort_values('overall_cef')
    
#     # Get the 5 lowest scoring samples
#     lowest_samples = df_sorted.head(5)
    
#     lowest_cef_samples[lang_pair] = lowest_samples

# # Also calculate summary statistics for overall CEF scores





# # %%
# # Print the example in a more readable format
# sample = lowest_cef_samples['en-es'].iloc[0]

# # Create markdown formatted output
# markdown_output = f"""# Sample Analysis - English to Spanish

# ## Source Text
# {sample['src']}

# ## Target Text
# {sample['trg']}

# ## Reference Translation
# {sample['reference_translation']}

# ## Metrics
# - **BLEU Score**: {sample['bleu']:.2f}
# - **CHRF Score**: {sample['chrf']:.2f}
# - **BERTScore**: {sample['bertscore']:.3f}
# - **Translation Quality Score**: {sample['translation_quality_score']:.1f}

# ## CEF Scores
# """
# cef_scores = sample['cef_scores']
# for key, value in cef_scores.items():
#     markdown_output += f"- **{key.replace('_', ' ').title()}**: {value:.3f}\n"

# markdown_output += f"""
# ## Quality Metrics
# - **Coverage**: {sample['coverage']:.3f}
# - **Consistency**: {sample['consistency']:.3f}
# - **Conformity**: {sample['conformity']:.3f}
# - **Overall CEF**: {sample['overall_cef']:.3f}

# ## Translation Quality Analysis
# {sample['translation_quality_analysis']}

# ## CEF Details
# """

# cef_details = sample['cef_details']
# for key, value in cef_details.items():
#     if isinstance(value, list):
#         markdown_output += f"### {key.replace('_', ' ').title()}\n"
#         for item in value:
#             if isinstance(item, dict):
#                 for sub_key, sub_value in item.items():
#                     markdown_output += f"- **{sub_key}**: {sub_value}\n"
#             else:
#                 markdown_output += f"- {item}\n"
#     else:
#         markdown_output += f"- **{key.replace('_', ' ').title()}**: {value}\n"

# print(markdown_output)

# # %%
# # Find samples with highest disparity between translation_quality_score and overall_cef
# def find_high_disparity_samples(merged_results, model_key, top_k=10):
#     """
#     Find samples with highest disparity between translation_quality_score and overall_cef
#     """
#     disparities = {}
    
#     for lang_pair, dataset in merged_results[model_key].items():
#         for i, sample in enumerate(dataset):
#             # Skip samples where translation_quality_score is -1
#             if sample['translation_quality_score'] == -1:
#                 continue
                
#             # Calculate absolute difference between scores
#             disparity = sample['translation_quality_score'] - sample['overall_cef']*10
#             disparities[f"{lang_pair}_{i}"] = {
#                 'lang_pair': lang_pair,
#                 'sample_idx': i,
#                 'disparity': disparity,
#                 'translation_quality_score': sample['translation_quality_score'],
#                 'overall_cef': sample['overall_cef'],
#                 'sample': sample
#             }
    
#     # Sort by disparity (highest first)
#     sorted_disparities = sorted(disparities.items(), key=lambda x: x[1]['disparity'], reverse=True)
    
#     return sorted_disparities[:top_k]

# # Find high disparity samples for the specified model
# model_key = f"/data/cef/{DATASET}/{DATASET}_llama3.2-1b_all_languages"
# high_disparity_samples = find_high_disparity_samples(merged_results, model_key, top_k=10)

# # %%
# # Print the first high disparity sample in markdown format
# if high_disparity_samples:
#     sample_info = high_disparity_samples[0][1]
#     sample = sample_info['sample']
    
#     markdown_output = f"""# High Disparity Sample Analysis - {sample_info['lang_pair'].upper()}

# ## Sample Info
# - **Language Pair**: {sample_info['lang_pair']}
# - **Sample Index**: {sample_info['sample_idx']}
# - **Disparity**: {sample_info['disparity']:.2f}
# - **Translation Quality Score**: {sample_info['translation_quality_score']:.1f}
# - **Overall CEF**: {sample_info['overall_cef']:.3f}

# ## Source Text
# {sample['src']}

# ## Target Text
# {sample['trg']}

# ## Reference Translation
# {sample['reference_translation']}

# ## Metrics
# - **BLEU Score**: {sample['bleu']:.2f}
# - **CHRF Score**: {sample['chrf']:.2f}
# - **BERTScore**: {sample['bertscore']:.3f}
# - **Translation Quality Score**: {sample['translation_quality_score']:.1f}

# ## CEF Scores
# """
#     cef_scores = sample['cef_scores']
#     for key, value in cef_scores.items():
#         markdown_output += f"- **{key.replace('_', ' ').title()}**: {value:.3f}\n"

#     markdown_output += f"""
# ## Quality Metrics
# - **Coverage**: {sample['coverage']:.3f}
# - **Consistency**: {sample['consistency']:.3f}
# - **Conformity**: {sample['conformity']:.3f}
# - **Overall CEF**: {sample['overall_cef']:.3f}

# ## Translation Quality Analysis
# {sample['translation_quality_analysis']}

# ## CEF Details
# """

#     cef_details = sample['cef_details']
#     for key, value in cef_details.items():
#         if isinstance(value, list):
#             markdown_output += f"### {key.replace('_', ' ').title()}\n"
#             for item in value:
#                 if isinstance(item, dict):
#                     for sub_key, sub_value in item.items():
#                         markdown_output += f"- **{sub_key}**: {sub_value}\n"
#                 else:
#                     markdown_output += f"- {item}\n"
#         else:
#             markdown_output += f"- **{key.replace('_', ' ').title()}**: {value}\n"

#     print(markdown_output)
# else:
#     print("No high disparity samples found.")

# # %%
# # Find samples with negative conformity, consistency, and coverage scores
# # Iterate over all datasets in merged_results and find samples with negative conformity, consistency, and coverage
# negative_cef_samples = []

# for dataset_path, lang_dict in merged_results.items():
#     for lang_pair, dataset in lang_dict.items():
#         for sample in dataset:
#             if (
#                 sample.get('conformity', 0) < 0 or
#                 sample.get('consistency', 0) < 0 or
#                 sample.get('coverage', 0) < 0
#             ):
#                 negative_cef_samples.append({
#                     'dataset_path': dataset_path,
#                     'lang_pair': lang_pair,
#                     'sample': sample
#                 })

# print(f"Found {len(negative_cef_samples)} samples with negative conformity, consistency, and coverage scores across all datasets.")
# for idx, entry in enumerate(negative_cef_samples[:10]):  # Show up to 10 samples
#     sample = entry['sample']
#     print(f"\nSample {idx+1}:")
#     print(f"Dataset: {entry['dataset_path']}")
#     print(f"Language Pair: {entry['lang_pair']}")
#     print(json.dumps(sample, indent=4))
#     print("\n\n")

# # %%
# # Find samples that have much higher coverage score with qwen2.5-72b than qwen3-0.6b

# # We'll define "much higher" as a difference of at least 0.2 (can adjust as needed)
# COVERAGE_DIFF_THRESHOLD = 0.7

# higher_coverage_samples = []

# # We want to compare coverage scores for the same sample (by id) between
# # llama3.2-1b-instruct and qwen2.5-72b for each language pair and dataset.

# # Get the relevant dataset paths for llama3.2-1b-instruct and qwen2.5-72b
# llama3_path = "/data/cef/ntrex/ntrex_llama3.2-1b_all_languages"
# qwen2_5_path = "/data/cef/ntrex/ntrex_qwen2.5-72b_all_languages"

# # Get the intersection of language pairs available in both
# lang_pairs = set(merged_results[llama3_path].keys()) & set(merged_results[qwen2_5_path].keys())

# for lang_pair in lang_pairs:
#     ds_llama3 = merged_results[llama3_path][lang_pair]
#     ds_2_5 = merged_results[qwen2_5_path][lang_pair]

#     # Build a mapping from id to sample for each dataset
#     id2sample_llama3 = {sample['id']: sample for sample in ds_llama3}
#     id2sample_2_5 = {sample['id']: sample for sample in ds_2_5}

#     # Only consider ids present in both
#     common_ids = set(id2sample_llama3.keys()) & set(id2sample_2_5.keys())

#     for sid in common_ids:
#         sample_llama3 = id2sample_llama3[sid]
#         sample_2_5 = id2sample_2_5[sid]
#         coverage_llama3 = sample_llama3.get('coverage')
#         coverage_2_5 = sample_2_5.get('coverage')
#         if coverage_llama3 is not None and coverage_2_5 is not None:
#             if (coverage_2_5 - coverage_llama3) >= COVERAGE_DIFF_THRESHOLD:
#                 higher_coverage_samples.append({
#                     'dataset_path': llama3_path,
#                     'lang_pair': lang_pair,
#                     'sample_id': sid,
#                     'sample_llama3.2-1b-instruct': sample_llama3,
#                     'sample_qwen2.5-72b': sample_2_5,
#                     'coverage_llama3.2-1b-instruct': coverage_llama3,
#                     'coverage_qwen2.5-72b': coverage_2_5,
#                     'coverage_diff': coverage_2_5 - coverage_llama3
#                 })

# print(f"Found {len(higher_coverage_samples)} samples where qwen2.5-72b coverage is at least {COVERAGE_DIFF_THRESHOLD} higher than llama3.2-1b-instruct.")

# import random
# for idx, entry in enumerate(random.sample(higher_coverage_samples, min(2, len(higher_coverage_samples)))):
#     print(f"\nSample {idx+1}:")
#     print(f"Dataset: {entry['dataset_path']}")
#     print(f"Language Pair: {entry['lang_pair']}")
#     print(f"Sample ID: {entry['sample_id']}")
#     print(f"Coverage llama3.2-1b-instruct: {entry['coverage_llama3.2-1b-instruct']}")
#     print(f"Coverage qwen2.5-72b: {entry['coverage_qwen2.5-72b']}")
#     print(f"Coverage diff: {entry['coverage_diff']}")
#     print("llama3.2-1b-instruct sample:")
#     print(json.dumps(entry['sample_llama3.2-1b-instruct'], indent=4))
#     print("qwen2.5-72b sample:")
#     print(json.dumps(entry['sample_qwen2.5-72b'], indent=4))
#     print("\n\n")
# # %%
# import json
# import os

# # Paths to the latest report.json files
# reports = {
#     'cef': '/data/cef/ntrex/results_cef/4c2e57fb-c847-4220-a62c-d28a68e4144b/report.json',
#     'traditional': '/data/cef/ntrex/results_trad/984b23d5-b52f-438c-8dfa-2ac39c4459af/report.json',
#     'translation_quality': '/data/cef/ntrex/results_tq/7461e2af-7f50-4af9-844b-dbf57774167e/report.json'
# }

# # Load all reports
# loaded_reports = {}
# for name, path in reports.items():
#     with open(path, 'r') as f:
#         loaded_reports[name] = json.load(f)
#     print(f"Loaded {name} report from {path}")

# # Combine reports
# combined_report = {
#     'scores': {},
#     'parameters': {}
# }

# # Get all language pairs
# all_lang_pairs = set()
# for report in loaded_reports.values():
#     if 'scores' in report:
#         all_lang_pairs.update(report['scores'].keys())

# print(f"\nFound language pairs: {sorted(all_lang_pairs)}")

# # Combine scores for each language pair
# for lang_pair in sorted(all_lang_pairs):
#     combined_report['scores'][lang_pair] = {}
    
#     for report_name, report in loaded_reports.items():
#         if 'scores' in report and lang_pair in report['scores']:
#             # Add scores with prefix to distinguish source
#             for score_type, score_data in report['scores'][lang_pair].items():
#                 combined_score_type = f"{score_type}_{report_name}"
#                 combined_report['scores'][lang_pair][combined_score_type] = score_data

# # Combine parameters
# for report_name, report in loaded_reports.items():
#     # Find the parameter key (it varies by report type)
#     param_key = None
#     for key in report.keys():
#         if 'params' in key:
#             param_key = key
#             break
    
#     if param_key:
#         combined_report['parameters'][report_name] = report[param_key]

# # Add metadata
# combined_report['metadata'] = {
#     'combined_from': list(reports.keys()),
#     'total_language_pairs': len(all_lang_pairs),
#     'language_pairs': sorted(all_lang_pairs)
# }

# # Save combined report
# output_path = 'combined_report_july18.json'
# with open(output_path, 'w') as f:
#     json.dump(combined_report, f, indent=2)

# print(f"\nCombined report saved to: {output_path}")

# # Print summary
# print(f"\n{'='*60}")
# print("COMBINED REPORT SUMMARY")
# print(f"{'='*60}")

# for lang_pair in sorted(all_lang_pairs):
#     print(f"\n{lang_pair.upper()}:")
#     print("-" * 40)
    
#     if lang_pair in combined_report['scores']:
#         for score_type, score_data in combined_report['scores'][lang_pair].items():
#             if isinstance(score_data, dict) and 'mean' in score_data:
#                 mean = score_data['mean']
#                 ci = score_data.get('confidence_interval', 'N/A')
#                 print(f"  {score_type:30}: {mean:.3f} ± {ci}")
#             else:
#                 print(f"  {score_type:30}: {score_data}")

# print(f"\nParameters from {len(combined_report['parameters'])} evaluation methods:")
# for method, params in combined_report['parameters'].items():
#     print(f"  {method}: {params.get('data_path', 'N/A')}")
# # %%

# # View the combined report as 3 dataframes, each for a language pair, without confidence interval and sample_size

# import pandas as pd

# # Get all language pairs (limit to 3 for display)
# lang_pairs = sorted(list(all_lang_pairs))[:3]

# for lang_pair in lang_pairs:
#     print(f"\n{'='*40}\nLanguage Pair: {lang_pair.upper()}\n{'='*40}")
#     scores = combined_report['scores'].get(lang_pair, {})
#     rows = []
#     for score_type, score_data in scores.items():
#         if isinstance(score_data, dict):
#             # Exclude confidence_interval and sample_size
#             filtered = {k: v for k, v in score_data.items() if k not in ['confidence_interval', 'sample_size']}
#             # If only mean remains, flatten
#             if set(filtered.keys()) == {'mean'}:
#                 rows.append({'Score Type': score_type, 'Mean': filtered['mean']})
#             else:
#                 row = {'Score Type': score_type}
#                 row.update(filtered)
#                 rows.append(row)
#         else:
#             rows.append({'Score Type': score_type, 'Value': score_data})
#     df = pd.DataFrame(rows)
#     # Round all float columns to two decimal places before displaying
#     float_cols = df.select_dtypes(include=['float', 'float64']).columns
#     df[float_cols] = df[float_cols].round(2)
#     display(df)
# %%
# List folders in results_cef directory from today
import os
from datetime import datetime

results_dir = "/data/cef/ntrex/results_cef_new"
today = ["Aug 22", "Aug 25", "Aug 29", "Sep 01", "Sep 11", "Sep 22"]  # Hardcoded date for July 25th

print(f"\n{'='*40}\nFolders from {today} in {results_dir}:\n{'='*40}")

# Get all folders and their stats
folders = []
for item in os.listdir(results_dir):
    full_path = os.path.join(results_dir, item)
    if os.path.isdir(full_path):  # Check if it's a directory
        stats = os.stat(full_path)
        # Convert timestamp to readable format
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%b %d")
        if mod_time in today:
            folders.append((full_path, stats))

# Print folders sorted by modification time
for folder, stats in sorted(folders, key=lambda x: x[1].st_mtime):
    mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mod_time}  {os.path.basename(folder)}/")



# %%
# Read all reports from today's folders
import json
reports = []
for folder, _ in sorted(folders, key=lambda x: x[1].st_mtime):
    report_path = os.path.join(folder, "report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
            reports.append(report)
    else:
        print(f"Warning: No report.json found in {folder}")

print(f"\nLoaded {len(reports)} reports from today's folders")

# %%
import pandas as pd
# Create a DataFrame from cef_params
cef_params_rows = []
for report in reports:
    params = report['cef_params']
    # Flatten the dictionary and add to rows
    cef_params_rows.append({
        'Data Path': params['data_path'],
        'Judge Model': params['judge_model'],
        'Num Questions': params['num_questions_to_generate'],
        'Token per Question': params['token_per_question'],
        'Bootstrap Iterations': params['bootstrap_iterations'],
        'Bootstrap Fraction': params['bootstrap_fraction']
    })

params_df = pd.DataFrame(cef_params_rows)
display(params_df)

# %%
# Filter reports where num_questions = 10
# Create separate DataFrames for each language pair where each model is a row
print("Filtering reports where num_questions = 10")
filtered_reports = []
for i, params in enumerate(cef_params_rows):
    if params['Num Questions'] == 10:
        filtered_reports.append(reports[i])

print(f"\nFound {len(filtered_reports)} reports with 10 questions")

FIXED_JUDGE = "deepseek-ai/DeepSeek-V3"
print(f"Filtering for judge: {FIXED_JUDGE}")
lang_pairs = ["en-fr", "en-de", "en-es", "en-jp", "en-ar"]
# lang_pairs = ["en-tir", "en-eus", "en-dzo", "en-mri", "en-khm"]
for lang_pair in lang_pairs:
    score_rows = []
    
    # Collect all models and their scores for this language pair and judge
    for report in filtered_reports:
        if (lang_pair in report['scores'] and 
            report['scores'][lang_pair] != {} and
            report['cef_params']['judge_model'] == FIXED_JUDGE):
            scores = report['scores'][lang_pair]
            data_path = report['cef_params']['data_path']
            if "nolist" in data_path:
                model = data_path.split('/')[-1].split('_')[-1] + "_nolist"
            else:
                model = data_path.split('/')[-1].split('_')[-1]
            
            row = {
                'Model': model,
                'Conformity': scores['conformity_score']['mean'],
                'Consistency': scores['consistency_score']['mean'],
                'Coverage': scores['coverage_score']['mean']
            }
            score_rows.append(row)
    
    if score_rows:  # Only display if we have data
        df = pd.DataFrame(score_rows)
        print(f"\nScores for {lang_pair} with judge {FIXED_JUDGE}:")
        display(df)
    else:
        print(f"\nNo data found for {lang_pair} with judge {FIXED_JUDGE}")

# %%
# Filter to show only specific models
target_models = ["qwen3-1.7b", "gemma3-4b", "gpt-oss-20b", "nllb-3.3B", "azure", "google"]
print(f"Filtering for specific models: {target_models}")

# Collect scores for all models and language pairs
scores = {}

for lang_pair in lang_pairs:
    score_rows = []
    
    # Collect all models and their scores for this language pair and judge
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
            
            # Only include if model is in our target list
            if model in target_models:
                row = {
                    'Model': model,
                    'Conformity': scores_data['conformity_score']['mean'],
                    'Consistency': scores_data['consistency_score']['mean'],
                    'Coverage': scores_data['coverage_score']['mean']
                }
                score_rows.append(row)
                
                # Store for heatmap
                if model not in scores:
                    scores[model] = {}
                scores[model][lang_pair] = {
                    'Conformity': scores_data['conformity_score']['mean'],
                    'Consistency': scores_data['consistency_score']['mean'],
                    'Coverage': scores_data['coverage_score']['mean']
                }
    
    if score_rows:  # Only display if we have data
        df = pd.DataFrame(score_rows)
        print(f"\nScores for {lang_pair} with judge {FIXED_JUDGE} (filtered models):")
        display(df)
    else:
        print(f"\nNo data found for {lang_pair} with judge {FIXED_JUDGE} (filtered models)")

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
# Filter to show only specific models

# %%
# Create separate DataFrames for each language pair, filtering for specific judge
print("Filtering reports where num_questions = 10")
filtered_reports = []
for i, params in enumerate(cef_params_rows):
    if params['Num Questions'] == 10:
        filtered_reports.append(reports[i])

print(f"\nFound {len(filtered_reports)} reports with 10 questions")

FIXED_JUDGE = "deepseek-ai/DeepSeek-V3"
print(f"Filtering for judge: {FIXED_JUDGE}")
lang_pairs = ["en-fr", "en-es", "en-jp", "en-ar"]
# lang_pairs = ["en-tir", "en-eus", "en-dzo", "en-mri", "en-khm"]
for lang_pair in lang_pairs:
    score_rows = []
    
    # Collect all models and their scores for this language pair and judge
    for report in filtered_reports:
        if (lang_pair in report['scores'] and 
            report['scores'][lang_pair] != {} and
            report['cef_params']['judge_model'] == FIXED_JUDGE):
            scores = report['scores'][lang_pair]
            data_path = report['cef_params']['data_path']
            if "nolist" in data_path:
                model = data_path.split('/')[-1].split('_')[-1] + "_nolist"
            else:
                model = data_path.split('/')[-1].split('_')[-1]
            
            row = {
                'Model': model,
                'Conformity': scores['conformity_score']['mean'],
                'Consistency': scores['consistency_score']['mean'],
                'Coverage': scores['coverage_score']['mean']
            }
            score_rows.append(row)
    
    if score_rows:  # Only display if we have data
        df = pd.DataFrame(score_rows)
        print(f"\nScores for {lang_pair} with judge {FIXED_JUDGE}:")
        display(df)
    else:
        print(f"\nNo data found for {lang_pair} with judge {FIXED_JUDGE}")
 
# %%
# Load and analyze results_ds for Qwen3-0.6B with different judges at num_questions=10
# %%
# List folders in results_cef directory from today
import os
from datetime import datetime

results_dir = "/data/cef/ntrex/results_cef/"
today = "Jul 25"  # Hardcoded date for July 25th

print(f"\n{'='*40}\nFolders from {today} in {results_dir}:\n{'='*40}")

# Get all folders and their stats
folders = []
for item in os.listdir(results_dir):
    full_path = os.path.join(results_dir, item)
    if os.path.isdir(full_path):  # Check if it's a directory
        stats = os.stat(full_path)
        # Convert timestamp to readable format
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%b %d")
        if mod_time == today:
            folders.append((full_path, stats))


# Print folders sorted by modification time
for folder, stats in sorted(folders, key=lambda x: x[1].st_mtime):
    mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mod_time}  {os.path.basename(folder)}/")


# %%
# Load reports from folders and filter for desired parameters
filtered_reports = []
filtered_folders = []
MODEL = "qwen3-0.6b"
for folder, _ in folders:
    report_path = os.path.join(folder, "report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
            # if (report['cef_params']['num_questions_to_generate'] == 10 and
            #     report['cef_params']['data_path'] == f'/data/cef/ntrex/ntrex_{MODEL}'):
            if (report['cef_params']['num_questions_to_generate'] >= 10 and
                report['cef_params']['data_path'] == f'/data/cef/ntrex/ntrex_{MODEL}' and 
                report['cef_params']['judge_model'] == "/models_llm/Qwen2.5-72B-Instruct"):
                filtered_reports.append(report)
                filtered_folders.append(folder)

print(f"\nFound {len(filtered_reports)} matching reports")

# Load and combine all result_ds datasets
all_results = {}
for report, folder in zip(filtered_reports, filtered_folders):
    result_ds_path = os.path.join(folder, 'results_ds')
    if os.path.exists(result_ds_path):
        ds = load_from_disk(result_ds_path)
        judge_model = report['cef_params']['judge_model']
        num_questions = report['cef_params']['num_questions_to_generate']
        print(f"\nLoaded dataset for judge {judge_model} with {len(ds)} examples")
        
        # Add judge model info to each example
        ds = ds.map(lambda x: {'conformity_score': x["cef_scores"]["conformity_score"] if "cef_scores" in x else None,
                               'consistency_score': x["cef_scores"]["consistency_score"] if "cef_scores" in x else None,
                               'coverage_score': x["cef_scores"]["coverage_score"] if "cef_scores" in x else None})
        
        # all_results[judge_model] = ds
        all_results[num_questions] = ds


# %%
# Extract conformity scores for each judge
languages = ["en-fr", "en-de", "en-es", "en-jp"]
score_type = "coverage_score"
score_type_name = "Coverage"

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.ravel()  # Flatten axes array for easier indexing

for idx, language in enumerate(languages):
    print(f"\nAnalyzing language pair: {language}")
    
    judge_scores = {}
    for judge, ds in all_results.items():
        print(f"Processing judge {judge} for language {language}")
        # Get conformity scores, filtering out None values
        scores = [x for x in ds[language][score_type] if x is not None]
        if scores:  # Only include if there are valid scores
            judge_scores[judge] = scores

    # Calculate correlations between all pairs of judges
    judges = list(judge_scores.keys())
    n_judges = len(judges)

    # Create correlation matrix
    corr_matrix = np.zeros((n_judges, n_judges))
    for i in range(n_judges):
        for j in range(n_judges):
            # Get scores for both judges
            scores1 = judge_scores[judges[i]]
            scores2 = judge_scores[judges[j]]
            
            # Calculate correlation if we have enough data points
            if len(scores1) > 1 and len(scores2) > 1:
                rmse = np.sqrt(np.mean((np.array(scores1) - np.array(scores2)) ** 2))
                corr_matrix[i,j] = rmse
    print(corr_matrix)
    
    # Plot correlation matrix in current subplot
    im = axes[idx].imshow(corr_matrix, cmap='Blues', vmin=0, vmax=0.3)
    fig.colorbar(im, ax=axes[idx], label='Correlation')

    # Add judge names as labels
    axes[idx].set_xticks(range(n_judges))
    axes[idx].set_xticklabels(judges, rotation=45, ha='right')
    axes[idx].set_yticks(range(n_judges))
    axes[idx].set_yticklabels(judges)

    # Add correlation values in boxes
    for i in range(n_judges):
        for j in range(n_judges):
            text = f'{corr_matrix[i, j]:.2f}'
            axes[idx].text(j, i, text, ha='center', va='center')

    axes[idx].set_title(f'RMSE of {score_type_name} Scores Between Judges ({language})')

    # Print average correlation for each judge with others
    print(f"\nRMSE of {score_type_name} with other judges for {language}:")
    for i, judge in enumerate(judges):
        # Exclude self-correlation (1.0)
        correlations = [c for j,c in enumerate(corr_matrix[i]) if i != j]
        avg_corr = np.mean(correlations)
        print(f"{judge}: {avg_corr:.3f} {score_type_name}")

plt.tight_layout()
plt.show()


# %%
# Create dot plot / strip plot for en-fr coverage_score across different judges
import matplotlib.pyplot as plt
import numpy as np

# Extract coverage scores for en-fr from all judges
language_pair = 'en-jp'
score_type = 'coverage_score'

# Collect data for plotting
judges = []
all_scores = []
colors = []
x_positions = []

# Define colors for different judges
judge_colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))
color_map = {}

for judge_idx, (judge, dataset_dict) in enumerate(all_results.items()):
    if language_pair in dataset_dict:
        dataset = dataset_dict[language_pair]
        scores = dataset[score_type]
        
        # Store judge name and color mapping
        # judge_name = judge.split('/')[-1] if '/' in judge else judge
        judge_name = judge
        color_map[judge_name] = judge_colors[judge_idx]
        
        # Create x positions for this judge's data points
        n_points = len(scores)
        x_pos = np.arange(n_points)
        
        judges.extend([judge_name] * n_points)
        all_scores.extend(scores)
        x_positions.extend(x_pos)
        colors.extend([judge_colors[judge_idx]] * n_points)

# Create the plot
plt.figure(figsize=(20, 8))

# Plot points for each judge
for judge_name in color_map.keys():
    judge_mask = [j == judge_name for j in judges]
    judge_x = [x for x, mask in zip(x_positions, judge_mask) if mask]
    judge_scores = [s for s, mask in zip(all_scores, judge_mask) if mask]
    
    plt.scatter(judge_x, judge_scores, 
               c=color_map[judge_name], 
               label=judge_name, 
               alpha=0.6, 
               s=30)

plt.xlabel('Data Point Index')
plt.ylabel('Coverage Score')
plt.title(f'Coverage Scores by Judge for {language_pair.upper()}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\nSummary statistics for {language_pair} {score_type}:")
for judge_name in color_map.keys():
    judge_mask = [j == judge_name for j in judges]
    judge_scores = [s for s, mask in zip(all_scores, judge_mask) if mask]
    
    if judge_scores:
        print(f"{judge_name}:")
        print(f"  Mean: {np.mean(judge_scores):.3f}")
        print(f"  Std:  {np.std(judge_scores):.3f}")
        print(f"  Min:  {np.min(judge_scores):.3f}")
        print(f"  Max:  {np.max(judge_scores):.3f}")
        print(f"  N:    {len(judge_scores)}")

# %%


























# %%
# List folders in results_cef directory from today
import os
from datetime import datetime

results_dir = "/data/cef/ntrex/results_cef_numquestionss/"
today = ["Sep 04"]  # Hardcoded date for July 25th

print(f"\n{'='*40}\nFolders from {today} in {results_dir}:\n{'='*40}")

# Get all folders and their stats
folders = []
for item in os.listdir(results_dir):
    full_path = os.path.join(results_dir, item)
    if os.path.isdir(full_path):  # Check if it's a directory
        stats = os.stat(full_path)
        # Convert timestamp to readable format
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%b %d")
        if mod_time in today:
            folders.append((full_path, stats))

# Print folders sorted by modification time
for folder, stats in sorted(folders, key=lambda x: x[1].st_mtime):
    mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mod_time}  {os.path.basename(folder)}/")

# %%
filtered_reports = []
filtered_folders = []
MODEL = "qwen3-1.7b"
for folder, _ in folders:
    report_path = os.path.join(folder, "report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
            # if (report['cef_params']['num_questions_to_generate'] == 10 and
            #     report['cef_params']['data_path'] == f'/data/cef/ntrex/ntrex_{MODEL}'):
            if (report['cef_params']['num_questions_to_generate'] >= 3):
                filtered_reports.append(report)
                filtered_folders.append(folder)

print(f"\nFound {len(filtered_reports)} matching reports")

# %%
from collections import defaultdict
all_results = defaultdict(list)
for report, folder in zip(filtered_reports, filtered_folders):
    # Skip folders that end with 'debug'
    if folder.endswith('debug'):
        continue
    result_ds_path = os.path.join(folder, 'results_ds')
    if os.path.exists(result_ds_path):
        ds = load_from_disk(result_ds_path)
        judge_model = report['cef_params']['judge_model']
        num_questions = report['cef_params']['num_questions_to_generate']
        model = report['cef_params']['data_path'].split('/')[-1].split('_')[-1]
        print(f"\nLoaded dataset for judge {judge_model} with {len(ds)} examples")
        
        # Add judge model info to each example
        ds = ds.map(lambda x: {'conformity_score': x["cef_scores"]["conformity_score"] if "cef_scores" in x else None,
                               'consistency_score': x["cef_scores"]["consistency_score"] if "cef_scores" in x else None,
                               'coverage_score': x["cef_scores"]["coverage_score"] if "cef_scores" in x else None})
        
        # all_results[judge_model] = ds
        all_results[str(num_questions) + "_" + model].append(ds)

# %%
# Calculate per-sample variance for en-jp across all datasets
language_pair = "en-jp"

# Initialize dictionaries to store results by model and num_questions
variance_results = {}

# Process all datasets
for dataset_key in all_results:
    datasets = all_results[dataset_key]
    
    # Parse the dataset key to get num_questions and model
    parts = dataset_key.split("_")
    num_questions = parts[0]
    model = "_".join(parts[1:])
    
    # Check if all datasets have the language pair
    if all(language_pair in ds for ds in datasets):
        print(f"\nCalculating per-sample variance for {language_pair} in {dataset_key}:")
        
        # Get the number of samples (assuming all datasets have same samples)
        num_samples = len(datasets[0][language_pair])
        
        # Initialize lists to store variances for each score type
        conformity_variances = []
        consistency_variances = []
        coverage_variances = []
        
        # Calculate variance for each sample across all datasets
        for sample_idx in range(num_samples):
            # Collect scores for this sample across all datasets
            conformity_scores = []
            consistency_scores = []
            coverage_scores = []
            
            for ds in datasets:
                sample = ds[language_pair][sample_idx]
                if sample['conformity_score'] is not None:
                    conformity_scores.append(sample['conformity_score'] * 100)
                if sample['consistency_score'] is not None:
                    consistency_scores.append(sample['consistency_score'] * 100)
                if sample['coverage_score'] is not None:
                    coverage_scores.append(sample['coverage_score'] * 100)
            
            # Calculate variance for each score type
            if len(conformity_scores) > 1:
                conformity_variances.append(np.var(conformity_scores))
            if len(consistency_scores) > 1:
                consistency_variances.append(np.var(consistency_scores))
            if len(coverage_scores) > 1:
                coverage_variances.append(np.var(coverage_scores))
        
        # Store results
        if model not in variance_results:
            variance_results[model] = {}
        
        variance_results[model][num_questions] = {
            'conformity_std': np.std(conformity_variances),
            'consistency_std': np.std(consistency_variances),
            'coverage_std': np.std(coverage_variances)
        }
        
        print(f"Conformity score variance - Std: {np.std(conformity_variances):.4f}")
        print(f"Consistency score variance - Std: {np.std(consistency_variances):.4f}")
        print(f"Coverage score variance - Std: {np.std(coverage_variances):.4f}")
    else:
        print(f"Not all datasets in {dataset_key} contain {language_pair}")

# Create tables for each model
import pandas as pd

for model in variance_results:
    print(f"\n{model.upper()} - Variance Standard Deviation for {language_pair}")
    print("="*50)
    
    # Prepare data for the table
    table_data = []
    for num_questions in sorted(variance_results[model].keys(), key=int):
        results = variance_results[model][num_questions]
        table_data.append({
            'Num Questions': num_questions,
            'Conformity': f"{results['conformity_std']:.4f}",
            'Consistency': f"{results['consistency_std']:.4f}",
            'Coverage': f"{results['coverage_std']:.4f}"
        })
    
    # Create and display the table
    df = pd.DataFrame(table_data)
    df = df.set_index('Num Questions')
    print(df.to_string()) 
    print()












# %%
# %%
# List folders in results_cef directory from today
import os
from datetime import datetime

results_dir = "/data/cef/ntrex/results_cef_roundtrip/"
today = ["Sep 01", "Sep 22", "Sep 23"]  # Hardcoded date for July 25th

print(f"\n{'='*40}\nFolders from {today} in {results_dir}:\n{'='*40}")

# Get all folders and their stats
folders = []
for item in os.listdir(results_dir):
    full_path = os.path.join(results_dir, item)
    if os.path.isdir(full_path):  # Check if it's a directory
        stats = os.stat(full_path)
        # Convert timestamp to readable format
        mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%b %d")
        if mod_time in today:
            folders.append((full_path, stats))

# Print folders sorted by modification time
for folder, stats in sorted(folders, key=lambda x: x[1].st_mtime):
    mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mod_time}  {os.path.basename(folder)}/")


# %%
filtered_reports = []
filtered_folders = []
MODEL = "original"
for folder, _ in folders:
    report_path = os.path.join(folder, "report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
            # if (report['cef_params']['num_questions_to_generate'] == 10 and
            #     report['cef_params']['data_path'] == f'/data/cef/ntrex/ntrex_{MODEL}'):
            if (report['cef_params']['num_questions_to_generate'] >= 3):
                filtered_reports.append(report)
                filtered_folders.append(folder)

print(f"\nFound {len(filtered_reports)} matching reports")


# %%
import json
reports = []
for folder, _ in sorted(folders, key=lambda x: x[1].st_mtime):
    report_path = os.path.join(folder, "report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
            reports.append(report)
    else:
        print(f"Warning: No report.json found in {folder}")

print(f"\nLoaded {len(reports)} reports from today's folders")

# %%
import pandas as pd
# Create a DataFrame from cef_params
cef_params_rows = []
for report in reports:
    params = report['cef_params']
    # Flatten the dictionary and add to rows
    cef_params_rows.append({
        'Data Path': params['data_path'],
        'Judge Model': params['judge_model'],
        'Num Questions': params['num_questions_to_generate'],
        'Token per Question': params['token_per_question'],
        'Bootstrap Iterations': params['bootstrap_iterations'],
        'Bootstrap Fraction': params['bootstrap_fraction']
    })

params_df = pd.DataFrame(cef_params_rows)
display(params_df)

# %%
# Filter reports where num_questions = 10
# Create separate DataFrames for each language pair where each model is a row
print("Filtering reports where num_questions = 10")
filtered_reports = []
for i, params in enumerate(cef_params_rows):
    if params['Num Questions'] == 10:
        filtered_reports.append(reports[i])

print(f"\nFound {len(filtered_reports)} reports with 10 questions")

FIXED_JUDGE = "deepseek-ai/DeepSeek-V3"
print(f"Filtering for judge: {FIXED_JUDGE}")
# lang_pairs = ["en-fr", "en-de", "en-es", "en-jp", "en-ar"]
lang_pairs = ["en-tir", "en-eus", "en-dzo", "en-mri", "en-khm"]
for lang_pair in lang_pairs:
    score_rows = []
    
    # Collect all models and their scores for this language pair and judge
    for report in filtered_reports:
        if (lang_pair in report['scores'] and 
            report['scores'][lang_pair] != {} and
            report['cef_params']['judge_model'] == FIXED_JUDGE):
            scores = report['scores'][lang_pair]
            data_path = report['cef_params']['data_path']
            if "nolist" in data_path:
                model = data_path.split('/')[-1].split('_')[-1] + "_nolist"
            else:
                model = data_path.split('/')[-1].split('_')[-1]
            
            row = {
                'Model': model,
                'Conformity': scores['conformity_score']['mean'],
                'Consistency': scores['consistency_score']['mean'],
                'Coverage': scores['coverage_score']['mean']
            }
            score_rows.append(row)
    
    if score_rows:  # Only display if we have data
        df = pd.DataFrame(score_rows)
        print(f"\nScores for {lang_pair} with judge {FIXED_JUDGE}:")
        display(df)
    else:
        print(f"\nNo data found for {lang_pair} with judge {FIXED_JUDGE}")

# %%
# Create separate DataFrames for each language pair, filtering for specific judge
FIXED_JUDGE = "/models_llm/Qwen2.5-72B-Instruct"
print(f"filtering for specific judge {FIXED_JUDGE}")
for lang_pair in lang_pairs:
    # First collect all unique data paths
    data_paths = set()
    for report in reports:
        if (lang_pair in report['scores'] and 
            report['cef_params']['judge_model'] == FIXED_JUDGE):
            data_path = report['cef_params']['data_path']
            model_name = data_path.split('/')[-1].split('_')[-1]  # Extract model name
            data_paths.add(model_name)
    
    # Create a row for each num_questions value
    score_rows = []
    num_questions_values = set(report['cef_params']['num_questions_to_generate'] 
                             for report in reports 
                             if report['cef_params']['judge_model'] == FIXED_JUDGE)
    
    for num_q in num_questions_values:
        row = {'Num Questions': num_q}
        # Initialize scores for all models as None
        for model in data_paths:
            row[f'Conformity_{model}'] = None
            row[f'Consistency_{model}'] = None
            row[f'Coverage_{model}'] = None
        
        # Fill in scores where we have them
        for report in reports:
            if (lang_pair in report['scores'] and 
                report['scores'][lang_pair] != {} and
                report['cef_params']['judge_model'] == FIXED_JUDGE and
                report['cef_params']['num_questions_to_generate'] == num_q):
                scores = report['scores'][lang_pair]
                data_path = report['cef_params']['data_path']
                if "no_list" in data_path:
                    model = data_path.split('/')[-1].split('_')[-1] + "_nolist"
                else:
                    model = data_path.split('/')[-1].split('_')[-1]
                
                row[f'Conformity_{model}'] = scores['conformity_score']['mean']
                row[f'Consistency_{model}'] = scores['consistency_score']['mean']
                row[f'Coverage_{model}'] = scores['coverage_score']['mean']
        
        score_rows.append(row)
    
    df = pd.DataFrame(score_rows)
    print(f"\nScores for {lang_pair} with judge {FIXED_JUDGE}:")
    display(df)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Prepare Data
models = ['gpt-oss-20b', 'llama3.1-8b', 'llama3.2-3b', 'qwen3-1.7b', 'qwen3-4b']

# Note Generation Data
notes_data = {
    'Conformity': [97.58, 96.42, 94.92, 94.50, 97.83],
    'Consistency': [91.50, 95.83, 93.42, 90.92, 95.83],
    'Coverage': [97.33, 92.00, 86.08, 92.92, 96.83]
}
df_notes = pd.DataFrame(notes_data, index=models)

# Summarization Data
summ_data = {
    'Conformity': [94.68, 95.84, 95.43, 94.76, 94.90],
    'Consistency': [99.21, 99.36, 99.15, 98.62, 99.15],
    'Coverage': [59.36, 56.99, 55.28, 41.89, 45.62],
    'Conciseness': [17.31, 24.06, 25.06, 17.26, 16.37]
}
df_summ = pd.DataFrame(summ_data, index=models)

# 2. Plotting Setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 4]})
cmap = "rocket" # Closest match to your image's color scheme

# Common heatmap arguments
heatmap_args = {
    'annot': True,
    'fmt': ".1f",
    'cmap': cmap,
    'cbar': False,
    'annot_kws': {"size": 11},
    'linewidths': 0.5,
    'linecolor': 'black'
}

# 3. Create Heatmaps
sns.heatmap(df_notes, ax=ax1, **heatmap_args)
sns.heatmap(df_summ, ax=ax2, **heatmap_args)

# 4. Styling and Labels
ax1.set_title('Note Generation', fontsize=14, pad=15)
ax2.set_title('Summarization', fontsize=14, pad=15)

for ax in [ax1, ax2]:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    # Adding the thick borders to match your image style
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

# 5. Colorbar (Manual addition for the whole figure)
mappable = ax1.collections[0]
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
cb = fig.colorbar(mappable, cax=cbar_ax)
cb.set_label('Score', fontsize=14)

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Input data (means only) ---
# Note generation
note_data = {
    "coverage": [97.33, 92.00, 86.08, 92.92, 96.83],
    "conformity": [97.58, 96.42, 94.92, 94.50, 97.83],
    "consistency": [91.50, 95.83, 93.42, 90.92, 95.83],
}
# Summarization
summ_data = {
    "coverage": [59.36, 56.99, 55.28, 41.89, 45.62],
    "conformity": [94.68, 95.84, 95.43, 94.76, 94.90],
    "consistency": [99.21, 99.36, 99.15, 98.62, 99.15],
    "conciseness": [17.31, 24.06, 25.06, 17.26, 16.37],
}

models = ["gpt-oss-20b", "llama3.1-8b", "llama3.2-3b", "qwen3-1.7b", "qwen3-4b"]

df_note = pd.DataFrame(note_data, index=models)
df_summ = pd.DataFrame(summ_data, index=models)

# Concatenate horizontally so we can draw one heatmap with a thick separator
df_all = pd.concat([df_note, df_summ], axis=1)

# Create friendly x-axis labels with task grouping
note_cols = ["Coverage", "Conformity", "Consistency"]
summ_cols = ["Coverage", "Conformity", "Consistency", "Conciseness"]
xlabels = [f"Note\n{c}" for c in note_cols] + [f"Summ\n{c}" for c in summ_cols]

# --- Plotting ---
sns.set(style="white")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
})

fig, ax = plt.subplots(figsize=(18, 6))

# Choose a colormap similar to the original (light-peach -> dark-purple).
# Use 'magma' reversed so high values become light-peach and low values dark-purple-like.
cmap = plt.cm.magma_r

# Plot heatmap across all columns but fix vmin/vmax to [0, 100] to match a percent scale
sns.heatmap(
    df_all,
    ax=ax,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 12, "weight": "bold"},
    linewidths=0.5,
    linecolor="white",
    cmap=cmap,
    cbar=True,
    vmin=0, vmax=100,
    xticklabels=xlabels,
    yticklabels=models,
)

# Improve layout: rotate x labels and align
plt.xticks(rotation=35, ha="right")

# Add a thicker vertical separator between Note and Summ blocks
n_note = df_note.shape[1]  # =3
# Position the separator after the last note column. Heatmap columns are 0..(N-1) so a vertical
# line at x = n_note will fall between blocks.
ax.vlines(n_note, *ax.get_ylim(), colors="black", linewidth=4)

# Cosmetic tweaks: bold model names on y-axis
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

# Colorbar label
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Score", fontsize=14)

# Title
ax.set_title("Note generation (left) and Summarization (right) — Coverage/Conformity/Consistency (and Conciseness for Summ)")

plt.tight_layout()
plt.savefig("task_heatmaps.png", dpi=300)
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Prepare and Join Data
models = ['qwen3-1.7b', 'gpt-oss-20b', 'llama3.1-8b', 'llama3.2-3b', 'qwen3-4b']

# Note Generation Data
notes_data = {
    'Conformity': [94.50, 97.58, 96.42, 94.92, 97.83],
    'Consistency': [90.92, 91.50, 95.83, 93.42, 95.83],
    'Coverage': [92.92, 97.33, 92.00, 86.08, 96.83]
}

# Summarization Data
summ_data = {
    'Conformity ': [94.76, 94.68, 95.84, 95.43, 94.90], # Space added to avoid column name collision
    'Consistency ': [98.62, 99.21, 99.36, 99.15, 99.15],
    'Coverage ': [41.89, 59.36, 56.99, 55.28, 45.62],
    'Conciseness': [17.26, 17.31, 24.06, 25.06, 16.37]
}

df_notes = pd.DataFrame(notes_data, index=models)
df_summ = pd.DataFrame(summ_data, index=models)

# Concatenate dataframes to join the plots
df_combined = pd.concat([df_notes, df_summ], axis=1)

# 2. Plotting
plt.figure(figsize=(18, 6))
cmap = "rocket"

# To make the colorbar consistent, we set vmin and vmax based on the full dataset
# Or set them manually to 0-100 or 15-100 depending on your preference
ax = sns.heatmap(df_combined, 
                 annot=True, 
                 fmt=".1f", 
                 cmap=cmap, 
                 vmin=df_combined.values.min(), 
                 vmax=100, 
                 linewidths=0.5, 
                 linecolor='black',
                 cbar_kws={'label': 'Score'})

# 3. Aesthetics & Task Headers
# Add a vertical line to separate 'Note Generation' from 'Summarization'
ax.axvline(3, color='black', lw=4)

# Set labels and ticks
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Customizing the Task Labels at the top
ax.text(1.5, -0.2, 'Note Generation', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(5, -0.2, 'Summarization', ha='center', va='center', fontsize=14, fontweight='bold')

# Thick outer borders
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Prepare and Join Data
# Re-ordering models to match the target image's Y-axis order
models = ['qwen3-1.7b', 'gpt-oss-20b', 'llama3.1-8b', 'llama3.2-3b', 'qwen3-4b']

# Note Generation Data (Re-ordered to match new model index)
notes_data = {
    'Conformity': [94.50, 97.58, 96.42, 94.92, 97.83],
    'Consistency': [90.92, 91.50, 95.83, 93.42, 95.83],
    'Coverage': [92.92, 97.33, 92.00, 86.08, 96.83]
}

# Summarization Data (Re-ordered to match new model index)
# Note: Added a trailing space to column names to ensure they are treated as distinct
# from the Note Generation columns during concatenation.
summ_data = {
    'Conformity ': [94.76, 94.68, 95.84, 95.43, 94.90],
    'Consistency ': [98.62, 99.21, 99.36, 99.15, 99.15],
    'Coverage ': [41.89, 59.36, 56.99, 55.28, 45.62],
    'Conciseness': [17.26, 17.31, 24.06, 25.06, 16.37]
}

df_notes = pd.DataFrame(notes_data, index=models)
df_summ = pd.DataFrame(summ_data, index=models)

# Concatenate dataframes to join the plots horizontally
df_combined = pd.concat([df_notes, df_summ], axis=1)

# 2. Plotting setup
# Reduced figsize to make boxes feel smaller and tighter
plt.figure(figsize=(15, 5))
cmap = "rocket"

# Calculate global min/max for consistent colorbar across both sections
vmin_val = df_combined.values.min()
vmax_val = 100

# Create Heatmap
ax = sns.heatmap(df_combined,
                 annot=True,
                 fmt=".1f",
                 cmap=cmap,
                 vmin=vmin_val,
                 vmax=vmax_val,
                 linewidths=0.5,   # Thin black borders between cells (like reference)
                 linecolor='black',
                 # Increased font size for numbers inside boxes for clarity
                 annot_kws={"size": 13},
                 cbar_kws={'label': 'Score'})

# 3. Aesthetics & Styling

# Customize Colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('Score', size=14, weight='bold')
cbar.ax.tick_params(labelsize=12)

# Add the thick vertical separator line between tasks
ax.axvline(3, color='black', lw=3)

# Style X and Y ticks for clarity (larger font, rotated X-ticks)
plt.xticks(rotation=45, ha='right', fontsize=13, color='black')
plt.yticks(rotation=0, fontsize=13, color='black')

# Remove standard axis labels as they are redundant
ax.set_xlabel('')
ax.set_ylabel('')

# Add custom Task Headers at the top
# Adjusted y-coordinate (-0.25) to place text above the plot area
ax.text(1.5, -0.25, 'Note Generation', ha='center', va='center', fontsize=16, fontweight='bold', color='black')
ax.text(5, -0.25, 'Summarization', ha='center', va='center', fontsize=16, fontweight='bold', color='black')

# Add thick outer border to the entire plot frame
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color('black')

# Ensure no background grid is visible
ax.grid(False)

# Final layout adjustment to accommodate the top headers and rotated x-ticks
plt.tight_layout()
# Add a little extra space at the top for the manual headers if needed
plt.subplots_adjust(top=0.9)

plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Data Preparation
models = ['qwen3-1.7b', 'gpt-oss-20b', 'llama3.1-8b', 'llama3.2-3b', 'qwen3-4b']

notes_data = {
    'Conformity': [94.50, 97.58, 96.42, 94.92, 97.83],
    'Consistency': [90.92, 91.50, 95.83, 93.42, 95.83],
    'Coverage': [92.92, 97.33, 92.00, 86.08, 96.83]
}

summ_data = {
    'Conformity ': [94.76, 94.68, 95.84, 95.43, 94.90],
    'Consistency ': [98.62, 99.21, 99.36, 99.15, 99.15],
    'Coverage ': [41.89, 59.36, 56.99, 55.28, 45.62],
    'Conciseness': [17.26, 17.31, 24.06, 25.06, 16.37]
}

df_notes = pd.DataFrame(notes_data, index=models)
df_summ = pd.DataFrame(summ_data, index=models)
df_combined = pd.concat([df_notes, df_summ], axis=1)

# 2. Plotting
# Adjusted figsize to be narrower (12) and taller (7) relative to the columns
plt.figure(figsize=(12, 7)) 

ax = sns.heatmap(df_combined,
                 annot=True,
                 fmt=".1f",
                 cmap="rocket",
                 vmin=df_combined.values.min(),
                 vmax=100,
                 square=True,          # This forces each cell to be a perfect square
                 linewidths=1.0,       # Clearer cell boundaries
                 linecolor='black',
                 annot_kws={"size": 12, "weight": "bold"}, # Bold text inside boxes
                 cbar_kws={'label': 'Score', 'shrink': 0.7}) # Shrink cbar to match square height

# 3. Customization & Formatting
# Bold vertical separator between tasks
ax.axvline(3, color='black', lw=4)

# Formatting Ticks (Removed grid-like clutter, focused on labels)
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold', color='black')
plt.yticks(rotation=0, fontsize=12, fontweight='bold', color='black')
ax.set_xlabel('')
ax.set_ylabel('')

# Task Headers (positioned slightly higher for clarity)
ax.text(1.5, -0.3, 'Note Generation', ha='center', fontsize=14, fontweight='bold')
ax.text(5.0, -0.3, 'Summarization', ha='center', fontsize=14, fontweight='bold')

# Thick Outer Border
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2.5)

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Prepare Data
models = ['qwen3-1.7b', 'gpt-oss-20b', 'llama3.1-8b', 'llama3.2-3b', 'qwen3-4b']

notes_data = {
    'Conformity': [94.50, 97.58, 96.42, 94.92, 97.83],
    'Consistency': [90.92, 91.50, 95.83, 93.42, 95.83],
    'Coverage': [92.92, 97.33, 92.00, 86.08, 96.83]
}

summ_data = {
    'Conformity ': [94.76, 94.68, 95.84, 95.43, 94.90],
    'Consistency ': [98.62, 99.21, 99.36, 99.15, 99.15],
    'Coverage ': [41.89, 59.36, 56.99, 55.28, 45.62],
    'Conciseness': [17.26, 17.31, 24.06, 25.06, 16.37]
}

df_combined = pd.concat([pd.DataFrame(notes_data, index=models), 
                         pd.DataFrame(summ_data, index=models)], axis=1)

# 2. Plotting
# Width adjusted to 10 to keep it compact/thin
fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(df_combined,
            annot=True,
            fmt=".1f",
            cmap="rocket",
            vmin=df_combined.values.min(),
            vmax=100,
            square=True,
            cbar_kws={'label': 'Score', 'shrink': 0.8},
            # REMOVING GRIDS: Set linewidth to 0
            linewidths=0, 
            annot_kws={"size": 11, "weight": "normal"})

# 3. Reference Style Customization

# Add the thick black vertical separator to match the "en-fr" style
ax.axvline(3, color='black', lw=3)

# Formatting Ticks to match the right-side image exactly
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

# Remove axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Task Headers (Centered above their respective sections)
ax.text(1.5, -0.1, 'Note Generation', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.text(5.0, -0.1, 'Summarization', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Thick outer frame only (No internal grid lines)
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Data Preparation
models = ['qwen3-1.7b', 'gpt-oss-20b', 'llama3.1-8b', 'llama3.2-3b', 'qwen3-4b']

notes_data = {
    'Conformity': [94.50, 97.58, 96.42, 94.92, 97.83],
    'Consistency': [90.92, 91.50, 95.83, 93.42, 95.83],
    'Coverage': [92.92, 97.33, 92.00, 86.08, 96.83]
}

summ_data = {
    'Conformity ': [94.76, 94.68, 95.84, 95.43, 94.90],
    'Consistency ': [98.62, 99.21, 99.36, 99.15, 99.15],
    'Coverage ': [41.89, 59.36, 56.99, 55.28, 45.62],
    'Conciseness': [100-17.26, 100-17.31, 100-24.06, 100-25.06, 100-16.37]
}

df_notes = pd.DataFrame(notes_data, index=models)
df_summ = pd.DataFrame(summ_data, index=models)
df_combined = pd.concat([df_notes, df_summ], axis=1)

# 2. Plotting
# Narrower figsize to prevent horizontal stretching
fig, ax = plt.subplots(figsize=(12, 5))

sns.heatmap(df_combined,
            annot=True,
            fmt=".1f",
            cmap="rocket",
            vmin=df_combined.values.min(),
            vmax=100,
            square=True,
            # cbar_kws={'label': 'Score', 'shrink': 0.8},
            cbar=False,
            linewidths=0, # Completely removes internal grid lines
            annot_kws={"size": 11})

# 3. Adding Ticks and Style Adjustments
# Explicitly turn on the ticks (the small lines)
ax.tick_params(left=True, bottom=True, which='major', length=5, width=1, color='black')

# Formatting Labels
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

# Remove redundant axis titles
ax.set_xlabel('')
ax.set_ylabel('')

# Task Headers
ax.text(1.5, -0.15, 'Note Generation', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.text(5.0, -0.15, 'Summarization', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Vertical divider between tasks
ax.axvline(3, color='black', lw=1.5)

# Outer border frame
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.savefig('plots/cef_note_summ.pdf', bbox_inches='tight')
plt.show()
# %%
# 2. Plotting (matched to right figure)
