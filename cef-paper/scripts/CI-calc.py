# import pandas as pd
# import numpy as np
# import random
# from pathlib import Path
# import nltk
# from nltk.tokenize import word_tokenize
# from scipy import stats
# import json
# from datasets import Dataset

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# def count_tokens(text):
#     """Count tokens in a text string using NLTK word tokenizer"""
#     if pd.isna(text) or text == '' or text is None:
#         return 0
#     return len(word_tokenize(str(text)))

# def load_arrow_dataset(data_path):
#     """Load dataset from Hugging Face datasets cache directory"""
#     try:
#         data_path = Path(data_path)
        
#         # Look for the main data file
#         data_file = data_path / "data-00000-of-00001.arrow"
        
#         if data_file.exists():
#             print(f"Loading Arrow dataset from: {data_file}")
#             # Load using datasets library
#             dataset = Dataset.from_file(str(data_file))
#             df = dataset.to_pandas()
#             print(f"Successfully loaded dataset with shape: {df.shape}")
#             print(f"Columns: {list(df.columns)}")
            
#         else:
#             # Try to find any data arrow files
#             data_files = list(data_path.glob("data-*.arrow"))
#             if data_files:
#                 print(f"Found data files: {[f.name for f in data_files]}")
#                 # Load the first data file
#                 dataset = Dataset.from_file(str(data_files[0]))
#                 df = dataset.to_pandas()
#                 print(f"Successfully loaded dataset with shape: {df.shape}")
#                 print(f"Columns: {list(df.columns)}")
#             else:
#                 raise FileNotFoundError(f"No data-*.arrow files found in {data_path}")
        
#         # Check if we have the expected columns
#         print(f"Dataset columns: {list(df.columns)}")
        
#         # Check for common column names that might represent id, src, trg
#         column_mapping = {}
        
#         # Look for ID column
#         id_candidates = ['id', 'index', 'idx', '__index_level_0__']
#         for col in id_candidates:
#             if col in df.columns:
#                 column_mapping['id'] = col
#                 break
        
#         # If no ID column found, create one
#         if 'id' not in column_mapping:
#             df.reset_index(inplace=True)
#             if 'index' in df.columns:
#                 column_mapping['id'] = 'index'
#             else:
#                 df['id'] = range(len(df))
#                 column_mapping['id'] = 'id'
        
#         # Look for source and target columns
#         src_candidates = ['src', 'source', 'en', 'english', 'text_en']
#         trg_candidates = ['trg', 'target', 'fr', 'french', 'text_fr']
        
#         for col in src_candidates:
#             if col in df.columns:
#                 column_mapping['src'] = col
#                 break
        
#         for col in trg_candidates:
#             if col in df.columns:
#                 column_mapping['trg'] = col
#                 break
        
#         # Rename columns if needed
#         if len(column_mapping) < 3:
#             print("Available columns and sample data:")
#             for col in df.columns:
#                 print(f"  {col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
            
#             # Manual mapping - you might need to adjust this based on actual column names
#             if 'translation' in df.columns:
#                 # Handle nested translation column (common in HF datasets)
#                 translation_data = df['translation'].iloc[0]
#                 if isinstance(translation_data, dict):
#                     src_key = 'en' if 'en' in translation_data else list(translation_data.keys())[0]
#                     trg_key = 'fr' if 'fr' in translation_data else list(translation_data.keys())[1]
                    
#                     df['src'] = df['translation'].apply(lambda x: x.get(src_key, '') if isinstance(x, dict) else '')
#                     df['trg'] = df['translation'].apply(lambda x: x.get(trg_key, '') if isinstance(x, dict) else '')
#                     column_mapping['src'] = 'src'
#                     column_mapping['trg'] = 'trg'
        
#         # Ensure we have the required columns
#         required_cols = ['id', 'src', 'trg']
#         if not all(col in column_mapping for col in required_cols):
#             missing_cols = [col for col in required_cols if col not in column_mapping]
#             raise ValueError(f"Could not find columns for: {missing_cols}. Available: {list(df.columns)}")
        
#         # Create final dataframe with standardized column names
#         final_df = pd.DataFrame()
#         final_df['id'] = df[column_mapping['id']]
#         final_df['src'] = df[column_mapping['src']]
#         final_df['trg'] = df[column_mapping['trg']]
        
#         print(f"Mapped columns: {column_mapping}")
#         print(f"Final dataset shape: {final_df.shape}")
#         print(f"Sample data:")
#         print(f"  ID: {final_df['id'].iloc[0]}")
#         print(f"  SRC: {final_df['src'].iloc[0][:100]}...")
#         print(f"  TRG: {final_df['trg'].iloc[0][:100]}...")
        
#         return final_df
        
#     except Exception as e:
#         print(f"Error loading Arrow dataset: {e}")
#         # Fallback: try to load using pandas directly
#         try:
#             print("Trying fallback method...")
#             data_files = list(Path(data_path).glob("*.arrow"))
#             if data_files:
#                 # Try loading with pyarrow directly
#                 import pyarrow as pa
#                 import pyarrow.parquet as pq
                
#                 table = pa.ipc.open_file(str(data_files[0])).read_all()
#                 df = table.to_pandas()
#                 print(f"Fallback successful. Shape: {df.shape}")
#                 return df
#         except Exception as fallback_error:
#             print(f"Fallback also failed: {fallback_error}")
        
#         raise

# def calculate_token_counts(df):
#     """Calculate token counts for src and trg columns"""
#     print("Calculating token counts...")
    
#     # Calculate token counts
#     print("Processing source texts...")
#     df['src_tokens'] = df['src'].apply(count_tokens)
    
#     print("Processing target texts...")
#     df['trg_tokens'] = df['trg'].apply(count_tokens)
    
#     df['total_tokens'] = df['src_tokens'] + df['trg_tokens']
    
#     print(f"Token count statistics:")
#     print(f"Source tokens - Mean: {df['src_tokens'].mean():.2f}, Std: {df['src_tokens'].std():.2f}")
#     print(f"Target tokens - Mean: {df['trg_tokens'].mean():.2f}, Std: {df['trg_tokens'].std():.2f}")
#     print(f"Total tokens - Mean: {df['total_tokens'].mean():.2f}, Std: {df['total_tokens'].std():.2f}")
    
#     return df

# def get_95_confidence_interval_samples(df, token_column='total_tokens'):
#     """Get samples within 95% confidence interval based on token counts"""
    
#     # Calculate 95% confidence interval
#     mean_tokens = df[token_column].mean()
#     std_tokens = df[token_column].std()
#     n = len(df)
    
#     # 95% confidence interval using t-distribution
#     confidence_level = 0.95
#     alpha = 1 - confidence_level
#     t_score = stats.t.ppf(1 - alpha/2, n-1)
    
#     margin_of_error = t_score * (std_tokens / np.sqrt(n))
    
#     lower_bound = mean_tokens - margin_of_error
#     upper_bound = mean_tokens + margin_of_error
    
#     print(f"\n95% Confidence Interval for {token_column}:")
#     print(f"Mean: {mean_tokens:.2f}")
#     print(f"Lower bound: {lower_bound:.2f}")
#     print(f"Upper bound: {upper_bound:.2f}")
    
#     # Filter samples within confidence interval
#     ci_samples = df[(df[token_column] >= lower_bound) & (df[token_column] <= upper_bound)]
    
#     print(f"Samples within 95% CI: {len(ci_samples)} out of {len(df)} ({len(ci_samples)/len(df)*100:.2f}%)")
    
#     return ci_samples, lower_bound, upper_bound

# def sample_random_subset(df, n_samples=1000, random_seed=42):
#     """Sample random subset of specified size"""
    
#     random.seed(random_seed)
#     np.random.seed(random_seed)
    
#     if len(df) < n_samples:
#         print(f"Warning: Dataset has only {len(df)} samples, less than requested {n_samples}")
#         sampled_df = df.copy()
#     else:
#         sampled_df = df.sample(n=n_samples, random_state=random_seed)
    
#     # Using set for efficient ID tracking (O(1) lookup vs O(n) for list)
#     sampled_ids = set(sampled_df['id'].tolist())
    
#     print(f"Sampled {len(sampled_df)} samples")
#     print(f"Sample IDs stored in set of size: {len(sampled_ids)}")
    
#     return sampled_df, sampled_ids

# def save_results(sampled_df, sampled_ids, output_dir, ci_bounds):
#     """Save the sampled dataset and metadata"""
    
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # Save the sampled dataset in the same format (CSV for compatibility)
#     sampled_data_path = output_path / 'sampled_data.csv'
#     sampled_df.to_csv(sampled_data_path, index=False)
#     print(f"Sampled dataset saved to: {sampled_data_path}")
    
#     # Also save as Arrow format to match original
#     try:
#         from datasets import Dataset
#         dataset = Dataset.from_pandas(sampled_df)
#         arrow_path = output_path / 'sampled_data.arrow'
#         dataset.save_to_disk(str(arrow_path))
#         print(f"Sampled dataset also saved as Arrow format to: {arrow_path}")
#     except Exception as e:
#         print(f"Could not save Arrow format: {e}")
    
#     # Save the IDs as a JSON file (sets are not directly JSON serializable)
#     ids_path = output_path / 'sampled_ids.json'
#     with open(ids_path, 'w') as f:
#         json.dump(list(sampled_ids), f)
#     print(f"Sampled IDs saved to: {ids_path}")
    
#     # Save metadata
#     metadata = {
#         'total_samples': len(sampled_df),
#         'confidence_interval_bounds': {
#             'lower': ci_bounds[0],
#             'upper': ci_bounds[1]
#         },
#         'token_statistics': {
#             'mean_src_tokens': float(sampled_df['src_tokens'].mean()),
#             'mean_trg_tokens': float(sampled_df['trg_tokens'].mean()),
#             'mean_total_tokens': float(sampled_df['total_tokens'].mean()),
#             'std_total_tokens': float(sampled_df['total_tokens'].std())
#         }
#     }
    
#     metadata_path = output_path / 'metadata.json'
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=2)
#     print(f"Metadata saved to: {metadata_path}")

# def main():
#     """Main function to execute the entire pipeline"""
    
#     # Configuration
#     data_path = "/data/cef/europarl/en-fr"
#     output_dir = "./sampled_fr_europarl"
#     n_samples = 1000
#     random_seed = 42
    
#     try:
#         # Load dataset
#         print("Loading Arrow dataset...")
#         df = load_arrow_dataset(data_path)
#         print(f"Loaded dataset with {len(df)} samples")
        
#         # Calculate token counts
#         df = calculate_token_counts(df)
        
#         # Get samples within 95% confidence interval
#         ci_samples, lower_bound, upper_bound = get_95_confidence_interval_samples(df)
        
#         # Sample random subset
#         sampled_df, sampled_ids = sample_random_subset(ci_samples, n_samples, random_seed)
        
#         # Save results
#         save_results(sampled_df, sampled_ids, output_dir, (lower_bound, upper_bound))
        
#         print(f"\nProcess completed successfully!")
#         print(f"Final sample size: {len(sampled_df)}")
#         print(f"Output directory: {output_dir}")
        
#     except Exception as e:
#         print(f"Error in main process: {e}")
#         raise

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import random
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from scipy import stats
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def count_tokens(text):
    """Count tokens in a text string using NLTK word tokenizer"""
    if pd.isna(text) or text == '' or text is None:
        return 0
    return len(word_tokenize(str(text)))

def load_arrow_dataset(data_path, language_pair):
    """Load dataset from Hugging Face datasets cache directory"""
    try:
        data_path = Path(data_path)
        print(f"\nLoading {language_pair} dataset from: {data_path}")
        
        # Look for the main data file
        data_file = data_path / "data-00000-of-00001.arrow"
        
        if data_file.exists():
            print(f"Loading Arrow dataset from: {data_file}")
            dataset = Dataset.from_file(str(data_file))
            df = dataset.to_pandas()
            
        else:
            # Try to find any data arrow files
            data_files = list(data_path.glob("data-*.arrow"))
            if data_files:
                print(f"Found data files: {[f.name for f in data_files]}")
                dataset = Dataset.from_file(str(data_files[0]))
                df = dataset.to_pandas()
            else:
                raise FileNotFoundError(f"No data-*.arrow files found in {data_path}")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Determine target language from language pair
        target_lang = language_pair.split('-')[1]  # e.g., 'fr' from 'en-fr'
        
        # Check for common column names
        column_mapping = {}
        
        # Look for ID column
        id_candidates = ['id', 'index', 'idx', '__index_level_0__']
        for col in id_candidates:
            if col in df.columns:
                column_mapping['id'] = col
                break
        
        # If no ID column found, create one
        if 'id' not in column_mapping:
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                column_mapping['id'] = 'index'
            else:
                df['id'] = range(len(df))
                column_mapping['id'] = 'id'
        
        # Look for source and target columns
        src_candidates = ['src', 'source', 'en', 'english', 'text_en']
        trg_candidates = ['trg', 'target', target_lang, f'text_{target_lang}']
        
        for col in src_candidates:
            if col in df.columns:
                column_mapping['src'] = col
                break
        
        for col in trg_candidates:
            if col in df.columns:
                column_mapping['trg'] = col
                break
        
        # Handle nested translation column
        if 'translation' in df.columns and ('src' not in column_mapping or 'trg' not in column_mapping):
            translation_data = df['translation'].iloc[0]
            if isinstance(translation_data, dict):
                src_key = 'en' if 'en' in translation_data else list(translation_data.keys())[0]
                trg_key = target_lang if target_lang in translation_data else [k for k in translation_data.keys() if k != src_key][0]
                
                df['src'] = df['translation'].apply(lambda x: x.get(src_key, '') if isinstance(x, dict) else '')
                df['trg'] = df['translation'].apply(lambda x: x.get(trg_key, '') if isinstance(x, dict) else '')
                column_mapping['src'] = 'src'
                column_mapping['trg'] = 'trg'
        
        # Create final dataframe with standardized column names
        final_df = pd.DataFrame()
        final_df['id'] = df[column_mapping['id']]
        final_df['src'] = df[column_mapping['src']]
        final_df['trg'] = df[column_mapping['trg']]
        
        print(f"Mapped columns: {column_mapping}")
        print(f"Final dataset shape: {final_df.shape}")
        
        return final_df
        
    except Exception as e:
        print(f"Error loading {language_pair} dataset: {e}")
        raise

def calculate_token_counts(df, language_pair):
    """Calculate token counts for src and trg columns"""
    print(f"\nCalculating token counts for {language_pair}...")
    
    # Calculate token counts with progress indication
    total_rows = len(df)
    print(f"Processing {total_rows} sentence pairs...")
    
    df['src_tokens'] = df['src'].apply(count_tokens)
    df['trg_tokens'] = df['trg'].apply(count_tokens)
    df['total_tokens'] = df['src_tokens'] + df['trg_tokens']
    
    print(f"Token count statistics for {language_pair}:")
    print(f"  Source (EN) tokens - Mean: {df['src_tokens'].mean():.2f}, Std: {df['src_tokens'].std():.2f}")
    print(f"  Target tokens - Mean: {df['trg_tokens'].mean():.2f}, Std: {df['trg_tokens'].std():.2f}")
    print(f"  Total tokens - Mean: {df['total_tokens'].mean():.2f}, Std: {df['total_tokens'].std():.2f}")
    
    return df

def get_95_confidence_interval_samples(df, language_pair, token_column='total_tokens'):
    """Get samples within 95% confidence interval based on token counts"""
    
    # Calculate 95% confidence interval
    mean_tokens = df[token_column].mean()
    std_tokens = df[token_column].std()
    n = len(df)
    
    # 95% confidence interval using t-distribution
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_score = stats.t.ppf(1 - alpha/2, n-1)
    
    margin_of_error = t_score * (std_tokens / np.sqrt(n))
    
    lower_bound = mean_tokens - margin_of_error
    upper_bound = mean_tokens + margin_of_error
    
    print(f"\n95% Confidence Interval for {language_pair} {token_column}:")
    print(f"  Mean: {mean_tokens:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    
    # Filter samples within confidence interval
    ci_samples = df[(df[token_column] >= lower_bound) & (df[token_column] <= upper_bound)]
    
    print(f"  Samples within 95% CI: {len(ci_samples)} out of {len(df)} ({len(ci_samples)/len(df)*100:.2f}%)")
    
    return ci_samples, lower_bound, upper_bound

def sample_random_subset(df, language_pair, n_samples=1000, random_seed=42):
    """Sample random subset of specified size"""
    
    if len(df) < n_samples:
        print(f"Warning: {language_pair} dataset has only {len(df)} samples, less than requested {n_samples}")
        sampled_df = df.copy()
    else:
        sampled_df = df.sample(n=n_samples, random_state=random_seed)
    
    # Using set for efficient ID tracking
    sampled_ids = set(sampled_df['id'].tolist())
    
    print(f"Sampled {len(sampled_df)} samples from {language_pair}")
    
    return sampled_df, sampled_ids

def save_language_results(sampled_df, sampled_ids, language_pair, base_output_dir, ci_bounds):
    """Save results for a specific language pair"""
    
    output_path = Path(base_output_dir) / f"sampled_{language_pair}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the sampled dataset
    sampled_data_path = output_path / f'sampled_{language_pair}_data.csv'
    sampled_df.to_csv(sampled_data_path, index=False)
    
    # Save the IDs
    ids_path = output_path / f'sampled_{language_pair}_ids.json'
    with open(ids_path, 'w') as f:
        json.dump(list(sampled_ids), f)
    
    # Save metadata
    metadata = {
        'language_pair': language_pair,
        'total_samples': len(sampled_df),
        'confidence_interval_bounds': {
            'lower': ci_bounds[0],
            'upper': ci_bounds[1]
        },
        'token_statistics': {
            'mean_src_tokens': float(sampled_df['src_tokens'].mean()),
            'mean_trg_tokens': float(sampled_df['trg_tokens'].mean()),
            'mean_total_tokens': float(sampled_df['total_tokens'].mean()),
            'std_total_tokens': float(sampled_df['total_tokens'].std())
        }
    }
    
    metadata_path = output_path / f'metadata_{language_pair}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results for {language_pair} saved to: {output_path}")
    return metadata

def plot_token_distributions(all_data, output_dir):
    """Create comprehensive plots comparing token distributions across languages"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Token Length Distribution Analysis: English-French, English-Spanish, English-German', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    plot_data = []
    for lang_pair, data in all_data.items():
        df = data['sampled_df']
        target_lang = lang_pair.split('-')[1].upper()
        
        # Add data for source (English) tokens
        for _, row in df.iterrows():
            plot_data.append({
                'language_pair': lang_pair,
                'target_language': target_lang,
                'token_type': 'English (Source)',
                'token_count': row['src_tokens']
            })
        
        # Add data for target language tokens
        for _, row in df.iterrows():
            plot_data.append({
                'language_pair': lang_pair,
                'target_language': target_lang,
                'token_type': f'{target_lang} (Target)',
                'token_count': row['trg_tokens']
            })
        
        # Add data for total tokens
        for _, row in df.iterrows():
            plot_data.append({
                'language_pair': lang_pair,
                'target_language': target_lang,
                'token_type': 'Total',
                'token_count': row['total_tokens']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 1. Distribution of source (English) tokens across language pairs
    ax1 = axes[0, 0]
    src_data = plot_df[plot_df['token_type'] == 'English (Source)']
    for lang_pair in ['en-fr', 'en-es', 'en-de']:
        if lang_pair in all_data:
            data = src_data[src_data['language_pair'] == lang_pair]['token_count']
            ax1.hist(data, alpha=0.6, label=f'{lang_pair.upper()}', bins=30, density=True)
    ax1.set_title('English (Source) Token Distribution')
    ax1.set_xlabel('Token Count')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of target language tokens
    ax2 = axes[0, 1]
    target_data = plot_df[plot_df['token_type'].str.contains('Target')]
    for lang_pair in ['en-fr', 'en-es', 'en-de']:
        if lang_pair in all_data:
            target_lang = lang_pair.split('-')[1].upper()
            data = target_data[target_data['language_pair'] == lang_pair]['token_count']
            ax2.hist(data, alpha=0.6, label=f'{target_lang}', bins=30, density=True)
    ax2.set_title('Target Language Token Distribution')
    ax2.set_xlabel('Token Count')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Total tokens distribution
    ax3 = axes[1, 0]
    total_data = plot_df[plot_df['token_type'] == 'Total']
    for lang_pair in ['en-fr', 'en-es', 'en-de']:
        if lang_pair in all_data:
            data = total_data[total_data['language_pair'] == lang_pair]['token_count']
            ax3.hist(data, alpha=0.6, label=f'{lang_pair.upper()}', bins=30, density=True)
    ax3.set_title('Total Token Distribution')
    ax3.set_xlabel('Token Count')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot comparison
    ax4 = axes[1, 1]
    box_data = []
    box_labels = []
    for lang_pair in ['en-fr', 'en-es', 'en-de']:
        if lang_pair in all_data:
            df = all_data[lang_pair]['sampled_df']
            box_data.extend([df['src_tokens'].values, df['trg_tokens'].values, df['total_tokens'].values])
            target_lang = lang_pair.split('-')[1].upper()
            box_labels.extend([f'EN\n({lang_pair})', f'{target_lang}\n({lang_pair})', f'Total\n({lang_pair})'])
    
    box_plot = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen'] * 3
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax4.set_title('Token Count Comparison (Box Plot)')
    ax4.set_ylabel('Token Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / 'token_distribution_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'token_distribution_comparison.pdf', bbox_inches='tight')
    
    print(f"Distribution plots saved to: {plot_path}")
    
    # Create summary statistics table
    summary_stats = []
    for lang_pair, data in all_data.items():
        df = data['sampled_df']
        target_lang = lang_pair.split('-')[1].upper()
        
        summary_stats.append({
            'Language Pair': lang_pair.upper(),
            'Target Language': target_lang,
            'Samples': len(df),
            'EN Mean': f"{df['src_tokens'].mean():.1f}",
            'EN Std': f"{df['src_tokens'].std():.1f}",
            f'{target_lang} Mean': f"{df['trg_tokens'].mean():.1f}",
            f'{target_lang} Std': f"{df['trg_tokens'].std():.1f}",
            'Total Mean': f"{df['total_tokens'].mean():.1f}",
            'Total Std': f"{df['total_tokens'].std():.1f}"
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = output_path / 'summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    plt.show()
    
    return summary_df

def main():
    """Main function to process all three language pairs"""
    
    # Configuration
    language_pairs = {
        'en-fr': '/data/cef/europarl/en-fr',
        'en-es': '/data/cef/europarl/en-es', 
        'en-de': '/data/cef/europarl/en-de'
    }
    
    base_output_dir = "./multilingual_europarl_analysis"
    n_samples = 1000
    random_seed = 42
    
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    all_data = {}
    all_metadata = {}
    
    print("="*60)
    print("MULTILINGUAL EUROPARL ANALYSIS")
    print("="*60)
    
    # Process each language pair
    for lang_pair, data_path in language_pairs.items():
        try:
            print(f"\n{'='*20} Processing {lang_pair.upper()} {'='*20}")
            
            # Load dataset
            df = load_arrow_dataset(data_path, lang_pair)
            print(f"Loaded {len(df)} samples for {lang_pair}")
            
            # Calculate token counts
            df = calculate_token_counts(df, lang_pair)
            
            # Get 95% confidence interval samples
            ci_samples, lower_bound, upper_bound = get_95_confidence_interval_samples(df, lang_pair)
            
            # Sample random subset
            sampled_df, sampled_ids = sample_random_subset(ci_samples, lang_pair, n_samples, random_seed)
            
            # Save results
            metadata = save_language_results(sampled_df, sampled_ids, lang_pair, base_output_dir, (lower_bound, upper_bound))
            
            # Store data for plotting
            all_data[lang_pair] = {
                'original_df': df,
                'ci_samples': ci_samples,
                'sampled_df': sampled_df,
                'sampled_ids': sampled_ids,
                'ci_bounds': (lower_bound, upper_bound)
            }
            all_metadata[lang_pair] = metadata
            
            print(f"✓ {lang_pair.upper()} processing completed")
            
        except Exception as e:
            print(f"✗ Error processing {lang_pair}: {e}")
            continue
    
    # Create comparative plots
    if all_data:
        print(f"\n{'='*20} Creating Comparative Plots {'='*20}")
        summary_df = plot_token_distributions(all_data, base_output_dir)
        print(summary_df)
        
        # Save combined metadata
        combined_metadata_path = Path(base_output_dir) / 'combined_analysis_metadata.json'
        with open(combined_metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"\n{'='*20} Analysis Complete {'='*20}")
        print(f"Results saved to: {base_output_dir}")
        print(f"Languages processed: {list(all_data.keys())}")
        print(f"Total samples per language: {n_samples}")
    else:
        print("No data was successfully processed!")

if __name__ == "__main__":
    main()