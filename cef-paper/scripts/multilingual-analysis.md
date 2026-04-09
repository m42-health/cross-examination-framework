============================================================
MULTILINGUAL EUROPARL ANALYSIS
============================================================

==================== Processing EN-FR ====================

Loading en-fr dataset from: /data/cef/europarl/en-fr
Loading Arrow dataset from: /data/cef/europarl/en-fr/data-00000-of-00001.arrow
Dataset shape: (87990, 3)
Columns: ['id', 'src', 'trg']
Mapped columns: {'id': 'id', 'src': 'src', 'trg': 'trg'}
Final dataset shape: (87990, 3)
Loaded 87990 samples for en-fr

Calculating token counts for en-fr...
Processing 87990 sentence pairs...
Token count statistics for en-fr:
  Source (EN) tokens - Mean: 334.60, Std: 339.55
  Target tokens - Mean: 348.69, Std: 354.22
  Total tokens - Mean: 683.30, Std: 692.44

95% Confidence Interval for en-fr total_tokens:
  Mean: 683.30
  Lower bound: 678.72
  Upper bound: 687.87
  Samples within 95% CI: 571 out of 87990 (0.65%)
Warning: en-fr dataset has only 571 samples, less than requested 1000
Sampled 571 samples from en-fr
Results for en-fr saved to: multilingual_europarl_analysis/sampled_en-fr
✓ EN-FR processing completed

==================== Processing EN-ES ====================

Loading en-es dataset from: /data/cef/europarl/en-es
Loading Arrow dataset from: /data/cef/europarl/en-es/data-00000-of-00001.arrow
Dataset shape: (86836, 3)
Columns: ['id', 'src', 'trg']
Mapped columns: {'id': 'id', 'src': 'src', 'trg': 'trg'}
Final dataset shape: (86836, 3)
Loaded 86836 samples for en-es

Calculating token counts for en-es...
Processing 86836 sentence pairs...
Token count statistics for en-es:
  Source (EN) tokens - Mean: 334.89, Std: 340.07
  Target tokens - Mean: 348.09, Std: 353.52
  Total tokens - Mean: 682.98, Std: 692.49

95% Confidence Interval for en-es total_tokens:
  Mean: 682.98
  Lower bound: 678.38
  Upper bound: 687.59
  Samples within 95% CI: 557 out of 86836 (0.64%)
Warning: en-es dataset has only 557 samples, less than requested 1000
Sampled 557 samples from en-es
Results for en-es saved to: multilingual_europarl_analysis/sampled_en-es
✓ EN-ES processing completed

==================== Processing EN-DE ====================

Loading en-de dataset from: /data/cef/europarl/en-de
Loading Arrow dataset from: /data/cef/europarl/en-de/data-00000-of-00001.arrow
Dataset shape: (87478, 3)
Columns: ['id', 'src', 'trg']
Mapped columns: {'id': 'id', 'src': 'src', 'trg': 'trg'}
Final dataset shape: (87478, 3)
Loaded 87478 samples for en-de

Calculating token counts for en-de...
Processing 87478 sentence pairs...
Token count statistics for en-de:
  Source (EN) tokens - Mean: 334.77, Std: 339.92
  Target tokens - Mean: 318.14, Std: 323.00
  Total tokens - Mean: 652.91, Std: 661.34

95% Confidence Interval for en-de total_tokens:
  Mean: 652.91
  Lower bound: 648.53
  Upper bound: 657.29
  Samples within 95% CI: 650 out of 87478 (0.74%)
Warning: en-de dataset has only 650 samples, less than requested 1000
Sampled 650 samples from en-de
Results for en-de saved to: multilingual_europarl_analysis/sampled_en-de
✓ EN-DE processing completed

==================== Creating Comparative Plots ====================
/home/nsaadi/cef-translation/src/cef_framework/CI-calc.py:619: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  box_plot = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
Distribution plots saved to: multilingual_europarl_analysis/token_distribution_comparison.png
Summary statistics saved to: multilingual_europarl_analysis/summary_statistics.csv
  Language Pair Target Language  Samples EN Mean EN Std FR Mean FR Std Total Mean Total Std ES Mean ES Std DE Mean DE Std
0         EN-FR              FR      571   332.6   19.2   350.2   19.2      682.8       2.5     NaN    NaN     NaN    NaN
1         EN-ES              ES      557   334.7   12.8     NaN    NaN      683.1       2.5   348.4   12.8     NaN    NaN
2         EN-DE              DE      650   335.5   14.7     NaN    NaN      653.1       2.6     NaN    NaN   317.6   14.7

==================== Analysis Complete ====================
Results saved to: ./multilingual_europarl_analysis
Languages processed: ['en-fr', 'en-es', 'en-de']
Total samples per language: 1000