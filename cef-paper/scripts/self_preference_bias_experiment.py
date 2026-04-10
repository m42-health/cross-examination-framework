#!/usr/bin/env python3
"""
Self-Preference Bias Experiment: CEF vs LLM-as-a-Judge
======================================================

This script tests whether CEF (Cross-Examination Framework) mitigates 
self-preference bias compared to traditional LLM-as-a-judge evaluation.

Experiment Design (2x2x2):
- 2 Translator Models: DeepSeek-V3.1, GPT-OSS-120B
- 2 Judge Models: DeepSeek-V3.1, GPT-OSS-120B  
- 2 Language Pairs: en-fr, en-jp
- 2 Evaluation Methods: CEF (QA-decomposed), LLM-as-a-Judge (holistic)

For each (translator, language_pair, sample):
  - self_eval:  score when judge == translator
  - cross_eval: score when judge != translator
  - bias = self_eval - cross_eval

Hypothesis:
  - LLM-as-a-Judge: significant self-preference bias (models favor own translations)
  - CEF: no significant self-preference bias (decomposed QA mitigates it)

Usage:
  # First start CEF server (in a separate terminal):
  cd src/cef_framework && gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8893 --workers 4

  # Then run experiment:
  python scripts/self_preference_bias_experiment.py
  python scripts/self_preference_bias_experiment.py --debug  # use 30 samples for testing
  python scripts/self_preference_bias_experiment.py --phase translate  # run only translation
  python scripts/self_preference_bias_experiment.py --phase analyze    # run only analysis (after other phases)
"""

import os
import sys
import json
import copy
import time
import logging
import requests
import subprocess
import signal
import atexit
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats as scipy_stats
from jinja2 import Template
from jsonfinder import jsonfinder
import fire

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CEF_DIR = SRC_DIR / "cef_framework"
EVAL_DIR = SRC_DIR / "evaluation"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EVAL_DIR))

from datasets import Dataset, DatasetDict, load_from_disk
import yaml

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MODELS = {
    "deepseek": {
        "name": "deepseek-ai/DeepSeek-V3.1",
        "worker": "worker-1",
        "port": 8892,
    },
    "gptoss": {
        "name": "openai/gpt-oss-120b",
        "worker": "worker-4",
        "port": 8892,
    }
}

LANG_PAIRS = ["en-fr", "en-jp"]
LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "jp": "Japanese",
}

DATA_PATH = "/data/cef/ntrex/ntrex_original"
RESULTS_BASE = "/data/cef/ntrex/self_preference_experiment"

CEF_SERVER_PORT = 8893
CEF_PIPELINE_PARAMS_PATH = str(CEF_DIR / "pipeline_params.yaml")
CEF_PROMPT_CATALOGUE_PATH = "prompts/translation.yaml"  # Relative to CEF server CWD (src/cef_framework/)
JUDGE_TEMPLATE_PATH = str(EVAL_DIR / "translation-quality2.jinja")

NUM_QUESTIONS = 10
REQUEST_TIMEOUT = 300
TRANSLATION_NUM_PROC = 16
CEF_NUM_PROC = 8
JUDGE_NUM_PROC = 16  # Parallel API calls; vLLM/sglang handle concurrent requests well


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: TRANSLATION
# ═══════════════════════════════════════════════════════════════════════

def _translate_single(example, api_url, model_name, source_lang_name, target_lang_name):
    """Translate a single example via API (pickle-friendly standalone function)."""
    messages = [
        {"role": "system", "content": f"You are a professional translator. Translate the following {source_lang_name} text to {target_lang_name}. Provide only the translation. Do not include explanations or apologies."},
        {"role": "user", "content": example["src"]}
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    for attempt in range(5):
        try:
            response = requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return {"trg": content.strip()}
        except Exception as e:
            logger.warning(f"Translation attempt {attempt+1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return {"trg": "ERROR: TRANSLATION_FAILED"}


def run_translation(ntrex_ds, results_path, debug=False):
    """Phase 1: Translate NTREX dataset with both models for en-fr and en-jp."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Generating translations with both models")
    logger.info("=" * 60)

    translations = {}  # {(model_key, lang_pair): Dataset}

    for model_key, model_info in MODELS.items():
        api_url = f"http://{model_info['worker']}:{model_info['port']}/v1/chat/completions"
        model_name = model_info["name"]

        for lang_pair in LANG_PAIRS:
            src_code, trg_code = lang_pair.split("-")
            src_name = LANG_NAMES[src_code]
            trg_name = LANG_NAMES[trg_code]

            cache_path = os.path.join(results_path, f"translations_{model_key}_{lang_pair}")
            if os.path.exists(cache_path):
                logger.info(f"  Loading cached translations: {model_key}/{lang_pair}")
                translations[(model_key, lang_pair)] = load_from_disk(cache_path)
                continue

            logger.info(f"  Translating {lang_pair} with {model_key} ({model_name})...")
            source_ds = ntrex_ds[lang_pair]
            if debug:
                source_ds = source_ds.select(range(min(30, len(source_ds))))

            translated_ds = source_ds.map(
                lambda x: _translate_single(x, api_url, model_name, src_name, trg_name),
                num_proc=TRANSLATION_NUM_PROC,
                load_from_cache_file=False,
                desc=f"Translate {lang_pair} with {model_key}"
            )

            translated_ds.save_to_disk(cache_path)
            translations[(model_key, lang_pair)] = translated_ds
            logger.info(f"  Saved {len(translated_ds)} translations to {cache_path}")

    return translations


def load_translations(results_path):
    """Load cached translations from disk."""
    translations = {}
    for model_key in MODELS:
        for lang_pair in LANG_PAIRS:
            cache_path = os.path.join(results_path, f"translations_{model_key}_{lang_pair}")
            if os.path.exists(cache_path):
                translations[(model_key, lang_pair)] = load_from_disk(cache_path)
            else:
                raise FileNotFoundError(f"Translations not found: {cache_path}. Run --phase translate first.")
    return translations


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: LLM-AS-A-JUDGE EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def _judge_parse_scores(response_text):
    """Parse 3-score JSON from LLM-as-a-judge response."""
    json_datas = list(jsonfinder(response_text))
    for tmp in json_datas:
        json_data = tmp[2]
        if json_data is None or not isinstance(json_data, dict):
            continue
        if "scores" in json_data and "analysis" in json_data:
            scores = json_data["scores"]
            if isinstance(scores, dict) and all(k in scores for k in ["coverage", "conformity", "consistency"]):
                return {
                    "judge_coverage": float(scores["coverage"]),
                    "judge_conformity": float(scores["conformity"]),
                    "judge_consistency": float(scores["consistency"]),
                }
    raise ValueError(f"No valid scores found in response")


def _judge_evaluate_single(example, api_url, model_name, template_source, lang_pair):
    """Evaluate a single translation using LLM-as-a-judge (standalone function)."""
    template = Template(template_source)
    src_code, trg_code = lang_pair.split("-")
    src_name = LANG_NAMES[src_code]
    trg_name = LANG_NAMES[trg_code]

    rendered = template.render(
        source_language=src_name,
        target_language=trg_name,
        source=example["src"],
        translation=example["trg"],
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful multilingual assistant."},
            {"role": "user", "content": rendered}
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }

    for attempt in range(5):
        try:
            response = requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _judge_parse_scores(content)
        except Exception as e:
            logger.warning(f"Judge attempt {attempt+1} failed: {e}")
            time.sleep(2 * (attempt + 1))

    return {
        "judge_coverage": -1.0,
        "judge_conformity": -1.0,
        "judge_consistency": -1.0,
    }


def run_judge_evaluation(translations, results_path):
    """Phase 2: Evaluate all translations with both models as LLM-as-a-judge."""
    logger.info("=" * 60)
    logger.info("PHASE 2: LLM-as-a-Judge Evaluation (holistic 3-score)")
    logger.info("=" * 60)

    with open(JUDGE_TEMPLATE_PATH, 'r') as f:
        template_source = f.read()

    judge_results = {}

    for translator_key in MODELS:
        for judge_key in MODELS:
            judge_info = MODELS[judge_key]
            api_url = f"http://{judge_info['worker']}:{judge_info['port']}/v1/chat/completions"
            model_name = judge_info["name"]

            for lang_pair in LANG_PAIRS:
                key = (translator_key, judge_key, lang_pair)
                cache_path = os.path.join(results_path, f"judge_{translator_key}_by_{judge_key}_{lang_pair}")

                if os.path.exists(cache_path):
                    logger.info(f"  Loading cached judge results: {key}")
                    judge_results[key] = load_from_disk(cache_path)
                    continue

                logger.info(f"  LLM-judge: {judge_key} evaluating {translator_key}'s {lang_pair} translations...")
                ds = translations[(translator_key, lang_pair)]

                evaluated_ds = ds.map(
                    lambda x: _judge_evaluate_single(x, api_url, model_name, template_source, lang_pair),
                    num_proc=JUDGE_NUM_PROC,
                    load_from_cache_file=False,
                    desc=f"Judge {judge_key} on {translator_key}/{lang_pair}"
                )

                evaluated_ds.save_to_disk(cache_path)
                judge_results[key] = evaluated_ds
                logger.info(f"  Saved judge results ({len(evaluated_ds)} samples) to {cache_path}")

    return judge_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: CEF EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def _cef_evaluate_single(example, cef_api_url, pipeline_params_dict,
                         prompt_catalogue_path, judge_api_url, judge_model,
                         num_questions):
    """Evaluate a single sample using CEF server (standalone function)."""
    pp = copy.deepcopy(pipeline_params_dict)
    pp["gen_qa_document"]["num_questions_to_generate"] = num_questions
    pp["gen_qa_summary"]["num_questions_to_generate"] = num_questions
    pp["gen_qa_document"]["llm_params"]["llm_model"] = judge_model
    pp["gen_qa_summary"]["llm_params"]["llm_model"] = judge_model
    pp["cross_examine"]["llm_params"]["llm_model"] = judge_model
    pp["base_endpoint"] = judge_api_url
    pp["gen_qa_document"]["base_endpoint"] = judge_api_url
    pp["gen_qa_summary"]["base_endpoint"] = judge_api_url
    pp["cross_examine"]["base_endpoint"] = judge_api_url

    payload = {
        "original_document": example["src"],
        "generated_document": example["trg"],
        "prompt_catalogue_path": prompt_catalogue_path,
        "config": pp
    }

    for attempt in range(3):
        try:
            response = requests.post(cef_api_url, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            scores = data.get("scores", {})
            return {
                "cef_coverage": scores.get("coverage_score"),
                "cef_conformity": scores.get("conformity_score"),
                "cef_consistency": scores.get("consistency_score"),
            }
        except Exception as e:
            logger.warning(f"CEF attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))

    return {
        "cef_coverage": None,
        "cef_conformity": None,
        "cef_consistency": None,
    }


def start_cef_server():
    """Start CEF server as a subprocess and wait for it to be ready."""
    logger.info("Starting CEF server on port %d...", CEF_SERVER_PORT)
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(CEF_SERVER_PORT),
         "--workers", "4", "--timeout-keep-alive", "600"],
        cwd=str(CEF_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def cleanup():
        if proc.poll() is None:
            logger.info("Shutting down CEF server (pid=%d)...", proc.pid)
            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)

    atexit.register(cleanup)

    # Wait for server to be ready
    health_url = f"http://localhost:{CEF_SERVER_PORT}/docs"
    for i in range(60):
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                logger.info("CEF server is ready!")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(2)

    raise RuntimeError("CEF server failed to start within 120 seconds")


def run_cef_evaluation(translations, results_path):
    """Phase 3: Evaluate all translations with both models as CEF judges."""
    logger.info("=" * 60)
    logger.info("PHASE 3: CEF Evaluation (QA-decomposed)")
    logger.info("=" * 60)

    cef_api_url = f"http://localhost:{CEF_SERVER_PORT}/evaluate"

    # Check if CEF server is already running
    try:
        requests.get(f"http://localhost:{CEF_SERVER_PORT}/docs", timeout=5)
        logger.info("CEF server already running on port %d", CEF_SERVER_PORT)
        cef_proc = None
    except requests.ConnectionError:
        cef_proc = start_cef_server()

    with open(CEF_PIPELINE_PARAMS_PATH, 'r') as f:
        pipeline_params = yaml.safe_load(f)

    cef_results = {}

    for translator_key in MODELS:
        for judge_key in MODELS:
            judge_info = MODELS[judge_key]
            judge_api_url = f"http://{judge_info['worker']}:{judge_info['port']}/v1"
            judge_model = judge_info["name"]

            for lang_pair in LANG_PAIRS:
                key = (translator_key, judge_key, lang_pair)
                cache_path = os.path.join(results_path, f"cef_{translator_key}_by_{judge_key}_{lang_pair}")

                if os.path.exists(cache_path):
                    logger.info(f"  Loading cached CEF results: {key}")
                    cef_results[key] = load_from_disk(cache_path)
                    continue

                logger.info(f"  CEF: {judge_key} evaluating {translator_key}'s {lang_pair} translations...")
                ds = translations[(translator_key, lang_pair)]

                evaluated_ds = ds.map(
                    lambda x: _cef_evaluate_single(
                        x, cef_api_url, pipeline_params,
                        CEF_PROMPT_CATALOGUE_PATH, judge_api_url, judge_model,
                        NUM_QUESTIONS
                    ),
                    num_proc=CEF_NUM_PROC,
                    load_from_cache_file=False,
                    desc=f"CEF {judge_key} on {translator_key}/{lang_pair}"
                )

                evaluated_ds.save_to_disk(cache_path)
                cef_results[key] = evaluated_ds
                logger.info(f"  Saved CEF results ({len(evaluated_ds)} samples) to {cache_path}")

    # Shutdown CEF server if we started it
    if cef_proc is not None and cef_proc.poll() is None:
        logger.info("Shutting down CEF server...")
        os.kill(cef_proc.pid, signal.SIGTERM)
        cef_proc.wait(timeout=10)

    return cef_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def build_results_dataframe(judge_results, cef_results):
    """Combine all evaluation results into a single DataFrame."""
    rows = []
    for translator_key in MODELS:
        for judge_key in MODELS:
            for lang_pair in LANG_PAIRS:
                jkey = (translator_key, judge_key, lang_pair)
                judge_ds = judge_results.get(jkey)
                cef_ds = cef_results.get(jkey)

                if judge_ds is None and cef_ds is None:
                    continue

                n = len(judge_ds) if judge_ds is not None else len(cef_ds)

                for i in range(n):
                    row = {
                        "sample_idx": i,
                        "sample_id": (judge_ds[i] if judge_ds else cef_ds[i]).get("id", i),
                        "translator": translator_key,
                        "judge": judge_key,
                        "lang_pair": lang_pair,
                        "is_self_eval": translator_key == judge_key,
                    }
                    if judge_ds is not None:
                        row["judge_coverage"] = judge_ds[i].get("judge_coverage", -1)
                        row["judge_conformity"] = judge_ds[i].get("judge_conformity", -1)
                        row["judge_consistency"] = judge_ds[i].get("judge_consistency", -1)
                    if cef_ds is not None:
                        row["cef_coverage"] = cef_ds[i].get("cef_coverage")
                        row["cef_conformity"] = cef_ds[i].get("cef_conformity")
                        row["cef_consistency"] = cef_ds[i].get("cef_consistency")
                    rows.append(row)

    return pd.DataFrame(rows)


def compute_paired_bias_test(results_df, method, metric):
    """
    Compute paired self-preference bias test for a given method and metric.
    
    For each (sample, translator, lang_pair):
      - self_score: score when judge == translator
      - cross_score: score when judge != translator
      - diff = self_score - cross_score
    
    Test H0: median(diff) = 0 via Wilcoxon signed-rank test.
    """
    score_col = f"{method}_{metric}"

    if score_col not in results_df.columns:
        return {"error": f"Column {score_col} not found"}

    # Build paired data
    self_df = results_df[results_df["is_self_eval"]].copy()
    cross_df = results_df[~results_df["is_self_eval"]].copy()

    # Merge on (sample_idx, translator, lang_pair) to get paired observations
    merge_keys = ["sample_idx", "translator", "lang_pair"]
    paired = self_df[merge_keys + [score_col]].merge(
        cross_df[merge_keys + [score_col]],
        on=merge_keys,
        suffixes=("_self", "_cross"),
        how="inner"
    )

    self_col = f"{score_col}_self"
    cross_col = f"{score_col}_cross"

    # Filter valid scores (non-null, non-negative)
    valid_mask = (
        paired[self_col].notna() & paired[cross_col].notna() &
        (paired[self_col] >= 0) & (paired[cross_col] >= 0)
    )
    paired = paired[valid_mask]

    if len(paired) < 10:
        return {"error": f"Too few valid paired observations ({len(paired)})"}

    self_scores = paired[self_col].values
    cross_scores = paired[cross_col].values
    diffs = self_scores - cross_scores

    self_mean = np.mean(self_scores)
    cross_mean = np.mean(cross_scores)
    bias = np.mean(diffs)

    # Wilcoxon signed-rank test
    # Filter out zero differences (Wilcoxon requires non-zero diffs)
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) < 10:
        # Fall back to Mann-Whitney U if too many ties
        stat, p_value = scipy_stats.mannwhitneyu(self_scores, cross_scores, alternative='two-sided')
        test_name = "Mann-Whitney U"
    else:
        stat, p_value = scipy_stats.wilcoxon(nonzero_diffs)
        test_name = "Wilcoxon signed-rank"

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.var(self_scores, ddof=1) + np.var(cross_scores, ddof=1)) / 2)
    cohens_d = bias / pooled_std if pooled_std > 0 else 0.0

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_label = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_label = "small"
    elif abs(cohens_d) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    return {
        "self_eval_mean": round(float(self_mean), 4),
        "cross_eval_mean": round(float(cross_mean), 4),
        "bias": round(float(bias), 4),
        "bias_pct": round(float(bias / cross_mean * 100), 2) if cross_mean != 0 else None,
        "test": test_name,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "cohens_d": round(float(cohens_d), 4),
        "effect_size": effect_label,
        "n_pairs": len(paired),
        "self_std": round(float(np.std(self_scores, ddof=1)), 4),
        "cross_std": round(float(np.std(cross_scores, ddof=1)), 4),
    }


def compute_per_model_bias(results_df, method, metric):
    """Compute bias separately for each translator model."""
    per_model = {}
    for translator in MODELS:
        sub_df = results_df[results_df["translator"] == translator].copy()
        per_model[translator] = compute_paired_bias_test(sub_df, method, metric)
    return per_model


def run_analysis(results_df, results_path):
    """Phase 4: Statistical analysis of self-preference bias."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Statistical Analysis")
    logger.info("=" * 60)

    methods = ["judge", "cef"]
    metrics = ["coverage", "conformity", "consistency"]
    analysis = {}

    for method in methods:
        analysis[method] = {}
        for metric in metrics:
            analysis[method][metric] = {
                "aggregate": compute_paired_bias_test(results_df, method, metric),
                "per_model": compute_per_model_bias(results_df, method, metric),
            }

    # ─── Print Results ───
    print("\n")
    print("=" * 80)
    print("   SELF-PREFERENCE BIAS ANALYSIS: CEF vs LLM-as-a-Judge")
    print("=" * 80)

    for method in methods:
        method_label = "CEF (QA-decomposed)" if method == "cef" else "LLM-as-a-Judge (holistic prompt)"
        print(f"\n{'━' * 80}")
        print(f"  Method: {method_label}")
        print(f"{'━' * 80}")

        for metric in metrics:
            agg = analysis[method][metric]["aggregate"]
            if "error" in agg:
                print(f"\n  {metric.upper()}: {agg['error']}")
                continue

            sig = "***" if agg["p_value"] < 0.001 else ("**" if agg["p_value"] < 0.01 else ("*" if agg["p_value"] < 0.05 else "n.s."))

            print(f"\n  {metric.upper()}:")
            print(f"    Self-eval mean:   {agg['self_eval_mean']:.4f}  (std={agg['self_std']:.4f})")
            print(f"    Cross-eval mean:  {agg['cross_eval_mean']:.4f}  (std={agg['cross_std']:.4f})")
            print(f"    Bias (self-cross): {agg['bias']:+.4f}  ({agg.get('bias_pct', 'N/A')}%)")
            print(f"    {agg['test']}: stat={agg['statistic']:.4f}, p={agg['p_value']:.6f} {sig}")
            print(f"    Cohen's d: {agg['cohens_d']:.4f} ({agg['effect_size']})")
            print(f"    N paired: {agg['n_pairs']}")

            # Per-model breakdown
            pm = analysis[method][metric]["per_model"]
            for model_key, model_stats in pm.items():
                if "error" in model_stats:
                    continue
                msig = "*" if model_stats["p_value"] < 0.05 else "n.s."
                print(f"      {model_key}: bias={model_stats['bias']:+.4f}, p={model_stats['p_value']:.4f} {msig}, d={model_stats['cohens_d']:.3f}")

    # ─── Summary Table ───
    print(f"\n{'=' * 80}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"\n  {'Method':<25} {'Metric':<15} {'Self':<10} {'Cross':<10} {'Bias':<10} {'p-value':<12} {'Sig?':<6} {'Cohen d':<10}")
    print(f"  {'─'*25} {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*6} {'─'*10}")

    for method in methods:
        label = "CEF" if method == "cef" else "LLM-as-Judge"
        for metric in metrics:
            agg = analysis[method][metric]["aggregate"]
            if "error" in agg:
                print(f"  {label:<25} {metric:<15} {'N/A':>10}")
                continue
            sig_str = "YES" if agg["significant_005"] else "NO"
            print(f"  {label:<25} {metric:<15} {agg['self_eval_mean']:<10.4f} {agg['cross_eval_mean']:<10.4f} {agg['bias']:<+10.4f} {agg['p_value']:<12.6f} {sig_str:<6} {agg['cohens_d']:<+10.4f}")

    # ─── Conclusion ───
    judge_sig = sum(
        1 for m in metrics
        if analysis["judge"][m]["aggregate"].get("significant_005", False)
    )
    cef_sig = sum(
        1 for m in metrics
        if analysis["cef"][m]["aggregate"].get("significant_005", False)
    )

    judge_avg_bias = np.mean([
        abs(analysis["judge"][m]["aggregate"].get("bias", 0))
        for m in metrics
        if "error" not in analysis["judge"][m]["aggregate"]
    ])
    cef_avg_bias = np.mean([
        abs(analysis["cef"][m]["aggregate"].get("bias", 0))
        for m in metrics
        if "error" not in analysis["cef"][m]["aggregate"]
    ])

    print(f"\n{'=' * 80}")
    print("  CONCLUSION")
    print(f"{'=' * 80}")
    print(f"\n  LLM-as-a-Judge: {judge_sig}/3 metrics show significant self-preference (avg |bias|={judge_avg_bias:.4f})")
    print(f"  CEF:            {cef_sig}/3 metrics show significant self-preference (avg |bias|={cef_avg_bias:.4f})")

    if judge_sig > cef_sig or (judge_sig == cef_sig and judge_avg_bias > cef_avg_bias * 1.5):
        print(f"\n  RESULT: CEF shows LESS self-preference bias than LLM-as-a-Judge.")
        print(f"  This supports the claim that CEF's decomposed QA-based evaluation")
        print(f"  mitigates self-preference bias inherent in holistic LLM scoring.")
    elif judge_sig == cef_sig and abs(judge_avg_bias - cef_avg_bias) / max(judge_avg_bias, cef_avg_bias, 1e-9) < 0.3:
        print(f"\n  RESULT: Both methods show comparable levels of self-preference bias.")
    else:
        print(f"\n  RESULT: CEF shows MORE self-preference bias than LLM-as-a-Judge.")
        print(f"  This does NOT support the bias mitigation claim.")

    print()

    # Save analysis (convert numpy types for JSON serialization)
    def numpy_to_python(obj):
        if isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_python(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(results_path, "bias_analysis.json"), "w") as f:
        json.dump(numpy_to_python(analysis), f, indent=2)
    logger.info("Analysis saved to %s/bias_analysis.json", results_path)

    return analysis


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(
    phase: str = "all",
    debug: bool = False,
    results_path: str = None,
):
    """
    Run the self-preference bias experiment.

    Args:
        phase: Which phase to run: 'all', 'translate', 'judge', 'cef', 'analyze'
        debug: If True, use only 30 samples per split for faster testing
        results_path: Override results directory path
    """
    if results_path is None:
        results_path = RESULTS_BASE + ("_debug" if debug else "")
    os.makedirs(results_path, exist_ok=True)

    logger.info("Self-Preference Bias Experiment")
    logger.info("  Phase: %s", phase)
    logger.info("  Debug: %s", debug)
    logger.info("  Results: %s", results_path)
    logger.info("  Models: %s", list(MODELS.keys()))
    logger.info("  Lang pairs: %s", LANG_PAIRS)

    # Load dataset
    logger.info("Loading NTREX dataset from %s", DATA_PATH)
    ntrex_ds = load_from_disk(DATA_PATH)
    if debug:
        for split in ntrex_ds:
            ntrex_ds[split] = ntrex_ds[split].select(range(min(30, len(ntrex_ds[split]))))
        logger.info("Debug mode: using %d samples per split", len(ntrex_ds[LANG_PAIRS[0]]))

    # Phase 1: Translation
    if phase in ("all", "translate"):
        translations = run_translation(ntrex_ds, results_path, debug=debug)
    else:
        translations = load_translations(results_path)

    # Phase 2: LLM-as-a-Judge
    if phase in ("all", "judge"):
        judge_results = run_judge_evaluation(translations, results_path)
    else:
        judge_results = {}
        for tk in MODELS:
            for jk in MODELS:
                for lp in LANG_PAIRS:
                    cp = os.path.join(results_path, f"judge_{tk}_by_{jk}_{lp}")
                    if os.path.exists(cp):
                        judge_results[(tk, jk, lp)] = load_from_disk(cp)

    # Phase 3: CEF
    if phase in ("all", "cef"):
        cef_results = run_cef_evaluation(translations, results_path)
    else:
        cef_results = {}
        for tk in MODELS:
            for jk in MODELS:
                for lp in LANG_PAIRS:
                    cp = os.path.join(results_path, f"cef_{tk}_by_{jk}_{lp}")
                    if os.path.exists(cp):
                        cef_results[(tk, jk, lp)] = load_from_disk(cp)

    # Phase 4: Analysis
    if phase in ("all", "judge", "cef", "analyze"):
        results_df = build_results_dataframe(judge_results, cef_results)
        csv_path = os.path.join(results_path, "all_results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info("Raw results saved to %s (%d rows)", csv_path, len(results_df))
        analysis = run_analysis(results_df, results_path)

    logger.info("Experiment complete. All results in %s", results_path)


if __name__ == "__main__":
    fire.Fire(main)

