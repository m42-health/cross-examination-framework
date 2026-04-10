# CEF: Cross-Examination Framework for LLM Evaluation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2601.19350)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](environment.yml)

Official code for the paper **"CEF: Cross-Examination Framework for Evaluating LLMs in Document Generation Tasks"** (ACL 2025).

> **Abstract.** Evaluating LLMs on long-form generation tasks — such as translation, summarization, and clinical note generation — remains a fundamental challenge. We propose the **Cross-Examination Framework (CEF)**, a reference-free, task-agnostic evaluation method that works by generating questions from the source and target documents and cross-examining each document using the other's questions. CEF produces four interpretable metrics: coverage, conformity, consistency, and conciseness. We demonstrate CEF's effectiveness across five diverse tasks — machine translation (FLORES, WMT25), abstractive summarization (CNN/DM), factuality assessment (FRANK), clinical note generation (ACI-Bench, DischargeME), and data-to-text (Rotowire) — and show that it aligns strongly with human judgments while being robust to self-preference bias across multiple LLM judges.

---

## 🏗️ Repository Structure

```
cef-translation/
├── src/
│   ├── cef_framework/          # Core CEF API server
│   │   ├── main.py             # FastAPI application
│   │   ├── QAG.py              # Question-Answer Generation
│   │   ├── LLM_API.py          # OpenAI-compatible LLM client
│   │   ├── utils.py            # Utilities
│   │   ├── pipeline_params.yaml # Server configuration
│   │   └── prompts/            # Task-specific prompt templates
│   │       ├── translation.yaml
│   │       ├── summarization.yaml
│   │       ├── hospital_course.yaml
│   │       ├── note_generation.yaml
│   │       └── boxscore.yaml
│   ├── evaluation/             # Evaluation runners (FLORES, WMT25, DischargeME, ...)
│   ├── summarization/          # CNN/DM summarization evaluation
│   ├── frank/                  # FRANK factuality benchmark
│   ├── note-generation/        # ACI-Bench clinical note evaluation
│   ├── rotowire/               # Rotowire data-to-text evaluation
│   ├── cef_robustness/         # Robustness / self-preference bias study
│   └── yes_only/               # Yes-only baseline experiment
├── scripts/                    # SLURM launchers & analysis scripts
│   ├── eval.sh                 # FLORES/WMT25 evaluation launcher
│   ├── eval_frank.sh           # FRANK benchmark launcher
│   ├── eval_cnndm.sh           # CNN/DM evaluation launcher
│   ├── eval_boxscore.sh        # Rotowire evaluation launcher
│   ├── eval_aci.sh             # ACI-Bench launcher
│   ├── eval_dischargeme_*.sh   # DischargeME launchers
│   ├── run_cef_server.sh       # CEF API server launcher
│   └── self_preference_bias_experiment.py  # Self-preference bias study
├── wmt25/                      # WMT25 analysis scripts (GEMBA/Human alignment)
├── notebooks/                  # Exploratory analysis notebooks
├── data/                       # Data directory (large files not tracked in git)
│   ├── processed/              # Processed sample files
│   └── raw/                    # Raw sample files
├── figures/                    # Paper figures
├── environment.yml             # Conda environment
└── EXPERIMENTS.md              # Step-by-step experiment reproduction guide
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Conda or Miniconda
- Access to an OpenAI-compatible LLM API (local vLLM or hosted)

### Setup

```bash
git clone https://github.com/<your-org>/cef-translation.git
cd cef-translation

# Create and activate the conda environment
conda env create -f environment.yml
conda activate cross-examination-framework
```

---

## ⚡ Quick Start: CEF API Server

CEF runs as a lightweight REST API that you point your evaluation scripts at.

### 1. Configure the server

Edit `src/cef_framework/pipeline_params.yaml` to set your LLM endpoint:

```yaml
base_endpoint: "http://<your-llm-host>:<port>/v1/"
api_key: "YOUR_API_KEY"          # "EMPTY" for local vLLM
model: "your-model-name"

gen_qa_document:
  base_endpoint: "http://<your-llm-host>:<port>/v1/"
  num_questions_to_generate: 10
  llm_params:
    llm_model: "your-model-name"
    max_tokens: 8000
    temperature: 0
# ... (see file for full options)
```

### 2. Start the server

```bash
cd src/cef_framework
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8893
```

Or use the convenience script:
```bash
bash scripts/run_cef_server.sh
```

### 3. Call the API

```bash
curl -X POST http://0.0.0.0:8893/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "original_document": "Patient is a 65-year-old male with hypertension presenting with chest pain.",
    "generated_document": "Le patient est un homme de 65 ans atteint d'\''hypertension présentant une douleur thoracique."
  }'
```

**Response:**
```json
{
  "scores": {
    "coverage_score": 0.95,
    "conformity_score": 1.0,
    "consistency_score": 0.90,
    "conciseness_score": -12.5,
    "overall_score": 0.95
  },
  "details": { ... }
}
```

#### Python client example

```python
import requests

response = requests.post("http://0.0.0.0:8893/evaluate", json={
    "original_document": "Source text...",
    "generated_document": "Generated / translated text..."
})
print(response.json()["scores"])
```

---

## 📊 CEF Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Coverage** | How thoroughly the output covers the source | 0–1 (↑ better) |
| **Conformity** | Non-contradiction score — output doesn't contradict source | 0–1 (↑ better) |
| **Consistency** | Non-hallucination — output stays faithful to source facts | 0–1 (↑ better) |
| **Conciseness** | Word-count reduction vs. source | % (positive = more concise) |
| **Overall** | Mean of coverage, conformity, consistency | 0–1 (↑ better) |

---

## 🔬 Experiments

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for step-by-step instructions to reproduce all paper experiments across:

| Task | Dataset | Script |
|------|---------|--------|
| Machine Translation | FLORES-200 | `scripts/eval.sh` |
| Machine Translation | WMT25 (en→ja) | `scripts/eval.sh` |
| Abstractive Summarization | CNN/DM | `scripts/eval_cnndm.sh` |
| Factuality Assessment | FRANK | `scripts/eval_frank.sh` |
| Clinical Note Generation | ACI-Bench | `scripts/eval_aci.sh` |
| Clinical Note Generation | DischargeME | `scripts/eval_dischargeme_*.sh` |
| Data-to-Text | Rotowire | `scripts/eval_boxscore.sh` |
| Self-Preference Bias | CNN/DM (multi-judge) | `scripts/self_preference_bias_experiment.py` |

---

## ⚙️ Configuration

### Task-Specific Prompts

Each task uses a different prompt template located in `src/cef_framework/prompts/`:

| File | Task |
|------|------|
| `translation.yaml` | Machine translation (FLORES, WMT25) |
| `summarization.yaml` | CNN/DM, FRANK |
| `hospital_course.yaml` | DischargeME (Brief Hospital Course) |
| `note_generation.yaml` | ACI-Bench (clinical note generation) |
| `boxscore.yaml` | Rotowire / data-to-text |

Pass the appropriate prompt file via `--prompt_catalogue_path` (or `--prompt_path`) in the evaluation scripts.

---

## 📄 License

This project is licensed under [CC BY-NC 4.0](LICENSE.md).

---

## 📚 Citation

If you use CEF in your research, please cite:

```bibtex
@misc{raha2026crossexaminationframeworktaskagnosticdiagnostic,
      title={Cross-Examination Framework: A Task-Agnostic Diagnostic for Information Fidelity in Text-to-Text Generation}, 
      author={Tathagata Raha and Clement Christophe and Nada Saadi and Hamza A Javed and Marco AF Pimentel and Ronnie Rajan and Praveenkumar Kanithi},
      year={2026},
      eprint={2601.19350},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.19350}, 
}
```

> 📄 [Full paper on arXiv](https://arxiv.org/pdf/2601.19350)

---

## 🆘 Support

- Open an issue on GitHub for bugs or questions
- See `src/cef_framework/README.md` for API server internals
- See `EXPERIMENTS.md` for detailed experiment reproduction
