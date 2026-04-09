# CEF Framework — API Server

The `cef_framework` module implements the Cross-Examination Framework as a RESTful API server.
It is the core engine used by all evaluation scripts in this repository.

## How It Works

CEF evaluates a **(source document, generated document)** pair by:
1. **Generating questions** from the source document (Step 1)
2. **Generating questions** from the generated document (Step 2)
3. **Cross-examining** each document using the other's questions — answering YES/NO (Step 3)
4. **Calculating scores** from the YES/NO answer patterns (Step 4)

This produces four metrics: **Coverage**, **Conformity**, **Consistency**, and **Conciseness**.

## System Requirements

- Python 3.10+
- No GPU required — the API server itself is CPU-only
- Network access to an OpenAI-compatible LLM API


## Installation

```bash
conda env create -f ../../environment.yml
conda activate cross-examination-framework
```

## Configuration

### `pipeline_params.yaml`

Controls LLM endpoints and generation parameters for each pipeline step:

```yaml
base_endpoint: "http://<LLM-HOST>:<PORT>/v1/"
api_key: "EMPTY"                     # or your actual API key
model: "your-model-name"

gen_qa_document:                     # Step 1: QA from source
  base_endpoint: "http://..."
  num_questions_to_generate: 10
  llm_params:
    llm_model: "your-model"
    max_tokens: 8000
    temperature: 0

gen_qa_summary:                      # Step 2: QA from generated doc
  num_questions_to_generate: 10
  # ...

cross_examine:                       # Step 3: cross-examination
  base_endpoint: "http://..."
  # ...

calc_scores:                         # Step 4: score computation
  read_dataset:
    input_field: [src, trg, doc_answers, sum_answers]
```

### `prompts/<task>.yaml`

Each file defines the system and user prompts for question generation and cross-examination. Select the appropriate file for your task:

| File | Task |
|------|------|
| `translation.yaml` | Machine translation |
| `summarization.yaml` | Summarization / FRANK |
| `hospital_course.yaml` | DischargeME (Brief Hospital Course) |
| `note_generation.yaml` | ACI-Bench clinical notes |
| `boxscore.yaml` | Rotowire data-to-text |

## Running the Server

```bash
cd src/cef_framework
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8893
```

## API Reference

### `POST /evaluate`

**Request:**
```json
{
  "original_document": "Source text",
  "generated_document": "Generated / translated text",
  "generated_qa_from_document": null,   // optional: skip Step 1
  "config": "pipeline_params.yaml",     // path or inline dict
  "prompt_catalogue_path": "prompts/translation.yaml"
}
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
  "details": {
    "qa_from_doc_count": 10,
    "qa_from_summary_count": 10,
    "doc_answers_to_summary_qs": [...],
    "summary_answers_to_doc_qs": [...]
  }
}
```

## License

CC BY-NC 4.0 — see [LICENSE.md](../../LICENSE.md).

## Citation

```bibtex
@article{raha2025cef,
  title={CEF: Cross-Examination Framework for Evaluating LLMs in Document Generation Tasks},
  author={Raha, Tathagata and others},
  journal={arXiv preprint arXiv:2601.19350},
  year={2025}
}
```
