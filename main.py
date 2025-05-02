import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

import logging
import uuid

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Assumed Imports ---
try:
    from utils import load_yaml
    from QAG import generate_qa_from_text, cross_examine, calculate_scores
    # Assuming you finalized the async version of LLM_Caller
    from LLM_API import LLM_Caller # Or LLM_API_multiprocess if that's the final name
except ImportError as e:
    logger.error(f"Error importing helper modules: {e}", exc_info=True)
    raise e

# --- Configuration ---
DEFAULT_CONFIG_PATH = "pipeline_params.yaml"
DEFAULT_PROMPT_CATALOGUE_PATH = "prompts.yaml"

# --- Pydantic Models ---
class QAPair(BaseModel):
    question: str
    answer: str # Or adjust type if answer format varies

class CrossExamRequest(BaseModel):
    original_document: str
    generated_document: str
    generated_qa_from_document: Optional[List[QAPair]] = None
    config_path: Optional[str] = Field(default=DEFAULT_CONFIG_PATH, description="Path to the pipeline parameters YAML file.")
    prompt_catalogue_path: Optional[str] = Field(default=DEFAULT_PROMPT_CATALOGUE_PATH, description="Path to the prompts YAML file.")

class CrossExamScores(BaseModel):
    coverage_score: Optional[float] = None
    conform_score: Optional[float] = None
    fact_score: Optional[float] = None
    overall_score: Optional[float] = None

# Define a model for the QA pair structure for clarity (Optional but good practice)

class CrossExamResponse(BaseModel):
    scores: CrossExamScores
    details: Optional[Dict[str, Any]] = None
    # Example structure for details:
    # {
    #     "qa_from_doc_count": int,
    #     "qa_from_summary_count": int,
    #     "questions_from_original_doc": List[QAPair] | List[Dict] | None,
    #     "questions_from_generated_doc": List[QAPair] | List[Dict] | None
    # }

# --- FastAPI App ---
app = FastAPI(
    title="Cross Examination API",
    description="Evaluates a generated document against an original document using LLM-based cross-examination.",
    version="1.0.1" # Bump version
)

# --- Helper Function (_get_llm_opts) ... ---
def _get_llm_opts(step_params: Dict, prompt_catalogue: Dict) -> Dict:
    """Constructs LLM options dictionary from config."""
    prompt_opts = step_params.get("prompts_from_catalogue", {})
    system_instruction_key = prompt_opts.get("system_instruction")
    user_instruction_key = prompt_opts.get("user_instruction")

    system_instruction = prompt_catalogue.get("system_instruction_set", {}).get(system_instruction_key)
    user_instruction = prompt_catalogue.get("user_instruction_set", {}).get(user_instruction_key)

    llm_opts = {"system_instruction": system_instruction, "user_instruction": user_instruction}
    llm_opts.update(step_params.get("llm_params", {})) # Merge specific params

    # Remove keys with None values as LLM_Caller might not expect them
    llm_opts = {k: v for k, v in llm_opts.items() if v is not None}
    return llm_opts


# --- Core Logic ---
async def perform_cross_examination(
    original_document: str,
    generated_document: str,
    config_path: str,
    prompt_catalogue_path: str,
    request_id: uuid.UUID,
    generated_qa_from_document: Optional[List[QAPair]] = None,
) -> Dict[str, Any]:
    """
    Performs the full cross-examination process for a single document pair.
    Returns scores and details including generated questions.
    """
    # Initialize variables to store results, handle potential errors
    qa_from_doc = None
    qa_from_summary = None
    summary_answers_to_doc_qs = None
    doc_answers_to_summary_qs = None
    calculated_scores = {}
    final_scores_obj = CrossExamScores() # Default empty scores

    logger.info(f"RID: {request_id} - Starting cross-examination...")
    logger.debug(f"RID: {request_id} - Original Doc Snippet: {original_document[:100]}...") # Log snippet
    logger.debug(f"RID: {request_id} - Generated Doc Snippet: {generated_document[:100]}...") # Log snippet

    # --- Load Config & Init LLM Callers ---
    try:
        pipeline_params = load_yaml(config_path)
        prompt_catalogue = load_yaml(prompt_catalogue_path)
        # ... Init LLM_Callers ...
        qa_doc_params = pipeline_params["gen_qa_document"]
        qa_sum_params = pipeline_params["gen_qa_summary"]
        cross_exam_params = pipeline_params["cross_examine"]
        score_params = pipeline_params["calc_scores"]

        llm_api_qa_gen_doc = LLM_Caller(qa_doc_params["base_endpoint"], prompt_filepath=prompt_catalogue_path)
        qa_sum_endpoint = qa_sum_params.get("base_endpoint", qa_doc_params["base_endpoint"])
        llm_api_qa_gen_sum = LLM_Caller(qa_sum_endpoint, prompt_filepath=prompt_catalogue_path)
        llm_api_cross_exam = LLM_Caller(cross_exam_params["base_endpoint"], prompt_filepath=prompt_catalogue_path)
    except Exception as e:
         logger.error(f"RID: {request_id} - Error during setup: {e}", exc_info=True)
         # Re-raise appropriate HTTPException or handle
         raise HTTPException(status_code=500, detail=f"RID: {request_id} - Server setup error: {e}")

    # --- Step 1: Generate QA from Original Document ---
    logger.info(f"RID: {request_id} - Step 1: Generating QA from Original Document...")
    try:
        llm_opts_qa_doc = _get_llm_opts(qa_doc_params, prompt_catalogue)
        if generated_qa_from_document is not None:
            # Convert Pydantic models to dicts
            try:
            
                qa_from_doc = [item.model_dump() for item in generated_qa_from_document]
                logger.info(f"RID: {request_id} - Using provided QA pairs, converted to dicts.")
            except AttributeError: # Fallback for Pydantic v1 if needed
                 qa_from_doc = [item.dict() for item in generated_qa_from_document]
                 logger.info(f"RID: {request_id} - Using provided QA pairs (Pydantic v1 fallback), converted to dicts.")
            except Exception as conversion_err:
                 logger.error(f"RID: {request_id} - Failed to convert provided QAPair models to dicts: {conversion_err}", exc_info=True)
                 qa_from_doc = None # Handle error, maybe proceed without QA or raise exception
        else:
            logger.info(f"RID: {request_id} - No QA provided, generating...")
            qa_from_doc = await generate_qa_from_text(
                text=original_document,
                llm_opts=llm_opts_qa_doc,
            llm_api=llm_api_qa_gen_doc,
            num_questions=qa_doc_params.get("num_questions_to_generate", 5), # Use .get w/ default
            prompt_catalogue_path=prompt_catalogue_path
        )
        qa_from_doc_repr = f"Count: {len(qa_from_doc)}, First: {qa_from_doc[0]}" if qa_from_doc else "None or Empty"
        print(qa_from_doc_repr)
        logger.debug(f"RID: {request_id} - Step 1 Result (qa_from_doc): {qa_from_doc_repr}")
        if qa_from_doc is not None and (not isinstance(qa_from_doc, list) or not all('question' in item and 'answer' in item for item in qa_from_doc)):
             logger.warning(f"RID: {request_id} - Unexpected format for qa_from_doc: {type(qa_from_doc)}. Check QAG.generate_qa_from_text output.")
             # Decide if this is critical - maybe set qa_from_doc to None or []?
             # qa_from_doc = None
    except Exception as e:
        logger.error(f"RID: {request_id} - Error in Step 1: {e}", exc_info=True)
        # Don't raise here, allow processing to continue if possible, log the error
        # The details dict will show qa_from_doc is None or empty

    # --- Step 2: Generate QA from Generated Document (Summary/Note) ---
    logger.info(f"RID: {request_id} - Step 2: Generating QA from Generated Document...")
    try:
        llm_opts_qa_sum = _get_llm_opts(qa_sum_params, prompt_catalogue)
        qa_from_summary = await generate_qa_from_text(
            text=generated_document,
            llm_opts=llm_opts_qa_sum,
            llm_api=llm_api_qa_gen_sum,
            num_questions=qa_sum_params.get("num_questions_to_generate", 5), # Use .get w/ default
            prompt_catalogue_path=prompt_catalogue_path
        )
        qa_from_summary_repr = f"Count: {len(qa_from_summary)}, First: {qa_from_summary[0]}" if qa_from_summary else "None or Empty"
        logger.debug(f"RID: {request_id} - Step 2 Result (qa_from_summary): {qa_from_summary_repr}")
        if qa_from_summary is not None and (not isinstance(qa_from_summary, list) or not all('question' in item and 'answer' in item for item in qa_from_summary)):
             logger.warning(f"RID: {request_id} - Unexpected format for qa_from_summary: {type(qa_from_summary)}. Check QAG.generate_qa_from_text output.")
             # qa_from_summary = None
    except Exception as e:
        logger.error(f"RID: {request_id} - Error in Step 2: {e}", exc_info=True)
        # Don't raise here

    # --- Step 3: Cross Examination ---
    logger.info(f"RID: {request_id} - Step 3: Performing Cross Examination...")
    try:
        llm_opts_cross_exam = _get_llm_opts(cross_exam_params, prompt_catalogue)

        # 3a: Ask Document Questions to Generated Document
        logger.info(f"RID: {request_id} -   Step 3a: Answering Document questions using Generated Document")
        summary_answers_to_doc_qs = await cross_examine(
            text=generated_document,
            gold_text=original_document,
            cross_question_set=qa_from_doc, # Pass potentially None list
            llm_opts=llm_opts_cross_exam,
            llm_api=llm_api_cross_exam,
            prompt_catalogue_path=prompt_catalogue_path
        )
        summary_answers_repr = f"Count: {len(summary_answers_to_doc_qs)}, First: {summary_answers_to_doc_qs[0]}" if summary_answers_to_doc_qs else "None or Empty"
        logger.debug(f"RID: {request_id} - Step 3a Result (summary_answers_to_doc_qs): {summary_answers_repr}")

        # 3b: Ask Summary Questions to Original Document
        logger.info(f"RID: {request_id} -   Step 3b: Answering Generated Document questions using Original Document")
        doc_answers_to_summary_qs = await cross_examine(
            text=original_document,
            gold_text=generated_document,
            cross_question_set=qa_from_summary, # Pass potentially None list
            llm_opts=llm_opts_cross_exam,
            llm_api=llm_api_cross_exam,
            prompt_catalogue_path=prompt_catalogue_path
        )
        doc_answers_repr = f"Count: {len(doc_answers_to_summary_qs)}, First: {doc_answers_to_summary_qs[0]}" if doc_answers_to_summary_qs else "None or Empty"
        logger.debug(f"RID: {request_id} - Step 3b Result (doc_answers_to_summary_qs): {doc_answers_repr}")

    except Exception as e:
        logger.error(f"RID: {request_id} - Error in Step 3: {e}", exc_info=True)
    

    # --- Step 4: Calculate Scores ---
    logger.info(f"RID: {request_id} - Step 4: Calculating Scores...")
    # Proceed only if cross-examination results are available
    if summary_answers_to_doc_qs is not None or doc_answers_to_summary_qs is not None:
        try:
            input_field_keys = score_params["read_dataset"]["input_field"]
            src_key, gen_key, doc_answers_key, sum_answers_key = input_field_keys[:4]
            qa_doc_key = "Generated_QA_from_Document" # Standard key assumed
            qa_sum_key = "Generated_QA_from_Generated_Summary" # Standard key assumed

            mock_row = {
                src_key: original_document,
                gen_key: generated_document,
                qa_doc_key: qa_from_doc if qa_from_doc else [], # Use empty list if None
                qa_sum_key: qa_from_summary if qa_from_summary else [], # Use empty list if None
                doc_answers_key: doc_answers_to_summary_qs, # Pass potentially None
                sum_answers_key: summary_answers_to_doc_qs, # Pass potentially None
            }
            mock_row_summary = {k: type(v).__name__ + (f" (len={len(v)})" if isinstance(v, list) else "") for k, v in mock_row.items()}
            logger.debug(f"RID: {request_id} - Mock Row Summary for calculate_scores: {mock_row_summary}")

            # calculate_scores should handle None inputs for answer lists gracefully
            calculated_scores = calculate_scores(mock_row, doc_answers_key, sum_answers_key)
            logger.debug(f"RID: {request_id} - Raw calculated_scores: {calculated_scores}")

            if not isinstance(calculated_scores, dict):
                logger.warning(f"RID: {request_id} - calculate_scores did not return a dict. Got: {type(calculated_scores)}. Scores will be empty.")
                calculated_scores = {}

            coverage = calculated_scores.get('coverage score')
            conform = calculated_scores.get('conform score')
            fact = calculated_scores.get('fact score')

            valid_scores = [s for s in [coverage, conform, fact] if isinstance(s, (int, float)) and s >= 0] # Ensure valid scores (e.g., >=0)
            overall = np.mean(valid_scores) if valid_scores else None

            final_scores_obj = CrossExamScores(
                coverage_score=coverage,
                conform_score=conform,
                fact_score=fact,
                overall_score=overall
            )
            logger.info(f"RID: {request_id} - Final Scores: {final_scores_obj.dict()}")

        except KeyError as e:
             logger.error(f"RID: {request_id} - Missing key during score calculation (Step 4): {e}", exc_info=True)
             # Keep default empty scores
        except Exception as e:
             logger.error(f"RID: {request_id} - Error during score calculation (Step 4): {e}", exc_info=True)
             # Keep default empty scores
    else:
        logger.warning(f"RID: {request_id} - Skipping score calculation as cross-examination results are missing.")


    # --- Construct Final Details ---
    details = {
        "qa_from_doc_count": len(qa_from_doc) if isinstance(qa_from_doc, list) else 0,
        "qa_from_summary_count": len(qa_from_summary) if isinstance(qa_from_summary, list) else 0,
        "questions_from_original_doc": qa_from_doc if qa_from_doc else [],
        "questions_from_generated_doc": qa_from_summary if qa_from_summary else [],
    }
    logger.info(f"RID: {request_id} - Cross-examination complete.")
    return {"scores": final_scores_obj, "details": details}


# --- API Endpoint ---
@app.post("/evaluate", response_model=CrossExamResponse)
async def evaluate_documents(
    request: CrossExamRequest = Body(...)
):
    """
    Receives original and generated documents, performs cross-examination,
    and returns coverage, conform, and fact scores, plus generated questions
    in the details field.
    """
    request_id = uuid.uuid4()
    logger.info(f"RID: {request_id} - Received evaluation request. Config path: {request.config_path}")
    try:
        # Call the core logic function, passing the request_id
        result = await perform_cross_examination(
            original_document=request.original_document,
            generated_qa_from_document=request.generated_qa_from_document,
            generated_document=request.generated_document,
            config_path=request.config_path,
            prompt_catalogue_path=request.prompt_catalogue_path,
            request_id=request_id
        )
        # Extract results safely
        final_scores = result.get('scores', CrossExamScores()) # Default to empty scores if key missing
        details = result.get('details')

        logger.info(f"RID: {request_id} - Evaluation successful. Overall Score: {getattr(final_scores, 'overall_score', 'N/A')}")
        return CrossExamResponse(scores=final_scores, details=details)
    except HTTPException as e:
         # Log FastAPI/HTTP errors specifically
         logger.error(f"RID: {request_id} - HTTPException during processing: {e.detail}", exc_info=True)
         raise # Re-raise the exception
    except Exception as e:
         # Catch any other unexpected errors during the await or processing
         logger.error(f"RID: {request_id} - Unexpected error in endpoint handler: {e}", exc_info=True)
         # Return a generic server error response
         raise HTTPException(status_code=500, detail=f"RID: {request_id} - Internal Server Error: An unexpected error occurred.")


# --- Main execution block ---
# (Keep as is)
if __name__ == "__main__":
    import uvicorn
    print("Starting Cross Examination API server...")
    # Example: uvicorn eval_as_api:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run("main:app", host="0.0.0.0", port=8000) # Use string format for uvicorn.run