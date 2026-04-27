import logging
from typing import Dict, Any, List

from cross_examination_framework.QAG import (
    generate_qa_from_text,
    cross_examine,
    calculate_scores,
)
from cross_examination_framework.LLM_API import LLM_Caller

logger = logging.getLogger(__name__)

async def evaluate_medical(
    original_document: str,
    generated_summary: str,
    api_key: str,
    base_endpoint: str = "https://api.openai.com/v1/",
    model: str = "gpt-4o-mini-2024-07-18",
    prompt_filepath: str = "cross_examination_framework/prompts.yaml",
) -> Dict[str, Any]:
    """
    Evaluates clinical entity preservation (drugs, dosages, contraindications)
    between an original document and a generated summary.
    """
    logger.info("Starting Medical Cross-Examination for clinical entity preservation...")
    
    try:
        # Initialize LLM callers
        llm_api = LLM_Caller(
            base_url=base_endpoint,
            api_key=api_key,
            default_model=model,
            prompt_filepath=prompt_filepath,
        )
        
        llm_opts = {
            "llm_model": model,
            "max_tokens": 2000,
            "temperature": 0.0,
            "system_instruction": "You are a helpful biomedical assistant.",
            "user_instruction": ""  # Will be dynamically loaded by QAG
        }
        
        # Step 1: Generate QA from Document
        logger.info("Step 1: Generating QA from Original Document (Medical Entities)...")
        qa_doc = await generate_qa_from_text(
            text=original_document,
            llm_opts=llm_opts.copy(),
            num_questions=5,
            llm_api=llm_api,
            prompt_catalogue_path=prompt_filepath,
            qa_instruction_key="qa_gen_document_medical"
        )
        
        # Step 2: Generate QA from Summary
        logger.info("Step 2: Generating QA from Summary (Medical Entities)...")
        qa_sum = await generate_qa_from_text(
            text=generated_summary,
            llm_opts=llm_opts.copy(),
            num_questions=5,
            llm_api=llm_api,
            prompt_catalogue_path=prompt_filepath,
            qa_instruction_key="qa_gen_summary_medical"
        )
        
        # Step 3: Cross Examine
        logger.info("Step 3: Performing Cross Examination...")
        doc_answers = await cross_examine(
            text=original_document,
            gold_text=generated_summary,
            cross_question_set=qa_sum if qa_sum else [],
            llm_opts=llm_opts.copy(),
            llm_api=llm_api,
            prompt_catalogue_path=prompt_filepath,
        )
        
        sum_answers = await cross_examine(
            text=generated_summary,
            gold_text=original_document,
            cross_question_set=qa_doc if qa_doc else [],
            llm_opts=llm_opts.copy(),
            llm_api=llm_api,
            prompt_catalogue_path=prompt_filepath,
        )
        
        # Step 4: Compute Scores
        logger.info("Step 4: Calculating Medical Entity Preservation Scores...")
        mock_row = {
            "input": original_document,
            "Generated_Summary": generated_summary,
            "Document_Answers_to_Summary_Questions": doc_answers,
            "Summary_Answers_to_Document_Questions": sum_answers,
            "Generated_QA_from_Document": qa_doc,
            "Generated_QA_from_Generated_Summary": qa_sum
        }
        
        scores = calculate_scores(
            samp=mock_row,
            key1="Generated_QA_from_Document",
            key2="Generated_QA_from_Generated_Summary"
        )
        
        return {
            "scores": scores,
            "details": {
                "medical_qa_from_doc": qa_doc,
                "medical_qa_from_sum": qa_sum,
                "doc_answers_to_sum_questions": doc_answers,
                "sum_answers_to_doc_questions": sum_answers
            }
        }
        
    except Exception as e:
        logger.error(f"Error during evaluate_medical: {e}", exc_info=True)
        return {"error": str(e)}
