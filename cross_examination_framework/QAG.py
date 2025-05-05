"""
Module containing functionality for the evaluation of LLM generated
summaries (QAG framework).

Authors: Tathagata Raha, Nada Saadi, Hamza Javed
Company: M42
Date: June 2024
"""

# Import relevant packages
import json
import re

# from LLM_API import LLM_Caller
# from utils import load_yaml, safe_load_questions
# from LLM_API import LLM_Caller
from .LLM_API import LLM_Caller
from .utils import load_yaml, safe_load_questions

from pathlib import Path
from typing import List
import nltk
from nltk.tokenize import word_tokenize
import asyncio

nltk.download("punkt_tab")


def default_llm_instructions(
    llm_opts, llm_task: str, prompt_filepath: str | Path = "prompts.yaml"
):
    """
    Utility function to update llm_opts dictionary to include task specific user
    and system instruction if they are not provided.

    Args:
    llm_opts: LLM chat completion options (see LLM_API.chat_completion arguments).
    llm_task: "gen_summary", "gen_qa", "gen_answer".
    prompt_filepath: Filepath to prompts catalogue in yaml format. Defaults to "prompts.yaml".
    """

    # Load catalogue of prompts if insturctions not provided
    if "system_instruction" not in llm_opts or "user_instruction" not in llm_opts:
        prompt_catalogue = load_yaml(prompt_filepath)

    # Set basic system_instruction for LLM call if it is not provided
    if "system_instruction" not in llm_opts:
        llm_opts["system_instruction"] = prompt_catalogue["system_instruction_set"][
            "basic_v0"
        ]

    # Check to see if user_instruction for LLM call provided
    if "user_instruction" not in llm_opts:
        if llm_task == "gen_summary":
            instruct_key = "gen_summary_v0"
        elif llm_task == "gen_qa":
            instruct_key = "qa_gen_document_v0"
        elif llm_task == "gen_answer":
            instruct_key = "gen_answers_v0"
        else:
            raise ValueError(
                f"{llm_task} is not a defined LLM task. Default prompts only defined for the following tasks: 'gen_summary', 'gen_qa', 'gen_answer'."
            )

        llm_opts["user_instruction"] = prompt_catalogue["user_instruction_set"][
            instruct_key
        ]


def generate_summary(
    document: str, llm_opts: dict, messages=None, llm_api: LLM_Caller = None
) -> str:
    """
    Generate a summarized version of a document using an LLM.

    TODO: generation should not be part of the QAG framework, refactor away.

    Args:
    document: Source text to summarize.
    use_messages: If needed to use the messages key instead of chat template
    llm_opts: LLM chat completion options (see LLM_API.chat_completion arguments).
    llm_api: Instance of the LLM_Caller object.

    Returns:
    summary: Text summary of the document.
    """

    # If LLM user and system instructions not provided set to default for this task
    default_llm_instructions(llm_opts=llm_opts, llm_task="gen_summary")

    # Add text to user instruction that must be summarized
    if messages is not None:
        llm_opts["system_instruction"] = messages[0]["content"]
        llm_opts["user_instruction"] = messages[1]["content"]
    else:
        prompt_catalogue = load_yaml("summarization_evaluation/prompts.yaml")
        llm_opts["user_instruction"] = prompt_catalogue["user_instruction_set"][
            "gen_summary_v0"
        ].format(text=document)
        # llm_opts["user_instruction"] = llm_opts["user_instruction"].format(text=document)

    # Perform LLM call
    if llm_api is None:
        llm_api = LLM_Caller()
    summary = llm_api.chat_completion(**llm_opts)

    return summary


async def generate_qa_from_text(
    text: str,
    llm_opts: dict,
    num_questions: int = 5,
    llm_api: LLM_Caller = None,
    screen_answers: bool = False,
    prompt_catalogue_path: str = "",
) -> List[dict]:
    """
    Generate question-answer pairs from a reference document using LLM.

    Args:
    text: Source text from which to generate questions.
    llm_opts: LLM chat completion options (see LLM_API.chat_completion arguments).
    num_questions: Number of questions to generate. Default is set to 5.
    llm_api: Instance of the LLM_Caller object. Default is to instantiate within function.
    screen_answers: Determines whether LLM queries the text with the questions a second time, answers
    produced in the second pass are saved as the final answers. Default is set to True.

    Returns:
    qa_pairs: Question-answer pairs in JSON format.
    """

    # If LLM user and system instructions not provided set to default for this task
    default_llm_instructions(llm_opts=llm_opts, llm_task="gen_qa")
    # print(text)
    # Add task specific parameters to the user instruction
    prompt_catalogue = load_yaml(prompt_catalogue_path)
    llm_opts["user_instruction"] = prompt_catalogue["user_instruction_set"][
        "qa_gen_document_v0"
    ].format(
        text=text, num_questions=num_questions, document_task="content of the document"
    )
    # print(llm_opts["user_instruction"])
    # LLM call to generate question-answer pairs from the text
    if llm_api is None:
        llm_api = LLM_Caller()
    llm_output = await llm_api.chat_completion(**llm_opts)

    # Ensure llm_output is a string
    if not isinstance(llm_output, str):
        # raise ValueError("Expected llm_output to be a string, but got {}".format(type(llm_output).__name__))
        print(
            f"This is an instance of llm_output NOT being a string {type(llm_output).__name__}!"
        )  # TODO: what causes llm_output to be None?
        return None
    else:
        # Postprocessing of the LLM output - extract the JSON part using a regular expression
        qa_pairs = safe_load_questions(llm_output)
        try:
            qa_pairs = json.loads(qa_pairs)
        except:
            print(f"No JSON content found in the LLM generated response!")
            return None

    # Screen answers in a second pass and update - assumes querying the text with a single question will return a more accurate result
    if screen_answers and qa_pairs is not None:
        update_answers = []
        prompt_catalogue = load_yaml(
            prompt_catalogue_path
        )  # TODO: just feed in prompt catalogue from the onset
        llm_opts["user_instruction"] = prompt_catalogue["user_instruction_set"][
            "gen_answers_v0"
        ]  # TODO: this is hardcoded too deep in the code
        for qa in qa_pairs:
            # print(f"\n~~\n{qa}\n")
            answer = await answer_questions_from_text(
                text=text, question=qa["question"], llm_opts=llm_opts, llm_api=llm_api
            )
            # print(f"\nOriginal Answer:{qa['answer']}\nUpdated Answer:{answer}\n")
            if "YES" in answer:
                update_answers.append({"question": qa["question"], "answer": "YES"})
            elif "NO" in answer:
                update_answers.append({"question": qa["question"], "answer": "NO"})
            elif "IDK" in answer:
                update_answers.append(
                    {"question": qa["question"], "answer": "IDK"}
                )  # TODO: how should we deal with this scenario
            else:
                print("LLM answer is indeterminable")

        qa_pairs = (
            update_answers  # TODO: make sure new qa_pairs same length as previously?
        )
    return qa_pairs


async def answer_questions_from_text(
    text: str,
    question: str,
    llm_opts: dict,
    llm_api: LLM_Caller = None,
    prompt_catalogue_path: str = "",
) -> str:
    """
    Query a text with a specified question using an LLM with set or specified prompts.

    Args:
    text: Source text from which to generate questions.
    question: Query to ask the text.
    llm_opts: LLM chat completion options (see LLM_API.chat_completion arguments).
    llm_api: Instance of the LLM_Caller object. Default is to instantiate within function.

    Returns:
    answer: LLM answer to the question posed.
    """

    # If LLM user and system instructions not provided set to default for this task
    default_llm_instructions(llm_opts=llm_opts, llm_task="gen_answer")
    # print(text, question)
    # print("************************************************")
    # Add task specific parameters to the user instruction
    prompt_catalogue = load_yaml(prompt_catalogue_path)
    llm_opts["user_instruction"] = prompt_catalogue["user_instruction_set"][
        "gen_answers_v0"
    ].format(text=text, question=question)
    # llm_opts["user_instruction"] = llm_opts["user_instruction"].format(text=text, question=question)

    # LLM call to generate answers from the text
    # print(llm_opts["user_instruction"])
    # print("*************************************************")
    if llm_api is None:
        llm_api = LLM_Caller()
    answer = await llm_api.chat_completion(**llm_opts)
    # print(answer)
    return answer


async def cross_examine(
    text: str,
    gold_text: str,
    cross_question_set: List[dict],
    llm_opts: dict,
    llm_api: LLM_Caller,
    prompt_catalogue_path: str,
) -> List[dict]:
    """
    Query a text from a set of independent questions, using an LLM.

    In the context of the QAG framework, question-answer pairs generated from a document
    can be used to cross-examine a summary, and vice-versa. Ground-truth answers should
    be included in the question set to be able to evaluate QAG scores downstream.

    Args:
    text: Source text to query the question set against.
    cross_question_set: List of question-answer pairs, of the form {question: "...?", answer: "YES/NO"}.
    llm_opts: LLM chat completion options (see LLM_API.chat_completion arguments).
    llm_api: Instance of the LLM_Caller object. Default is to instantiate within function.

    Return:
    predicted_answers: predicted answers for each question-answer pair added.
    """

    predicted_answers = []

    if cross_question_set is None:
        return None
    else:
        for qa_pair in cross_question_set:
            pred_answer = await answer_questions_from_text(
                text=text,
                question=qa_pair.get("question", None),
                llm_opts=llm_opts,
                llm_api=llm_api,
                prompt_catalogue_path=prompt_catalogue_path,
            )
            gold_answer = await answer_questions_from_text(
                text=gold_text,
                question=qa_pair.get("question", None),
                llm_opts=llm_opts,
                llm_api=llm_api,
                prompt_catalogue_path=prompt_catalogue_path,
            )
            predicted_answers.append(
                {
                    "predicted": pred_answer,
                    "gold": gold_answer,
                    "question": qa_pair["question"],
                }
            )
        return predicted_answers


def calculate_summary_reduction(samp, src, out):
    """
    Calculate the percentage reduction in word count between 'description' and 'generated_summary'.

    Parameters:
    samp (HF dataset sample): A sample of HF dataset containing 'description' and 'generated_summary' keys.
    src (key): Dataset key indicating the source i.e. document to be summarised
    out (key): Dataset key indicating the destination i.e. the summary

    Returns:
    float: The percentage reduction in word count.
    """
    description_words = word_tokenize(samp[src])
    summary_words = word_tokenize(samp[out])

    description_word_count = len(description_words)
    summary_word_count = len(summary_words)

    if description_word_count == 0:
        raise ValueError("Description cannot be empty")

    reduction_ratio = (
        description_word_count - summary_word_count
    ) / description_word_count
    reduction_percentage = reduction_ratio * 100

    return {"conciseness score": reduction_percentage}


def calculate_scores(samp, key1, key2) -> dict:
    questions_on_description = samp[key1]
    questions_on_summaries = samp[key2]
    if questions_on_summaries is not None:
        coverage = total = len(questions_on_summaries)
        for question in questions_on_summaries:
            if question["predicted"] == "IDK" and question["gold"] != "IDK":
                coverage -= 1
            elif question["gold"] == "IDK":
                coverage -= 1
                total -= 1
        if total != 0:
            coverage /= total
        else:
            coverage = 0
        conform = total = len(questions_on_summaries)
        for question in questions_on_summaries:
            if (
                question["predicted"] != "IDK"
                and question["predicted"] != question["gold"]
            ):
                conform -= 1
            elif question["gold"] == "IDK":
                conform -= 1
                total -= 1
        if total != 0:
            conform /= total
        else:
            conform = 0
    else:
        coverage = -1
        conform = -1
    if questions_on_description is not None:
        fact = total = len(questions_on_description)
        for question in questions_on_description:
            if question["predicted"] == "IDK" and question["gold"] != "IDK":
                fact -= 1
            elif question["gold"] == "IDK":
                fact -= 1
                total -= 1
        if total != 0:
            fact /= total
        else:
            fact = 0
    else:
        fact = -1
    return {
        "coverage score": coverage,
        "conformity score": conform,
        "consistency score": fact,
    }
