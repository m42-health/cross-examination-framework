# --- LLM_API.py (MODIFIED for Async) ---

# Import relevant packages
import httpx # Requires: pip install httpx
import asyncio
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from utils import load_yaml
import os # Added for path joining, though httpx base_url helps

# Setup basic logging if needed, or integrate with FastAPI's logger
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLM_Caller():
    """
    An ASYNCHRONOUS class for interacting with the API of hosted LLMs using httpx.
    """
    def __init__(self, base_endpoint: str = "http://0.0.0.0:8000/v1",
                 prompt_filepath: Optional[str | Path] = "prompts.yaml"):
        """
        Initialize the async LLM API caller object.

        Args:
            base_endpoint: URL address API calls are hosted. Defaults to "http://0.0.0.0:8000/v1"
            prompt_filepath: Filepath to prompts catalogue in yaml format. Defaults to "prompts.yaml".
        """
        if not base_endpoint:
             raise ValueError("base_endpoint cannot be empty")

        self.base_endpoint = base_endpoint # Store for reference if needed elsewhere
        self.docs_endpoint = os.path.join(str(base_endpoint), "docs") # Keep for repr

        # Initialize client here
        # Increased timeout for potentially long LLM operations
        self.client = httpx.AsyncClient(base_url=base_endpoint, timeout=120.0)
        logger.info(f"Async LLM_Caller initialized for base URL: {self.client.base_url}")

        # Load prompts synchronously during init (minor blocking ok, or load externally)
        # Consider loading once globally if prompts don't change per instance
        try:
            self.prompt_catalogue = load_yaml(prompt_filepath) if prompt_filepath is not None else None
        except Exception as e:
             logger.error(f"Failed to load prompt catalogue from {prompt_filepath}: {e}")
             self.prompt_catalogue = None # Handle initialization failure gracefully

        # NOTE: Cannot call async list_models here directly.
        # Call `await instance.list_models()` after creation if needed.
        self.available_models: Optional[List[dict]] = None # Initialize as None

    def __repr__(self):
        """
        Returns: Description of this object class including a link to API docs page
        """
        status = "Models listed" if self.available_models is not None else "Models not listed yet (call await list_models())"
        return f"Async LLM API Caller ({status}). Docs: {self.docs_endpoint}"

    async def list_models(self) -> Optional[List[dict]]:
        """
        ASYNCHRONOUSLY pull information on all the models currently hosted on API page.
        Updates self.available_models upon successful completion.

        Returns:
            List of models available and their details, or None if an error occurs.
        """
        models_url = "/models" # Relative path to base_url
        try:
            response = await self.client.get(models_url)
            response.raise_for_status()  # raise an exception for HTTP errors
            data = response.json().get('data', []) # Safely get 'data'
            self.available_models = data # Update instance attribute
            return data

        except httpx.RequestError as e:
            logger.error(f"Error listing models from {e.request.url!r}: {e}")
            self.available_models = None
            return None
        except httpx.HTTPStatusError as e:
             logger.error(f"Error listing models: HTTP {e.response.status_code} from {e.request.url!r}")
             self.available_models = None
             return None
        except Exception as e: # Catch other potential errors like JSONDecodeError
             logger.error(f"Unexpected error listing models: {e}")
             self.available_models = None
             return None

    # _update_prompt_catalogue remains synchronous as it deals with local file I/O
    # and is likely called infrequently or during setup. If needed async, use aiofiles.
    def _update_prompt_catalogue(self, prompt_filepath: str | Path = "prompts.yaml") -> dict:
        """
        Read in catalogue of prompts (recorded in a yaml file) and update attribute.

        Args:
            prompt_filepath: filepath to catalogue of prompts to be read. Defaults to "prompts.yaml".
        """
        try:
            self.prompt_catalogue = load_yaml(prompt_filepath)
            logger.info(f"Prompt catalogue updated from {prompt_filepath}")
        except Exception as e:
             logger.error(f"Failed to update prompt catalogue from {prompt_filepath}: {e}")
             # Decide whether to keep the old one or set to None
             # self.prompt_catalogue = None


    async def chat_completion(self, system_instruction: str, user_instruction: str,
                        llm_model: Optional[str] = None, # Make optional if sometimes omitted
                        max_tokens: int = 2000,
                        temperature: float = 0,
                        stop: Optional[List[str]] = ["<|eot_id|>"], # Use Optional
                        stop_token_ids: Optional[List[int]] = [128009]) -> str: # Use Optional
        """
        ASYNCHRONOUSLY uses chat completion functionality via httpx.

        Args:
            system_instruction: System instruction part of the prompt.
            user_instruction: User instruction part of the prompt.
            llm_model: Language model to use. Pass None if endpoint determines model.
            max_tokens: Maximum number of tokens to generate. Defaults to 2000.
            temperature: Temperature parameter for generating responses. Defaults to 0.
            stop: List of tokens to stop LLM generation on. Defaults to ["<|eot_id|>"] or None.
            stop_token_ids: List of token IDs to stop on. Defaults to [128009] or None.

        Returns:
            LLM generated response string.
        Raises:
             httpx exceptions on network/HTTP errors, ValueError/KeyError on response parsing issues.
        """
        # Relative path, assuming base_url ends with /v1 or similar
        endpoint_path = "/chat/completions"

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if user_instruction:
             messages.append({"role": "user", "content": user_instruction})
        else:
             # Handle missing user instruction if necessary for the model
             logger.warning("chat_completion called without user_instruction.")
             # raise ValueError("User instruction is required")


        data: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Only include optional parameters if they are not None
        if llm_model:
             data["model"] = llm_model
        if stop:
             data["stop"] = stop
        if stop_token_ids:
             data["stop_token_ids"] = stop_token_ids

        response_data = {} # For logging in case of parsing error
        try:
            logger.debug(f"Sending chat completion request to {endpoint_path} with model: {llm_model}")
            response = await self.client.post(endpoint_path, json=data, headers={"Content-Type": "application/json"})
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx
            response_data = response.json()

            # Robust extraction
            choices = response_data.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                 message = choices[0].get("message")
                 if message and isinstance(message, dict):
                      content = message.get("content", "")
                 else:
                      content = ""
            else:
                 content = ""

            if not content:
                 logger.warning(f"Received empty content string from LLM. Full response: {response_data}")

            return content.strip()

        # Let specific httpx errors propagate up or handle them
        except httpx.TimeoutException as e:
            logger.error(f"Request timed out: {e.request.url!r}")
            raise # Re-raise the original exception
        except httpx.RequestError as e:
            logger.error(f"Request error: {e.request.url!r} - {e}")
            raise
        except httpx.HTTPStatusError as e:
             logger.error(f"HTTP error: {e.response.status_code} for {e.request.url!r}")
             try:
                 logger.error(f"Response body: {e.response.text}")
             except Exception:
                 logger.error("Could not read error response body.")
             raise
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
             logger.error(f"Error parsing LLM response: {e}. Response Data: {response_data}")
             # Consider raising a custom exception or returning an error indicator
             raise ValueError(f"Failed to parse LLM response: {e}") from e

    async def close(self):
        """Closes the underlying HTTPX client."""
        await self.client.aclose()
        logger.info(f"LLM_Caller client for {self.client.base_url} closed.")


# # Example use of class and methods (needs to be run in an async context)
# async def main_example():
#     print("Running async LLM_Caller example...")
#     llm = LLM_Caller()
#     try:
#         # Perform chat completion 
#         llm_response = await llm.chat_completion(
#             system_instruction="You are a helpful assistant specialized in medical scenarios.",
#             user_instruction="Help me I have pain in my chest, radiating to my left arm.",
#             llm_model="Model_ID"
            
#         )
#         print("\nLLM Response:")
#         print(llm_response)
#     except Exception as e:
#          print(f"\nAn error occurred during the example: {e}")
#     finally:
#         await llm.close()

# if __name__ == "__main__":

#     asyncio.run(main_example())