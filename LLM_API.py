import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union # Union added for Path|str

# Assuming load_yaml is defined elsewhere, e.g.:
import yaml

def load_yaml(yaml_filepath: Union[str, Path]) -> dict:
    """
    Read in YAML files.

    Args:
        yaml_filepath: Filepath to yaml file to be read.

    Returns:
        Contents of the YAML file as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
        ValueError: If there's an error parsing the YAML content.
    """
    try:
        # Ensure Path objects are converted to strings if needed by open()
        filepath_str = str(yaml_filepath)
        with open(filepath_str, 'r') as file:
            data = yaml.safe_load(file)
        # Ensure data is a dictionary, handle empty files or non-dict structures
        if data is None:
             logger.warning(f"YAML file is empty or invalid: {filepath_str}")
             return {}
        if not isinstance(data, dict):
             raise ValueError(f"YAML file does not contain a dictionary structure: {filepath_str}")
        return data
    except FileNotFoundError:
        logger.error(f"YAML file not found: {yaml_filepath}")
        raise FileNotFoundError(f"File not found: {yaml_filepath}")
    except IOError as e:
        logger.error(f"IOError reading YAML file {yaml_filepath}: {e}")
        raise IOError(f"Error reading file: {e}")
    except yaml.YAMLError as e:
        logger.error(f"YAMLError parsing YAML file {yaml_filepath}: {e}")
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error loading YAML file {yaml_filepath}: {e}")
        raise

# Configure logging (basic example)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError, AuthenticationError, InternalServerError, APITimeoutError


class LLM_Caller:

    def __init__(self,
                 base_url: str = "http://worker-1:8888/v1/",
                 api_key: str = "test",
                 default_model: Optional[str] = None, # Added default model
                 prompt_filepath: Optional[Union[str, Path]] = "prompts.yaml"):
        """
        Initialize the async LLM API caller object using the openai library.

        Args:
            base_url: URL address of the OpenAI-compatible API endpoint (e.g., "http://localhost:8000/v1").
            api_key: API key for the endpoint. Often a placeholder like "test" or "EMPTY" for local models.
            default_model: The default model name to use if not specified in chat_completion.
                           Required by the openai library for chat completions.
            prompt_filepath: Filepath to prompts catalogue in yaml format. Defaults to "prompts.yaml".
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        if not default_model:
            logger.warning("No default_model provided. Calls to chat_completion MUST specify llm_model.")
            # Or raise ValueError("default_model must be provided") depending on desired strictness

        self.base_url = base_url # Store for reference
        self.api_key = api_key # Store for reference
        self.default_model = default_model # Store default model

        # Initialize AsyncOpenAI client here
        # Increased timeout for potentially long LLM operations
        try:
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120.0,
            )
            logger.info(f"Async OpenAI LLM_Caller initialized for base URL: {self.client.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}")
            raise

        # Load prompts synchronously during init
        try:
            self.prompt_catalogue = load_yaml(prompt_filepath) if prompt_filepath is not None else None
        except Exception as e:
            logger.error(f"Failed to load prompt catalogue from {prompt_filepath}: {e}")
            self.prompt_catalogue = None # Handle initialization failure gracefully

        self.available_models: Optional[List[dict]] = None # Initialize as None

    def __repr__(self):
        """
        Returns: Description of this object class.
        """
        status = f"Models listed ({len(self.available_models)})" if self.available_models is not None else "Models not listed yet (call await list_models())"
        return f"Async OpenAI LLM API Caller (Base URL: {self.base_url}, {status})."

    async def list_models(self) -> Optional[List[dict]]:
        """
        ASYNCHRONOUSLY pull information on all the models currently hosted via the API.
        Updates self.available_models upon successful completion.

        Note: Compatibility of the '/models' endpoint accessed via client.models.list()
              depends on the specific OpenAI-compatible server implementation.

        Returns:
            List of models available (as dictionaries) and their details, or None if an error occurs.
        """
        try:
            logger.debug("Attempting to list models...")
            models_response = await self.client.models.list()
            # The response object is iterable and contains model objects. Convert to dicts.
            # Use model_dump() or dict() depending on pydantic version used by openai lib
            try:
                 # Newer Pydantic v2+ style
                 data = [model.model_dump() for model in models_response.data]
            except AttributeError:
                 # Older Pydantic v1 style or if model_dump isn't present
                 data = [model.dict() for model in models_response.data]

            self.available_models = data # Update instance attribute
            logger.info(f"Successfully listed {len(data)} models.")
            return data

        except AuthenticationError as e:
            logger.error(f"Authentication error listing models: {e}. Check API Key.")
            self.available_models = None
            return None
        except APIConnectionError as e:
            logger.error(f"Connection error listing models from {self.base_url}: {e}")
            self.available_models = None
            return None
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded listing models: {e}")
            self.available_models = None
            return None
        except InternalServerError as e:
             logger.error(f"Internal server error listing models: {e}")
             self.available_models = None
             return None
        except APITimeoutError as e:
             logger.error(f"Timeout listing models: {e}")
             self.available_models = None
             return None
        except OpenAIError as e: # Catch other OpenAI specific errors
            logger.error(f"OpenAI API error listing models: {e}")
            self.available_models = None
            return None
        except Exception as e: # Catch other potential errors
            logger.error(f"Unexpected error listing models: {e}")
            self.available_models = None
            return None

    # _update_prompt_catalogue remains synchronous as it deals with local file I/O
    def _update_prompt_catalogue(self, prompt_filepath: Union[str, Path] = "prompts.yaml"):
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

    async def chat_completion(self,
                              system_instruction: str,
                              user_instruction: str,
                              llm_model: Optional[str] = None,
                              max_tokens: int = 2000,
                              temperature: float = 0.0, # Changed default to 0.0 for consistency
                              stop: Optional[Union[str, List[str]]] = None, # OpenAI lib supports str or list[str]
                              # stop_token_ids removed as it's not standard in openai lib chat completions create method
                             ) -> str:
        """
        ASYNCHRONOUSLY uses chat completion functionality via the openai library.

        Args:
            system_instruction: System instruction part of the prompt. Can be empty string if not needed.
            user_instruction: User instruction part of the prompt.
            llm_model: Language model to use (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B").
                       If None, uses the default_model set during initialization.
            max_tokens: Maximum number of tokens to generate. Defaults to 2000.
            temperature: Controls randomness. Lower values make output more focused. Defaults to 0.0.
            stop: Sequence(s) where the API will stop generating further tokens. Can be a string or list of strings.

        Returns:
            LLM generated response string. Returns empty string if generation fails or yields no content.

        Raises:
            OpenAIError: For API-related errors (connection, authentication, rate limits, etc.).
            ValueError: If essential parameters (like model) are missing or response parsing fails.
        """
        target_model = llm_model or self.default_model
        if not target_model:
             raise ValueError("No llm_model provided and no default_model set during initialization.")

        messages = []
        if system_instruction: # Only add if non-empty
            messages.append({"role": "system", "content": system_instruction})
        if user_instruction:
            messages.append({"role": "user", "content": user_instruction})
        else:

            logger.warning("chat_completion called without user_instruction.")

        if not messages:
             raise ValueError("Cannot call chat completion without any messages (system or user).")

        try:
            logger.debug(f"Sending chat completion request with model: {target_model}")
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                # Other parameters like top_p, frequency_penalty, etc., can be added here if needed
            )

        
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content.strip()
                else:
                    # Log if message object or content is unexpectedly None/empty
                    logger.warning(f"LLM response choice message or content is empty. Role: {message.role if message else 'N/A'}. Full choice: {response.choices[0]}")
                    content = ""
            else:
                logger.warning(f"LLM response did not contain any choices. Full response object: {response}")
                content = ""

            if not content:
                logger.warning(f"Received empty content string from LLM model {target_model}.")


            return content

        except AuthenticationError as e:
            logger.error(f"Authentication error during chat completion: {e}. Check API Key.")
            raise
        except APIConnectionError as e:
            logger.error(f"Connection error during chat completion to {self.base_url}: {e}")
            raise
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded during chat completion: {e}")
            raise
        except InternalServerError as e:
             logger.error(f"Internal server error during chat completion: {e}")
             raise
        except APITimeoutError as e:
             logger.error(f"Timeout during chat completion: {e}")
             raise
        except OpenAIError as e: # Catch other OpenAI specific errors
            # Includes BadRequestError (400) which can happen for invalid params
            logger.error(f"OpenAI API error during chat completion (model: {target_model}): {e}")
            # You might want to inspect e.status_code or e.body here
            raise
        except Exception as e: # Catch unexpected errors during the process
            logger.error(f"Unexpected error during chat completion: {e}")
            # Wrapping in ValueError might obscure original type, consider re-raising e
            raise ValueError(f"Failed during chat completion processing: {e}") from e


    async def close(self):
        """Closes the underlying OpenAI AsyncClient."""
        if hasattr(self, 'client') and self.client:
            await self.client.close()
            logger.info(f"LLM_Caller_OpenAI client for {self.base_url} closed.")
        else:
             logger.warning("Attempted to close LLM_Caller_OpenAI, but client was not initialized.")