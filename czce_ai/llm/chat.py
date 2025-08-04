# coding: utf-8
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI as AsyncOpenAIClient,
    OpenAI as OpenAIClient,
    RateLimitError,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

from czce_ai.llm.exceptions import ModelProviderError
from czce_ai.llm.message import Message
from czce_ai.utils.log import logger


@dataclass
class LLMChat(ABC):
    name: str
    model: str
    # The role of the assistant message.
    assistant_message_role: str = "assistant"
    # Client parameters
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    # Request parameters
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = 4096
    presence_penalty: Optional[float] = None
    response_format: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    extra_body: Optional[Dict[str, Any]] = None
    # OpenAI clients
    client: Optional[OpenAIClient] = None
    async_client: Optional[AsyncOpenAIClient] = None

    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }

    def _get_client_params(self) -> Dict[str, Any]:
        if not self.api_key:
            logger.error(
                "OPENAI_API_KEY not set. Please set the OPENAI_API_KEY environment variable."
            )
        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}
        return client_params

    def get_client(self) -> OpenAIClient:
        """Returns an OpenAI client."""
        if self.client:
            return self.client
        client_params: Dict[str, Any] = self._get_client_params()
        self.client = OpenAIClient(**client_params)
        return self.client

    def get_async_client(self) -> AsyncOpenAIClient:
        """Returns an asynchronous OpenAI client."""
        if self.async_client:
            return self.async_client
        client_params: Dict[str, Any] = self._get_client_params()
        # Create a new async HTTP client with custom limits
        client_params["http_client"] = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
        )
        self.async_client = AsyncOpenAIClient(**client_params)
        return self.async_client

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """Returns keyword arguments for API requests."""
        # Define base request parameters
        base_params = {
            "frequency_penalty": self.frequency_penalty,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "extra_body": self.extra_body,
        }
        # Filter out None values
        request_params = {k: v for k, v in base_params.items() if v is not None}
        return request_params

    def invoke(
        self, messages: List[Message]
    ) -> Union[ChatCompletion, ParsedChatCompletion]:
        try:
            return self.get_client().chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],  # type: ignore
                **self.request_kwargs,
            )
        except RateLimitError as e:
            logger.error(f"Rate limit error from {self.name} API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.model,
            ) from e
        except APIConnectionError as e:
            logger.error(f"API connection error from {self.name} API: {e}")
            raise ModelProviderError(
                message=str(e), model_name=self.name, model_id=self.model
            ) from e
        except APIStatusError as e:
            logger.error(f"API status error from {self.name} API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.model,
            ) from e
        except Exception as e:
            logger.error(f"Error from {self.name} API: {e}")
            raise ModelProviderError(
                message=str(e), model_name=self.name, model_id=self.model
            ) from e

    def invoke_stream(self, messages: List[Message]) -> Iterator[ChatCompletionChunk]:
        """Send a streaming chat completion request to the OpenAI API."""
        try:
            yield from self.get_client().chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],  # type: ignore
                stream=True,
                stream_options={"include_usage": True},
                **self.request_kwargs,
            )  # type: ignore
        except RateLimitError as e:
            logger.error(f"Rate limit error from {self.name} API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.model,
            ) from e
        except APIConnectionError as e:
            logger.error(f"API connection error from {self.name} API: {e}")
            raise ModelProviderError(
                message=str(e), model_name=self.name, model_id=self.model
            ) from e
        except APIStatusError as e:
            logger.error(f"API status error from {self.name} API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.model,
            ) from e
        except Exception as e:
            logger.error(f"Error from {self.name} API: {e}")
            raise ModelProviderError(
                message=str(e), model_name=self.name, model_id=self.model
            ) from e

    def invoke_parsed(
        self, messages: List[Message], response_format: BaseModel
    ) -> Union[ChatCompletion, ParsedChatCompletion]:
        response_formatted = {
            "type": "json_schema",
            "json_schema": {
                "name": "query",
                "schema": response_format.model_json_schema(),
            },
        }
        try:
            return self.get_client().chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],  # type: ignore
                response_format=response_formatted,
                **self.request_kwargs,
            )
        except RateLimitError as e:
            logger.error(f"Rate limit error from {self.name} API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.model,
            ) from e
        except APIConnectionError as e:
            logger.error(f"API connection error from {self.name} API: {e}")
            raise ModelProviderError(
                message=str(e), model_name=self.name, model_id=self.model
            ) from e
        except APIStatusError as e:
            logger.error(f"API status error from {self.name} API: {e}")
            error_message = e.response.json().get("error", {})
            error_message = (
                error_message.get("message", "Unknown model error")
                if isinstance(error_message, dict)
                else error_message
            )
            raise ModelProviderError(
                message=error_message,
                status_code=e.response.status_code,
                model_name=self.name,
                model_id=self.model,
            ) from e
        except Exception as e:
            logger.error(f"Error from {self.name} API: {e}")
            raise ModelProviderError(
                message=str(e), model_name=self.name, model_id=self.model
            ) from e