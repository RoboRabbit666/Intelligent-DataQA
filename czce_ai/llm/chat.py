from abc import ABC
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx
from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import OpenAI as OpenAIClient
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel

from czce_ai.llm.message import Message
from czce_ai.utils.log import logger

from .error_handler import handle_llm_exceptions


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
        """Returns an OpenAI client.

        Returns:
            OpenAIClient: An instance of the OpenAI client.
        """
        if self.client:
            return self.client
        client_params: Dict[str, Any] = self._get_client_params()
        self.client = OpenAIClient(**client_params)
        return self.client

    def get_async_client(self) -> AsyncOpenAIClient:
        """Returns an asynchronous OpenAI client.

        Returns:
            AsyncOpenAIClient: An instance of the asynchronous OpenAI client.
        """
        if self.async_client:
            return self.async_client
        client_params: Dict[str, Any] = self._get_client_params()
        # Create a new async HTTP client with custom
        client_params["http_client"] = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=180)
        )
        return AsyncOpenAIClient(**client_params)

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """Returns keyword arguments for API requests.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
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

    @handle_llm_exceptions
    def invoke(
        self, messages: List[Message]
    ) -> Union[ChatCompletion, ParsedChatCompletion]:
        return self.get_client().chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            **self.request_kwargs,
        )

    @handle_llm_exceptions
    def invoke_stream(self, messages: List[Message]) -> Iterator[ChatCompletionChunk]:
        """Send a streaming chat completion request to the OpenAI API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Iterator[ChatCompletionChunk]: An iterator of chat completion chunks.
        """
        yield from self.get_client().chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            stream=True,
            stream_options={"include_usage": True},
            **self.request_kwargs,
        )  # type: ignore

    @handle_llm_exceptions
    async def ainvoke(
        self, messages: List[Message]
    ) -> Union[ChatCompletion, ParsedChatCompletion]:
        return await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            **self.request_kwargs,
        )

    @handle_llm_exceptions
    async def ainvoke_stream(
        self, messages: List[Message]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Send a streaming chat completion request to the OpenAI API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            AsyncIterator[ChatCompletionChunk]: An iterator of chat completion chunks.
        """
        async for chunk in await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            stream=True,
            stream_options={"include_usage": True},
            **self.request_kwargs,
        ):
            yield chunk

    @handle_llm_exceptions
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
        return self.get_client().chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            response_format=response_formatted,
            **self.request_kwargs,
        )

    @handle_llm_exceptions
    async def ainvoke_parsed(
        self, messages: List[Message], response_format: BaseModel
    ) -> Union[ChatCompletion, ParsedChatCompletion]:
        response_formatted = {
            "type": "json_schema",
            "json_schema": {
                "name": "query",
                "schema": response_format.model_json_schema(),
            },
        }
        return await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            response_format=response_formatted,
            **self.request_kwargs,
        )