"""LiteLLM provider for multi-provider LLM support via LangChain."""

import logging
import warnings
from typing import Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

# Suppress Pydantic serialization warnings caused by ChatLiteLLM returning
# ChatGeneration/AIMessage subtypes with provider-specific metadata fields.
# See chat_model_factory.py for full explanation.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings",
    category=UserWarning,
    module=r"pydantic\.main",
)

from .provider import LLMProvider, LLMResponse, coerce_content

logger = logging.getLogger(__name__)


class LiteLLMProvider(LLMProvider):
    """LLM provider using ChatLiteLLM (LangChain wrapper around LiteLLM).

    Mirrors the AnthropicProvider pattern: construct a LangChain chat model,
    call invoke() for plain generation, with_structured_output() for schemas.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        max_output_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

        client_kwargs: dict = {
            "model": model_name,
            "max_retries": max_retries,
            "request_timeout": timeout,
            "temperature": 0.0,
        }

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["api_base"] = base_url
        if max_output_tokens:
            client_kwargs["max_tokens"] = max_output_tokens

        self.client = ChatLiteLLM(**client_kwargs)

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate response using ChatLiteLLM.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            LLMResponse with generated content

        Raises:
            RuntimeError: If generation fails
        """
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.client.invoke(messages)
            content = coerce_content(response.content)

            return LLMResponse(
                content=content,
                model=self.model_name,
                tokens_used=(
                    response.usage_metadata.get("total_tokens")
                    if hasattr(response, "usage_metadata") and response.usage_metadata
                    else None
                ),
            )
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e

    def generate_structured(
        self, prompt: str, system: Optional[str] = None, schema: Type[BaseModel] = None
    ) -> BaseModel:
        """Generate structured output using LangChain's with_structured_output().

        Args:
            prompt: User prompt
            system: Optional system prompt
            schema: Pydantic model class defining output structure

        Returns:
            Instance of the provided Pydantic schema

        Raises:
            RuntimeError: If generation fails
            ValueError: If schema is not provided
        """
        if schema is None:
            raise ValueError("schema parameter is required")

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        try:
            structured_llm = self.client.with_structured_output(schema)
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            raise RuntimeError(f"Structured generation failed: {str(e)}") from e
