"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    model: str
    tokens_used: Optional[int] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system: Optional system prompt

        Returns:
            LLMResponse containing the generated text
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider using LangChain."""

    def __init__(self, model_name: str, api_key: str, max_retries: int = 3, timeout: int = 60):
        """Initialize the Anthropic provider.

        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key
            max_retries: Maximum retry attempts for API calls
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

        self.client = ChatAnthropic(
            model=model_name, anthropic_api_key=api_key, max_retries=max_retries, timeout=timeout
        )

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate a response using Claude.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            LLMResponse with generated content
        """
        messages = []

        if system:
            messages.append(SystemMessage(content=system))

        messages.append(HumanMessage(content=prompt))

        try:
            response = self.client.invoke(messages)
            content = self._coerce_content(response.content)

            return LLMResponse(
                content=content,
                model=self.model_name,
                tokens_used=(
                    response.usage_metadata.get("total_tokens")
                    if hasattr(response, "usage_metadata")
                    else None
                ),
            )
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e

    def generate_structured(
        self, prompt: str, system: Optional[str] = None, schema: Type[BaseModel] = None
    ) -> BaseModel:
        """Generate structured output using Pydantic schema.

        This method uses LangChain's with_structured_output() to guarantee
        valid structured output without markdown wrapping.

        Args:
            prompt: User prompt
            system: Optional system prompt
            schema: Pydantic model class defining the expected output structure

        Returns:
            Instance of the provided Pydantic schema with validated data

        Raises:
            RuntimeError: If LLM generation fails
            ValueError: If schema is not provided
        """
        if schema is None:
            raise ValueError("schema parameter is required for generate_structured()")

        messages = []

        if system:
            messages.append(SystemMessage(content=system))

        messages.append(HumanMessage(content=prompt))

        try:
            # Use with_structured_output for guaranteed valid output
            structured_llm = self.client.with_structured_output(schema)
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            raise RuntimeError(f"Structured LLM generation failed: {str(e)}") from e

    @staticmethod
    def _coerce_content(content: Any) -> str:
        """Ensure LangChain responses are flattened into plain text."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue

                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
                    continue

                if isinstance(block, dict):
                    text = block.get("text")
                    if text:
                        parts.append(text)
                    continue
            return "".join(parts).strip()

        return str(content)
