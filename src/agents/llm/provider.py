"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Optional
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
            model=model_name,
            anthropic_api_key=api_key,
            max_retries=max_retries,
            timeout=timeout
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

            return LLMResponse(
                content=response.content,
                model=self.model_name,
                tokens_used=response.usage_metadata.get('total_tokens') if hasattr(response, 'usage_metadata') else None
            )
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e
