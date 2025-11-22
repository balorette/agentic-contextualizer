"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


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

    def __init__(self, model_name: str, api_key: str):
        """Initialize the Anthropic provider.

        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key
        """
        self.model_name = model_name
        self.api_key = api_key
        # TODO: Initialize LangChain Anthropic client

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate a response using Claude."""
        # TODO: Implement using LangChain
        raise NotImplementedError("LLM integration will be implemented in next phase")
