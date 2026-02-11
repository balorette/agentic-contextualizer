"""LiteLLM provider for multi-provider LLM support."""

from typing import Optional, Type
from pydantic import BaseModel
import litellm
from .provider import LLMProvider, LLMResponse


class LiteLLMProvider(LLMProvider):
    """LLM provider using LiteLLM for multi-provider support."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """Initialize LiteLLM provider.

        Args:
            model_name: LiteLLM model identifier (e.g., gpt-4o, claude-3-5-sonnet-20241022)
            api_key: API key for the provider (optional for local models)
            base_url: Custom API endpoint URL (maps to api_base)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate response using LiteLLM."""
        raise NotImplementedError("To be implemented in next task")

    def generate_structured(
        self, prompt: str, system: Optional[str] = None, schema: Type[BaseModel] = None
    ) -> BaseModel:
        """Generate structured output using Pydantic schema."""
        raise NotImplementedError("To be implemented in later task")
