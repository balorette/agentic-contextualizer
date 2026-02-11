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
        """Generate response using LiteLLM.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            LLMResponse with generated content

        Raises:
            RuntimeError: If generation fails with clear error message
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
            }

            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["api_base"] = self.base_url

            response = litellm.completion(**kwargs)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model_name,
                tokens_used=response.usage.total_tokens if hasattr(response, "usage") else None,
            )
        except litellm.AuthenticationError as e:
            provider = self._detect_provider(self.model_name)
            raise RuntimeError(
                f"{provider} authentication failed. "
                f"Set {provider.upper()}_API_KEY in your .env file"
            ) from e
        except litellm.RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded for {self.model_name}") from e
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e

    def _detect_provider(self, model_name: str) -> str:
        """Detect provider from model name for error messages."""
        if model_name.startswith("gpt-") or model_name.startswith("o1"):
            return "OpenAI"
        elif model_name.startswith("claude"):
            return "Anthropic"
        elif model_name.startswith("gemini"):
            return "Google"
        elif model_name.startswith("ollama"):
            return "Ollama"
        else:
            return "LLM Provider"

    def generate_structured(
        self, prompt: str, system: Optional[str] = None, schema: Type[BaseModel] = None
    ) -> BaseModel:
        """Generate structured output using Pydantic schema."""
        raise NotImplementedError("To be implemented in later task")
