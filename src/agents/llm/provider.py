"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

if TYPE_CHECKING:
    from ..config import Config
    from .rate_limiting import TPMThrottle


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
    @staticmethod
    def _rl() -> InMemoryRateLimiter:
        return InMemoryRateLimiter(
            requests_per_second=0.7,  # ~0.7 req/sec, approx one request every ~1.4 seconds
            check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
            max_bucket_size=20,  # Controls the maximum burst size.
        )
    
    def __init__(self, model_name: str, api_key: str, max_retries: int = 3, timeout: int = 60, base_url: Optional[str] = None):
        """Initialize the Anthropic provider.

        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key
            max_retries: Maximum retry attempts for API calls
            timeout: Request timeout in seconds
            base_url: Optional custom API endpoint URL
        """
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = base_url
        self.rate_limiter = self._rl()

        client_kwargs = {
            "model": model_name,
            "anthropic_api_key": api_key,
            "max_retries": max_retries,
            "timeout": timeout,
            "rate_limiter": self.rate_limiter,
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = ChatAnthropic(**client_kwargs)

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


def _strip_provider_prefix(model_name: str) -> str:
    """Strip provider prefix from model names like 'openai:gpt-4o'."""
    return model_name.split(":", 1)[1] if ":" in model_name else model_name


def _resolve_api_key_for_model(model_name: str, config: "Config") -> Optional[str]:
    """Resolve which API key to use based on model name.

    Args:
        model_name: LiteLLM model identifier (may include provider prefix)
        config: Application configuration

    Returns:
        Appropriate API key or None for local models
    """
    normalized = _strip_provider_prefix(model_name)

    if normalized.startswith("gpt-") or normalized.startswith("o1"):
        return config.openai_api_key or config.api_key
    elif normalized.startswith("claude"):
        return config.anthropic_api_key or config.api_key
    elif normalized.startswith("gemini") or normalized.startswith("vertex"):
        return config.google_api_key or config.api_key
    elif normalized.startswith("ollama") or normalized.startswith("lmstudio"):
        return None  # Local models don't need API keys
    else:
        # For gateways or unknown providers, fall back to any available key
        return (
            config.api_key
            or config.openai_api_key
            or config.anthropic_api_key
            or config.google_api_key
        )


def create_llm_provider(config: "Config", throttle: "TPMThrottle | None" = None) -> LLMProvider:
    """Factory function to create the appropriate LLM provider.

    Returns a RateLimitedProvider wrapping the inner provider with
    TPM throttling, token estimation, and 429 retry handling.

    Args:
        config: Application configuration
        throttle: Optional shared TPMThrottle instance. If None, a new one
            is created from config values.

    Returns:
        RateLimitedProvider wrapping AnthropicProvider or LiteLLMProvider
    """
    from .rate_limiting import RateLimitedProvider, TPMThrottle, RetryHandler
    from .token_estimator import LiteLLMTokenEstimator

    if config.llm_provider == "litellm":
        from .litellm_provider import LiteLLMProvider

        api_key = _resolve_api_key_for_model(config.model_name, config)

        inner = LiteLLMProvider(
            model_name=config.model_name,
            api_key=api_key,
            base_url=config.api_base_url,
            max_retries=config.max_retries,
            timeout=config.timeout,
            max_output_tokens=config.max_output_tokens,
        )
    else:
        inner = AnthropicProvider(
            model_name=config.model_name,
            api_key=config.anthropic_api_key or config.api_key,
            base_url=config.api_base_url,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    if throttle is None:
        throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)

    return RateLimitedProvider(
        provider=inner,
        throttle=throttle,
        estimator=LiteLLMTokenEstimator(),
        retry_handler=RetryHandler(config.retry_max_attempts, config.retry_initial_wait),
        max_tokens_per_call=config.max_tokens_per_call,
    )
