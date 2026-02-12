"""LiteLLM provider for multi-provider LLM support."""

import json
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
        max_output_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

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
            if self.max_output_tokens:
                kwargs["max_tokens"] = self.max_output_tokens

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
        """Generate structured output using Pydantic schema.

        Uses provider-specific structured output when available,
        falls back to JSON mode + parsing for others.

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
            messages.append({"role": "system", "content": system})

        # Add schema instructions to the prompt for better JSON mode results
        schema_json = schema.model_json_schema()
        enhanced_prompt = f"{prompt}\n\nRespond with a JSON object matching this schema:\n{schema_json}"
        messages.append({"role": "user", "content": enhanced_prompt})

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
            if self.max_output_tokens:
                kwargs["max_tokens"] = self.max_output_tokens

            # Use JSON mode for broad compatibility
            # This works across OpenAI, Anthropic, and most other providers
            kwargs["response_format"] = {"type": "json_object"}
            response = litellm.completion(**kwargs)

            # Parse the JSON response and validate against schema
            content = response.choices[0].message.content
            data = json.loads(content)
            return schema(**data)

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse JSON response: {str(e)}\nContent: {content}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Structured generation failed: {str(e)}") from e
