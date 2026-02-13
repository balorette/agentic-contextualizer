"""Shared chat model and middleware builder for agent factories.

Consolidates the duplicated LLM setup logic from factory.py and scoper/agent.py
into a single authoritative location.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ..config import Config
from .provider import _resolve_api_key_for_model

if TYPE_CHECKING:
    from ..middleware.token_budget import TokenBudgetMiddleware


def _format_model_name_for_langchain(model_name: str) -> str:
    """Format model name for LangChain's init_chat_model.

    LangChain expects format: "provider:model"
    """
    if ":" in model_name:
        return model_name

    if model_name.startswith("gpt-") or model_name.startswith("o1"):
        return f"openai:{model_name}"
    elif model_name.startswith("claude"):
        return f"anthropic:{model_name}"
    elif model_name.startswith("gemini") or model_name.startswith("vertex"):
        return f"google-genai:{model_name}"
    else:
        return f"anthropic:{model_name}"


def build_chat_model(
    config: Config,
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    use_litellm: bool = False,
    debug: bool = False,
) -> BaseChatModel:
    """Build a chat model instance from config and options.

    Encapsulates LiteLLM-vs-standard branching, API key resolution,
    kwargs construction, and error handling.

    Args:
        config: Application configuration
        model_name: LLM model identifier (with or without provider prefix)
        base_url: Optional custom API endpoint URL
        api_key: Optional API key (resolved from config if not provided)
        use_litellm: Force use of ChatLiteLLM
        debug: Enable verbose logging

    Returns:
        Configured BaseChatModel instance
    """
    should_use_litellm = (
        use_litellm
        or base_url is not None
        or config.llm_provider == "litellm"
    )

    if not api_key:
        api_key = _resolve_api_key_for_model(model_name, config)

    if debug:
        print(f"[DEBUG] Creating LangChain model:")
        print(f"  - Original model_name: {model_name}")
        print(f"  - Using LiteLLM: {should_use_litellm}")
        print(f"  - base_url: {base_url or 'None'}")
        print(f"  - api_key: {'***' + api_key[-4:] if api_key else 'None'}")

    if should_use_litellm:
        return _build_litellm_model(config, model_name, base_url, api_key, debug)
    else:
        return _build_standard_model(model_name, base_url, api_key, debug)


def _build_litellm_model(
    config: Config,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    debug: bool,
) -> BaseChatModel:
    """Build a ChatLiteLLM model instance."""
    from langchain_litellm import ChatLiteLLM

    if debug:
        import os
        os.environ["LITELLM_LOG"] = "DEBUG"

    litellm_kwargs: dict = {
        "model": model_name,
        "temperature": 0.0,
    }

    if api_key:
        litellm_kwargs["api_key"] = api_key
    if base_url:
        litellm_kwargs["api_base"] = base_url
    if config.max_retries:
        litellm_kwargs["max_retries"] = config.max_retries
    if config.timeout:
        litellm_kwargs["request_timeout"] = config.timeout
    if config.max_output_tokens:
        litellm_kwargs["max_tokens"] = config.max_output_tokens

    if debug:
        print(f"  - ChatLiteLLM kwargs:")
        for k, v in litellm_kwargs.items():
            if k == "api_key":
                print(f"      {k}: ***{v[-4:] if v else 'None'}")
            else:
                print(f"      {k}: {v}")

    model = ChatLiteLLM(**litellm_kwargs)

    if debug:
        print(f"  - Created ChatLiteLLM successfully")

    return model


def _build_standard_model(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    debug: bool,
) -> BaseChatModel:
    """Build a standard LangChain chat model via init_chat_model."""
    formatted_model_name = _format_model_name_for_langchain(model_name)

    model_kwargs: dict = {}
    if base_url:
        model_kwargs["base_url"] = base_url
    if api_key:
        model_kwargs["api_key"] = api_key

    if debug:
        print(f"  - Formatted model_name: {formatted_model_name}")

    try:
        return init_chat_model(formatted_model_name, **model_kwargs)
    except Exception as e:
        if "404" in str(e) or "401" in str(e):
            raise RuntimeError(
                f"Failed to initialize model '{formatted_model_name}'.\n"
                f"For custom LiteLLM gateways, set LLM_PROVIDER=litellm in .env\n"
                f"Original error: {e}"
            ) from e
        raise


def build_token_middleware(
    config: Config,
    model_name: str,
) -> "TokenBudgetMiddleware":
    """Build a TokenBudgetMiddleware with TPM throttle from config.

    Args:
        config: Application configuration
        model_name: Model name for token estimation

    Returns:
        Configured TokenBudgetMiddleware
    """
    from .rate_limiting import TPMThrottle
    from .token_estimator import LiteLLMTokenEstimator
    from ..middleware.token_budget import TokenBudgetMiddleware

    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    estimator = LiteLLMTokenEstimator()

    return TokenBudgetMiddleware(
        max_input_tokens=config.max_input_tokens,
        max_tool_output_chars=config.max_tool_output_chars,
        throttle=throttle,
        estimator=estimator,
        model_name=model_name,
    )
