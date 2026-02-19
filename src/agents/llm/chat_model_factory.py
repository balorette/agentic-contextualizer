"""Shared chat model and middleware builder for agent factories.

Consolidates the duplicated LLM setup logic from factory.py and scoper/agent.py
into a single authoritative location.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ..config import Config
from .provider import _resolve_api_key_for_model, _strip_provider_prefix

if TYPE_CHECKING:
    from .rate_limiting import TPMThrottle
    from ..middleware.token_budget import TokenBudgetMiddleware
    from ..middleware.budget import BudgetTracker

logger = logging.getLogger(__name__)


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
        logger.debug("Creating LangChain model:")
        logger.debug("  - Original model_name: %s", model_name)
        logger.debug("  - Using LiteLLM: %s", should_use_litellm)
        logger.debug("  - base_url: %s", base_url or "None")
        logger.debug("  - api_key: %s", "set" if api_key else "None")

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

    # Suppress Pydantic serialization warnings caused by ChatLiteLLM returning
    # ChatGeneration/AIMessage subtypes with provider-specific metadata fields
    # (e.g. Anthropic's cache_creation).  Pydantic v2 warns when serializing
    # union members whose actual type carries extra fields beyond the base
    # schema, but the data still round-trips correctly.
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings",
        category=UserWarning,
        module=r"pydantic\.main",
    )

    if debug:
        import litellm
        litellm.set_verbose = True

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
        safe_kwargs = {
            k: ("set" if k == "api_key" else v)
            for k, v in litellm_kwargs.items()
        }
        logger.debug("  - ChatLiteLLM kwargs: %s", safe_kwargs)

    model = ChatLiteLLM(**litellm_kwargs)

    if debug:
        logger.debug("  - Created ChatLiteLLM successfully")

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
        logger.debug("  - Formatted model_name: %s", formatted_model_name)

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
    throttle: "TPMThrottle | None" = None,
    budget_tracker: "BudgetTracker | None" = None,
) -> "TokenBudgetMiddleware":
    """Build a TokenBudgetMiddleware with TPM throttle from config.

    Args:
        config: Application configuration
        model_name: Model name for token estimation
        throttle: Optional shared TPMThrottle instance. If None, a new one
            is created from config values.
        budget_tracker: Optional BudgetTracker for cumulative cost enforcement.

    Returns:
        Configured TokenBudgetMiddleware
    """
    from .rate_limiting import TPMThrottle
    from .token_estimator import LiteLLMTokenEstimator
    from ..middleware.token_budget import TokenBudgetMiddleware

    if throttle is None:
        throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    estimator = LiteLLMTokenEstimator()

    return TokenBudgetMiddleware(
        max_input_tokens=config.max_input_tokens,
        max_tool_output_chars=config.max_tool_output_chars,
        throttle=throttle,
        estimator=estimator,
        model_name=_strip_provider_prefix(model_name),
        budget_tracker=budget_tracker,
    )
