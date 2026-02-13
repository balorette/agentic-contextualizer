"""Token estimation for pre-call rate limiting."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import litellm

logger = logging.getLogger(__name__)


@runtime_checkable
class TokenEstimator(Protocol):
    """Protocol for estimating token counts before sending to an LLM."""

    def estimate(self, messages: list[dict], model: str) -> int:
        """Return estimated token count for a message list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model identifier (e.g. 'claude-3-5-sonnet-20241022').

        Returns:
            Estimated number of tokens.
        """
        ...


class LiteLLMTokenEstimator:
    """Token estimator using litellm.token_counter().

    Delegates to the appropriate tokenizer per model:
    - OpenAI models: tiktoken (exact)
    - Claude/Gemini: tiktoken approximation (~85-90% accurate)

    Falls back to character-based estimation (~4 chars/token) on errors.
    """

    CHARS_PER_TOKEN_FALLBACK = 4

    def estimate(self, messages: list[dict], model: str) -> int:
        """Estimate tokens using litellm's model-aware counter."""
        try:
            return litellm.token_counter(model=model, messages=messages)
        except Exception:
            logger.debug(
                "litellm.token_counter failed for model %s, using char-based fallback",
                model,
            )
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return max(1, total_chars // self.CHARS_PER_TOKEN_FALLBACK)
