"""Rate limiting for LLM providers — TPM throttle, retry handler, and decorator."""

from __future__ import annotations

import logging
import random
import time
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TPMThrottle:
    """Sliding window token-per-minute throttle.

    Tracks actual token usage in a configurable window and blocks
    when the estimated next call would exceed the budget.

    Thread-safe via threading.Lock.
    """

    def __init__(self, max_tpm: int, safety_factor: float = 0.85):
        self.max_tpm = max_tpm
        self.effective_limit = int(max_tpm * safety_factor)
        self._window_seconds: float = 60.0
        self._lock = threading.Lock()
        self._usage_log: deque[tuple[float, int]] = deque()

    def _evict_old(self) -> None:
        """Remove entries outside the sliding window. Caller must hold _lock."""
        cutoff = time.monotonic() - self._window_seconds
        while self._usage_log and self._usage_log[0][0] < cutoff:
            self._usage_log.popleft()

    @property
    def current_usage(self) -> int:
        """Tokens consumed in the current window."""
        with self._lock:
            self._evict_old()
            return sum(tokens for _, tokens in self._usage_log)

    @property
    def remaining_budget(self) -> int:
        """Tokens available before hitting the effective limit."""
        return max(0, self.effective_limit - self.current_usage)

    def record_usage(self, actual_tokens: int) -> None:
        """Record actual token usage after a call completes."""
        with self._lock:
            self._usage_log.append((time.monotonic(), actual_tokens))

    def wait_if_needed(self, estimated_tokens: int) -> float:
        """Block until safe to send estimated_tokens. Returns seconds waited."""
        total_waited = 0.0
        while True:
            with self._lock:
                self._evict_old()
                usage = sum(tokens for _, tokens in self._usage_log)
                if usage + estimated_tokens <= self.effective_limit:
                    return total_waited

                # Calculate wait time until enough old entries expire
                needed = (usage + estimated_tokens) - self.effective_limit
                freed = 0
                wait_until = time.monotonic()
                for ts, count in self._usage_log:
                    freed += count
                    if freed >= needed:
                        wait_until = ts + self._window_seconds
                        break
                else:
                    wait_until = time.monotonic() + self._window_seconds

                wait_seconds = max(0.1, wait_until - time.monotonic())

            logger.info(
                "TPM throttle: waiting %.1fs (%d/%d tokens used in window)",
                wait_seconds,
                usage,
                self.effective_limit,
            )
            time.sleep(wait_seconds)
            total_waited += wait_seconds


class TPMExhaustedError(Exception):
    """Raised when max retry attempts exhausted on 429s."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class TokenBudgetExceededError(Exception):
    """Raised when a single call exceeds max_tokens_per_call."""

    pass


@dataclass
class RateLimitInfo:
    """Provider-reported rate limit state."""

    input_tokens_remaining: Optional[int] = None
    output_tokens_remaining: Optional[int] = None
    reset_at: Optional[datetime] = None


class RetryHandler:
    """Handles 429 retries with exponential backoff and header extraction.

    Backoff schedule: initial_wait * 2^attempt with +/-25% jitter.
    Respects retry-after header when available.
    """

    _RATE_LIMIT_KEYWORDS = ("rate limit", "rate_limit", "429", "too many requests")

    def __init__(self, max_attempts: int = 3, initial_wait: float = 2.0):
        self.max_attempts = max_attempts
        self.initial_wait = initial_wait

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error."""
        msg = str(error).lower()
        return any(kw in msg for kw in self._RATE_LIMIT_KEYWORDS)

    def execute_with_retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call fn, retrying on rate limit errors with exponential backoff."""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not self._is_rate_limit_error(e):
                    raise
                last_error = e
                if attempt + 1 >= self.max_attempts:
                    break
                wait = self.initial_wait * (2 ** attempt)
                jitter = wait * 0.25 * (2 * random.random() - 1)
                wait = max(0.01, wait + jitter)
                logger.warning(
                    "Rate limited by provider. Retry %d/%d in %.1fs",
                    attempt + 1,
                    self.max_attempts,
                    wait,
                )
                time.sleep(wait)

        raise TPMExhaustedError(
            f"Rate limit: {self.max_attempts}/{self.max_attempts} retries exhausted. "
            f"Last error: {last_error}",
        )

    def extract_rate_limit_info(self, headers: dict) -> Optional[RateLimitInfo]:
        """Extract rate limit info from response headers.

        Supports Anthropic and OpenAI header formats.
        Returns None for unrecognized providers.
        """
        # Anthropic headers
        if "anthropic-ratelimit-input-tokens-remaining" in headers:
            reset_at = None
            reset_str = headers.get("anthropic-ratelimit-input-tokens-reset")
            if reset_str:
                try:
                    reset_at = datetime.fromisoformat(reset_str)
                except ValueError:
                    pass
            return RateLimitInfo(
                input_tokens_remaining=int(
                    headers["anthropic-ratelimit-input-tokens-remaining"]
                ),
                output_tokens_remaining=int(
                    headers.get("anthropic-ratelimit-output-tokens-remaining", 0)
                ) or None,
                reset_at=reset_at,
            )

        # OpenAI headers
        if "x-ratelimit-remaining-tokens" in headers:
            return RateLimitInfo(
                input_tokens_remaining=int(headers["x-ratelimit-remaining-tokens"]),
            )

        return None
