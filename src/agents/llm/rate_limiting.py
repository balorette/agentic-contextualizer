"""Rate limiting for LLM providers — TPM throttle, retry handler, and decorator."""

from __future__ import annotations

import logging
import time
import threading
from collections import deque

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
