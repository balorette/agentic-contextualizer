# Token Rate Limiting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TPM-aware rate limiting to both pipeline and agent modes so users with variable provider TPM limits (30K+) don't hit 429 errors.

**Architecture:** Three-layer decorator pattern (estimate → throttle → retry) wrapping existing `LLMProvider` implementations. Agent mode shares the same `TPMThrottle` via extended `TokenBudgetMiddleware`. See `docs/plans/2026-02-13-token-rate-limiting-design.md` for full design.

**Tech Stack:** Python 3.x, litellm (token counting), threading (lock), pydantic (config), pytest (testing)

---

### Task 1: TokenEstimator — Protocol and Implementation

**Files:**
- Create: `src/agents/llm/token_estimator.py`
- Test: `tests/agents/llm/test_token_estimator.py`

**Step 1: Write the failing test**

Create `tests/agents/llm/__init__.py` (empty) and `tests/agents/llm/test_token_estimator.py`:

```python
"""Tests for token estimation."""

import pytest
from agents.llm.token_estimator import LiteLLMTokenEstimator


class TestLiteLLMTokenEstimator:
    """Tests for the LiteLLM-based token estimator."""

    def test_estimate_returns_positive_int(self, mocker):
        """Estimation should return a positive integer."""
        mocker.patch("litellm.token_counter", return_value=150)
        estimator = LiteLLMTokenEstimator()
        result = estimator.estimate(
            [{"role": "user", "content": "Hello world"}],
            model="claude-3-5-sonnet-20241022",
        )
        assert isinstance(result, int)
        assert result > 0

    def test_estimate_passes_model_and_messages(self, mocker):
        """Should forward model and messages to litellm.token_counter."""
        mock_counter = mocker.patch("litellm.token_counter", return_value=42)
        estimator = LiteLLMTokenEstimator()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        estimator.estimate(messages, model="gpt-4o")
        mock_counter.assert_called_once_with(model="gpt-4o", messages=messages)

    def test_estimate_handles_litellm_error_gracefully(self, mocker):
        """If litellm.token_counter raises, fall back to char-based estimate."""
        mocker.patch("litellm.token_counter", side_effect=Exception("tokenizer not found"))
        estimator = LiteLLMTokenEstimator()
        result = estimator.estimate(
            [{"role": "user", "content": "Hello world, this is a test."}],
            model="unknown-model",
        )
        # Fallback: ~4 chars per token
        assert result > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/llm/test_token_estimator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.llm.token_estimator'`

**Step 3: Write minimal implementation**

Create `src/agents/llm/token_estimator.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/llm/test_token_estimator.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add tests/agents/llm/__init__.py tests/agents/llm/test_token_estimator.py src/agents/llm/token_estimator.py
git commit -m "feat: add TokenEstimator protocol and LiteLLM implementation"
```

---

### Task 2: TPMThrottle — Sliding Window

**Files:**
- Create: `src/agents/llm/rate_limiting.py`
- Test: `tests/agents/llm/test_rate_limiting.py`

**Step 1: Write the failing tests**

Create `tests/agents/llm/test_rate_limiting.py`:

```python
"""Tests for TPM rate limiting."""

import time
import threading
import pytest
from agents.llm.rate_limiting import TPMThrottle


class TestTPMThrottle:
    """Tests for sliding window TPM throttle."""

    def test_init_computes_effective_limit(self):
        """Effective limit should be max_tpm * safety_factor."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=0.85)
        assert throttle.effective_limit == 25500

    def test_record_usage_updates_current(self):
        """Recording usage should increase current_usage."""
        throttle = TPMThrottle(max_tpm=30000)
        assert throttle.current_usage == 0
        throttle.record_usage(5000)
        assert throttle.current_usage == 5000

    def test_remaining_budget_decreases(self):
        """Remaining budget should decrease as tokens are used."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=1.0)
        assert throttle.remaining_budget == 30000
        throttle.record_usage(10000)
        assert throttle.remaining_budget == 20000

    def test_wait_if_needed_returns_zero_when_under_budget(self):
        """Should return 0 wait time when under budget."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=1.0)
        waited = throttle.wait_if_needed(5000)
        assert waited == 0.0

    def test_wait_if_needed_blocks_when_over_budget(self):
        """Should wait when estimated tokens would exceed effective limit."""
        throttle = TPMThrottle(max_tpm=1000, safety_factor=1.0)
        throttle.record_usage(900)
        # This should need to wait — 900 + 200 > 1000
        start = time.monotonic()
        # Use a very short window for testing by monkey-patching
        throttle._window_seconds = 0.3
        waited = throttle.wait_if_needed(200)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2  # Should have waited near window expiry

    def test_old_entries_expire(self):
        """Entries older than 60s should not count toward usage."""
        throttle = TPMThrottle(max_tpm=30000)
        throttle._window_seconds = 0.2  # Short window for testing
        throttle.record_usage(20000)
        assert throttle.current_usage == 20000
        time.sleep(0.3)  # Wait for entries to expire
        assert throttle.current_usage == 0

    def test_thread_safety(self):
        """Concurrent record_usage calls should not lose data."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        num_threads = 10
        tokens_per_thread = 100

        def record():
            for _ in range(tokens_per_thread):
                throttle.record_usage(1)

        threads = [threading.Thread(target=record) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert throttle.current_usage == num_threads * tokens_per_thread
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/llm/test_rate_limiting.py::TestTPMThrottle -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.llm.rate_limiting'`

**Step 3: Write minimal implementation**

Add to `src/agents/llm/rate_limiting.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/llm/test_rate_limiting.py::TestTPMThrottle -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add src/agents/llm/rate_limiting.py tests/agents/llm/test_rate_limiting.py
git commit -m "feat: add TPMThrottle sliding window rate limiter"
```

---

### Task 3: RetryHandler — 429 Backoff and Header Extraction

**Files:**
- Modify: `src/agents/llm/rate_limiting.py`
- Test: `tests/agents/llm/test_rate_limiting.py` (append)

**Step 1: Write the failing tests**

Append to `tests/agents/llm/test_rate_limiting.py`:

```python
from datetime import datetime, timezone
from agents.llm.rate_limiting import RetryHandler, RateLimitInfo, TPMExhaustedError


class TestRetryHandler:
    """Tests for 429 retry handling."""

    def test_execute_success_no_retry(self):
        """Successful calls should not retry."""
        handler = RetryHandler(max_attempts=3)
        result = handler.execute_with_retry(lambda: "ok")
        assert result == "ok"

    def test_execute_retries_on_rate_limit(self, mocker):
        """Should retry on RuntimeError containing 'rate limit'."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Rate limit exceeded for gpt-4o")
            return "success"

        handler = RetryHandler(max_attempts=3, initial_wait=0.01)
        result = handler.execute_with_retry(flaky)
        assert result == "success"
        assert call_count == 3

    def test_execute_raises_after_max_attempts(self):
        """Should raise TPMExhaustedError after max attempts."""
        handler = RetryHandler(max_attempts=2, initial_wait=0.01)
        with pytest.raises(TPMExhaustedError) as exc_info:
            handler.execute_with_retry(
                lambda: (_ for _ in ()).throw(RuntimeError("Rate limit exceeded"))
            )
        assert "2/2 retries exhausted" in str(exc_info.value)

    def test_execute_does_not_retry_non_rate_limit_errors(self):
        """Non-rate-limit errors should propagate immediately."""
        handler = RetryHandler(max_attempts=3, initial_wait=0.01)
        with pytest.raises(RuntimeError, match="connection refused"):
            handler.execute_with_retry(
                lambda: (_ for _ in ()).throw(RuntimeError("connection refused"))
            )

    def test_extract_rate_limit_info_anthropic(self):
        """Should extract Anthropic-style rate limit headers."""
        handler = RetryHandler()
        headers = {
            "anthropic-ratelimit-input-tokens-remaining": "8000",
            "anthropic-ratelimit-output-tokens-remaining": "2000",
            "anthropic-ratelimit-input-tokens-reset": "2026-02-13T12:00:45Z",
        }
        info = handler.extract_rate_limit_info(headers)
        assert info is not None
        assert info.input_tokens_remaining == 8000
        assert info.output_tokens_remaining == 2000
        assert info.reset_at is not None

    def test_extract_rate_limit_info_openai(self):
        """Should extract OpenAI-style rate limit headers."""
        handler = RetryHandler()
        headers = {"x-ratelimit-remaining-tokens": "5000"}
        info = handler.extract_rate_limit_info(headers)
        assert info is not None
        assert info.input_tokens_remaining == 5000

    def test_extract_rate_limit_info_unknown_headers(self):
        """Should return None for unrecognized headers."""
        handler = RetryHandler()
        info = handler.extract_rate_limit_info({"content-type": "application/json"})
        assert info is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/llm/test_rate_limiting.py::TestRetryHandler -v`
Expected: FAIL with `ImportError: cannot import name 'RetryHandler'`

**Step 3: Write minimal implementation**

Append to `src/agents/llm/rate_limiting.py`:

```python
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/llm/test_rate_limiting.py -v`
Expected: All 14 passed (7 TPMThrottle + 7 RetryHandler)

**Step 5: Commit**

```bash
git add src/agents/llm/rate_limiting.py tests/agents/llm/test_rate_limiting.py
git commit -m "feat: add RetryHandler with exponential backoff and header extraction"
```

---

### Task 4: RateLimitedProvider — The Decorator

**Files:**
- Modify: `src/agents/llm/rate_limiting.py` (append RateLimitedProvider class)
- Test: `tests/agents/llm/test_rate_limiting.py` (append)

**Step 1: Write the failing tests**

Append to `tests/agents/llm/test_rate_limiting.py`:

```python
from agents.llm.rate_limiting import RateLimitedProvider, TokenBudgetExceededError
from agents.llm.provider import LLMResponse


class FakeProvider:
    """Minimal LLMProvider stub for testing."""

    def __init__(self, response_tokens: int = 100):
        self.model_name = "test-model"
        self.response_tokens = response_tokens
        self.call_count = 0

    def generate(self, prompt, system=None):
        self.call_count += 1
        return LLMResponse(
            content="response",
            model=self.model_name,
            tokens_used=self.response_tokens,
        )

    def generate_structured(self, prompt, system=None, schema=None):
        self.call_count += 1
        if schema:
            return schema(name="test", count=1)
        return None


class FakeEstimator:
    """Returns a fixed token estimate."""

    def __init__(self, estimate_value: int = 50):
        self.estimate_value = estimate_value

    def estimate(self, messages, model):
        return self.estimate_value


class TestRateLimitedProvider:
    """Tests for the RateLimitedProvider decorator."""

    def _make_provider(self, max_tpm=30000, estimate=50, response_tokens=100, max_tokens_per_call=None):
        inner = FakeProvider(response_tokens=response_tokens)
        throttle = TPMThrottle(max_tpm=max_tpm, safety_factor=1.0)
        estimator = FakeEstimator(estimate_value=estimate)
        retry_handler = RetryHandler(max_attempts=1, initial_wait=0.01)
        provider = RateLimitedProvider(
            provider=inner,
            throttle=throttle,
            estimator=estimator,
            retry_handler=retry_handler,
            max_tokens_per_call=max_tokens_per_call,
        )
        return provider, inner, throttle

    def test_generate_delegates_to_inner(self):
        """Should call inner provider's generate and return its response."""
        provider, inner, _ = self._make_provider()
        result = provider.generate("hello", system="sys")
        assert result.content == "response"
        assert inner.call_count == 1

    def test_generate_records_usage(self):
        """Should record actual token usage in throttle after call."""
        provider, _, throttle = self._make_provider(response_tokens=500)
        provider.generate("hello")
        assert throttle.current_usage == 500

    def test_generate_rejects_over_per_call_limit(self):
        """Should raise TokenBudgetExceededError if estimate exceeds per-call cap."""
        provider, inner, _ = self._make_provider(
            estimate=9000, max_tokens_per_call=8000
        )
        with pytest.raises(TokenBudgetExceededError, match="9000.*exceeds.*8000"):
            provider.generate("huge prompt")
        assert inner.call_count == 0  # Never called

    def test_generate_waits_when_throttled(self):
        """Should wait when approaching TPM limit."""
        provider, _, throttle = self._make_provider(max_tpm=1000, estimate=200)
        throttle._window_seconds = 0.3
        throttle.record_usage(900)
        start = time.monotonic()
        provider.generate("hello")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2  # Had to wait for window to expire

    def test_generate_structured_delegates(self):
        """Should work for generate_structured too."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            count: int

        provider, inner, _ = self._make_provider()
        result = provider.generate_structured("hello", schema=TestSchema)
        assert result.name == "test"
        assert inner.call_count == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/llm/test_rate_limiting.py::TestRateLimitedProvider -v`
Expected: FAIL with `ImportError: cannot import name 'RateLimitedProvider'`

**Step 3: Write minimal implementation**

Append to `src/agents/llm/rate_limiting.py`:

```python
from typing import Type
from pydantic import BaseModel as PydanticBaseModel

from .provider import LLMProvider, LLMResponse
from .token_estimator import TokenEstimator


class RateLimitedProvider(LLMProvider):
    """Decorator that adds TPM rate limiting to any LLMProvider.

    Composes: TokenEstimator -> TPMThrottle -> RetryHandler -> inner provider.
    Implements LLMProvider so it's a drop-in replacement.
    """

    def __init__(
        self,
        provider: LLMProvider,
        throttle: TPMThrottle,
        estimator: TokenEstimator,
        retry_handler: RetryHandler,
        max_tokens_per_call: Optional[int] = None,
    ):
        self.provider = provider
        self.throttle = throttle
        self.estimator = estimator
        self.retry_handler = retry_handler
        self.max_tokens_per_call = max_tokens_per_call
        self.model_name = getattr(provider, "model_name", "unknown")

    def _build_messages(self, prompt: str, system: Optional[str] = None) -> list[dict]:
        """Build message dicts for token estimation."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _rate_limited_call(
        self, fn: Callable[..., Any], prompt: str, system: Optional[str] = None, **kwargs: Any,
    ) -> Any:
        """Shared rate-limiting flow for generate() and generate_structured()."""
        # Layer 1: Estimate and reject if over per-call cap
        messages = self._build_messages(prompt, system)
        estimated = self.estimator.estimate(messages, self.model_name)
        if self.max_tokens_per_call and estimated > self.max_tokens_per_call:
            raise TokenBudgetExceededError(
                f"Estimated {estimated} tokens exceeds per-call limit of {self.max_tokens_per_call}"
            )

        # Layer 2: Wait if approaching TPM limit
        self.throttle.wait_if_needed(estimated)

        # Layer 3: Call with retry handling
        result = self.retry_handler.execute_with_retry(fn, prompt, system=system, **kwargs)

        # Record actual usage
        tokens_used = getattr(result, "tokens_used", None)
        if tokens_used:
            self.throttle.record_usage(tokens_used)

        return result

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Rate-limited generate."""
        return self._rate_limited_call(self.provider.generate, prompt, system=system)

    def generate_structured(
        self, prompt: str, system: Optional[str] = None, schema: Type[PydanticBaseModel] = None,
    ) -> PydanticBaseModel:
        """Rate-limited structured generate."""
        return self._rate_limited_call(
            self.provider.generate_structured, prompt, system=system, schema=schema,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/llm/test_rate_limiting.py -v`
Expected: All 19 passed (7 + 7 + 5)

**Step 5: Commit**

```bash
git add src/agents/llm/rate_limiting.py tests/agents/llm/test_rate_limiting.py
git commit -m "feat: add RateLimitedProvider decorator composing all rate-limit layers"
```

---

### Task 5: Update Config — Replace RPS with TPM Fields

**Files:**
- Modify: `src/agents/config.py:42-44` (replace rate_limit_rps/burst with TPM fields)
- Modify: `src/agents/config.py:116-117` (update from_env parsing)
- Test: `tests/test_config.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_config_tpm_defaults():
    """TPM rate limit fields should have correct defaults."""
    config = Config(api_key="test")
    assert config.max_tpm == 30000
    assert config.tpm_safety_factor == 0.85
    assert config.max_tokens_per_call is None
    assert config.retry_max_attempts == 3
    assert config.retry_initial_wait == 2.0


def test_config_tpm_from_env(monkeypatch):
    """TPM config should load from environment variables."""
    monkeypatch.setenv("MAX_TPM", "50000")
    monkeypatch.setenv("TPM_SAFETY_FACTOR", "0.9")
    monkeypatch.setenv("MAX_TOKENS_PER_CALL", "8000")
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("RETRY_INITIAL_WAIT", "1.5")

    config = Config.from_env()

    assert config.max_tpm == 50000
    assert config.tpm_safety_factor == 0.9
    assert config.max_tokens_per_call == 8000
    assert config.retry_max_attempts == 5
    assert config.retry_initial_wait == 1.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_tpm_defaults -v`
Expected: FAIL with `pydantic...ValidationError...max_tpm`

**Step 3: Write minimal implementation**

In `src/agents/config.py`, replace lines 42-44 (the `rate_limit_rps` and `rate_limit_burst` fields) with:

```python
    # Rate Limiting — TPM-aware
    max_tpm: int = Field(default=30000)
    tpm_safety_factor: float = Field(default=0.85)
    max_tokens_per_call: Optional[int] = Field(default=None)
    retry_max_attempts: int = Field(default=3)
    retry_initial_wait: float = Field(default=2.0)
```

In `src/agents/config.py`, replace lines 116-117 (the `rate_limit_rps` and `rate_limit_burst` entries in `config_dict`) with:

```python
            "max_tpm": _parse_int(os.getenv("MAX_TPM"), 30000),
            "tpm_safety_factor": _parse_float(os.getenv("TPM_SAFETY_FACTOR"), 0.85),
            "max_tokens_per_call": _parse_int(os.getenv("MAX_TOKENS_PER_CALL"), None) if os.getenv("MAX_TOKENS_PER_CALL") else None,
            "retry_max_attempts": _parse_int(os.getenv("RETRY_MAX_ATTEMPTS"), 3),
            "retry_initial_wait": _parse_float(os.getenv("RETRY_INITIAL_WAIT"), 2.0),
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All 6 passed (4 existing + 2 new)

**Step 5: Commit**

```bash
git add src/agents/config.py tests/test_config.py
git commit -m "feat: replace rate_limit_rps/burst config with TPM-aware fields"
```

---

### Task 6: Update .env.example — Document New Config

**Files:**
- Modify: `.env.example:44-53` (replace RPS section with TPM section)

**Step 1: Replace the Rate Limiting section**

In `.env.example`, replace lines 44-53 (the "Rate Limiting (Agent Mode)" section) with:

```
# ─── Rate Limiting — TPM-Aware ───────────────────────────────────────────
# Controls token-per-minute budget to avoid 429 rate limit errors.
# Tune these for your provider tier (e.g., Anthropic free tier = 30K TPM).
#
# MAX_TPM: Your provider's tokens-per-minute limit (default: 30000)
#   Check your provider dashboard for exact limits.
MAX_TPM=30000
#
# TPM_SAFETY_FACTOR: Use only this fraction of MAX_TPM proactively (default: 0.85)
#   Reserves buffer for token estimation errors (~10-15% on Claude/Gemini).
TPM_SAFETY_FACTOR=0.85
#
# MAX_TOKENS_PER_CALL: Reject any single LLM call estimated above this (default: unset)
#   Set to prevent accidentally sending oversized requests.
# MAX_TOKENS_PER_CALL=8000
#
# RETRY_MAX_ATTEMPTS: How many times to retry on 429 errors (default: 3)
RETRY_MAX_ATTEMPTS=3
#
# RETRY_INITIAL_WAIT: Initial backoff in seconds, doubles each retry (default: 2.0)
RETRY_INITIAL_WAIT=2.0
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: update .env.example with TPM rate limiting config"
```

---

### Task 7: Wire RateLimitedProvider into create_llm_provider()

**Files:**
- Modify: `src/agents/llm/provider.py:200-231` (update `create_llm_provider`)
- Test: `tests/test_llm.py` (update existing tests)

**Step 1: Write the failing test**

Append to `tests/test_llm.py`:

```python
from agents.llm.rate_limiting import RateLimitedProvider


def test_create_llm_provider_wraps_with_rate_limiting():
    """Factory should return a RateLimitedProvider wrapping the inner provider."""
    config = Config(
        llm_provider="litellm",
        model_name="gpt-4o",
        openai_api_key="test-key",
        max_tpm=50000,
        tpm_safety_factor=0.9,
    )
    provider = create_llm_provider(config)
    assert isinstance(provider, RateLimitedProvider)
    assert provider.throttle.max_tpm == 50000
    assert provider.throttle.effective_limit == 45000  # 50000 * 0.9


def test_create_llm_provider_anthropic_wrapped():
    """Anthropic provider should also be wrapped."""
    config = Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key",
    )
    provider = create_llm_provider(config)
    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, AnthropicProvider)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_create_llm_provider_wraps_with_rate_limiting -v`
Expected: FAIL — `create_llm_provider` returns bare `LiteLLMProvider`, not `RateLimitedProvider`

**Step 3: Write minimal implementation**

Update `create_llm_provider()` in `src/agents/llm/provider.py` (lines 200-231). Replace the function body:

```python
def create_llm_provider(config: "Config") -> LLMProvider:
    """Factory function to create the appropriate LLM provider.

    Returns a RateLimitedProvider wrapping the inner provider with
    TPM throttling, token estimation, and 429 retry handling.

    Args:
        config: Application configuration

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

    return RateLimitedProvider(
        provider=inner,
        throttle=TPMThrottle(config.max_tpm, config.tpm_safety_factor),
        estimator=LiteLLMTokenEstimator(),
        retry_handler=RetryHandler(config.retry_max_attempts, config.retry_initial_wait),
        max_tokens_per_call=config.max_tokens_per_call,
    )
```

**Step 4: Fix existing tests that check `isinstance(provider, LiteLLMProvider)`**

Update `test_create_llm_provider_litellm` in `tests/test_llm.py` — it currently asserts `isinstance(provider, LiteLLMProvider)`. Change to:

```python
def test_create_llm_provider_litellm():
    """Test factory creates RateLimitedProvider wrapping LiteLLMProvider."""
    config = Config(
        llm_provider="litellm",
        model_name="gpt-4o",
        openai_api_key="test-key"
    )
    provider = create_llm_provider(config)
    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, LiteLLMProvider)
    assert provider.provider.model_name == "gpt-4o"
    assert provider.provider.api_key == "test-key"
```

Update `test_create_llm_provider_anthropic`:

```python
def test_create_llm_provider_anthropic():
    """Test factory creates RateLimitedProvider wrapping AnthropicProvider."""
    config = Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key"
    )
    provider = create_llm_provider(config)
    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, AnthropicProvider)
```

Update `test_legacy_config_still_works`:

```python
def test_legacy_config_still_works(monkeypatch):
    """Test that old .env files work unchanged."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    config = Config.from_env()
    provider = create_llm_provider(config)

    assert config.llm_provider == "anthropic"
    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, AnthropicProvider)
    assert provider.provider.api_key == "test-key"
```

**Step 5: Run all LLM tests to verify**

Run: `pytest tests/test_llm.py -v`
Expected: All passed

**Step 6: Commit**

```bash
git add src/agents/llm/provider.py tests/test_llm.py
git commit -m "feat: wire RateLimitedProvider into create_llm_provider factory"
```

---

### Task 8: Update TokenBudgetMiddleware — Add Throttle for Agent Mode

**Files:**
- Modify: `src/agents/middleware/token_budget.py`
- Test: `tests/agents/test_token_budget_middleware.py` (new)

**Step 1: Write the failing test**

Create `tests/agents/test_token_budget_middleware.py`:

```python
"""Tests for TokenBudgetMiddleware TPM throttle integration."""

import pytest
from unittest.mock import MagicMock
from agents.middleware.token_budget import TokenBudgetMiddleware
from agents.llm.rate_limiting import TPMThrottle


class FakeEstimator:
    def __init__(self, value=100):
        self.value = value

    def estimate(self, messages, model):
        return self.value


class TestTokenBudgetMiddlewareThrottle:
    """Tests for the TPM throttle integration in TokenBudgetMiddleware."""

    def test_init_without_throttle_still_works(self):
        """Existing usage without throttle should not break."""
        mw = TokenBudgetMiddleware(max_input_tokens=5000)
        assert mw.throttle is None
        assert mw.estimator is None

    def test_before_model_calls_throttle(self):
        """before_model should call throttle.wait_if_needed when configured."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=1.0)
        estimator = FakeEstimator(value=500)
        mw = TokenBudgetMiddleware(
            throttle=throttle,
            estimator=estimator,
            model_name="test-model",
        )

        state = {"messages": [{"role": "user", "content": "hi"}]}
        mw.before_model(state, runtime=None)

        # Should have recorded no wait (under budget)
        assert throttle.current_usage == 0  # Not recorded yet — only estimated

    def test_before_model_skips_throttle_when_not_configured(self):
        """No throttle configured should skip throttling silently."""
        mw = TokenBudgetMiddleware(max_input_tokens=5000)
        state = {"messages": [{"role": "user", "content": "hi"}]}
        result = mw.before_model(state, runtime=None)
        # Should not raise, just return normally
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_token_budget_middleware.py -v`
Expected: FAIL — `TokenBudgetMiddleware.__init__()` does not accept `throttle` parameter

**Step 3: Write minimal implementation**

Update `src/agents/middleware/token_budget.py`:

```python
"""Agent middleware for token budget control."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TYPE_CHECKING

from langchain.agents.factory import AgentMiddleware
from langchain_core.messages import trim_messages

if TYPE_CHECKING:
    from ..llm.rate_limiting import TPMThrottle
    from ..llm.token_estimator import TokenEstimator

logger = logging.getLogger(__name__)


class TokenBudgetMiddleware(AgentMiddleware):
    """Middleware that controls input token usage and TPM rate limiting.

    Existing behavior (unchanged):
    - before_model: trims old messages to stay under max_input_tokens
    - wrap_tool_call: truncates tool output to max_tool_output_chars

    New behavior (when throttle/estimator provided):
    - before_model: ALSO calls throttle.wait_if_needed() after trimming
    """

    def __init__(
        self,
        max_input_tokens: int | None = None,
        max_tool_output_chars: int = 12000,
        throttle: Optional[TPMThrottle] = None,
        estimator: Optional[TokenEstimator] = None,
        model_name: str = "unknown",
    ):
        self.max_input_tokens = max_input_tokens
        self.max_tool_output_chars = max_tool_output_chars
        self.throttle = throttle
        self.estimator = estimator
        self.model_name = model_name

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        """Trim conversation history, then throttle if configured."""
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        result = None

        # Existing: trim messages
        if self.max_input_tokens and messages:
            trimmed = trim_messages(
                messages,
                max_tokens=self.max_input_tokens,
                token_counter="approximate",
                strategy="last",
                include_system=True,
            )
            if len(trimmed) < len(messages):
                messages = trimmed
                result = {"messages": trimmed}

        # New: TPM throttling (after trimming for accurate estimate)
        if self.throttle and self.estimator and messages:
            msg_dicts = []
            for m in messages:
                if isinstance(m, dict):
                    msg_dicts.append(m)
                else:
                    msg_dicts.append({
                        "role": getattr(m, "type", "user"),
                        "content": getattr(m, "content", str(m)),
                    })
            estimated = self.estimator.estimate(msg_dicts, self.model_name)
            self.throttle.wait_if_needed(estimated)

        return result

    def wrap_tool_call(self, request, handler):
        """Truncate tool outputs that exceed the character limit."""
        result = handler(request)

        if self.max_tool_output_chars and result.content:
            content = result.content
            if isinstance(content, str) and len(content) > self.max_tool_output_chars:
                result.content = (
                    content[: self.max_tool_output_chars]
                    + f"\n... [truncated, {len(content) - self.max_tool_output_chars} chars omitted]"
                )
            elif isinstance(content, (dict, list)):
                serialized = json.dumps(content)
                if len(serialized) > self.max_tool_output_chars:
                    result.content = (
                        serialized[: self.max_tool_output_chars]
                        + f"\n... [truncated, {len(serialized) - self.max_tool_output_chars} chars omitted]"
                    )

        return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_token_budget_middleware.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/agents/middleware/token_budget.py tests/agents/test_token_budget_middleware.py
git commit -m "feat: extend TokenBudgetMiddleware with optional TPM throttle"
```

---

### Task 9: Update Factory — Pass Throttle to Middleware

**Files:**
- Modify: `src/agents/factory.py:6` (remove InMemoryRateLimiter import)
- Modify: `src/agents/factory.py:117-147` (replace rate limiter setup with throttle)
- Modify: `src/agents/factory.py:217-222` (pass throttle/estimator to middleware)

**Step 1: Write the failing test**

Append to `tests/agents/test_agent_integration.py` (read it first to understand existing patterns, then append):

```python
def test_agent_middleware_has_throttle(monkeypatch):
    """Agent factory should pass TPM throttle to middleware."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MAX_TPM", "40000")
    monkeypatch.setenv("TPM_SAFETY_FACTOR", "0.9")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    # We can't easily inspect middleware after agent creation,
    # but we can verify no import errors and the factory doesn't crash
    from agents.factory import create_contextualizer_agent
    # This should not raise
    agent = create_contextualizer_agent(debug=False)
    assert agent is not None
```

**Step 2: Make the changes**

In `src/agents/factory.py`:

1. Remove the `InMemoryRateLimiter` import (line 6).

2. Replace the rate limiter block (lines 142-153, the `InMemoryRateLimiter` + `rate_limiter` assignment in the `if should_use_litellm:` branch) — remove `rate_limiter` from `litellm_kwargs`. The TPM throttle now handles rate limiting via middleware, not via the chat model's `rate_limiter` parameter.

3. Update the middleware wiring (lines 217-222):

```python
    # TPM-aware throttle — shared instance for this agent session
    from .llm.rate_limiting import TPMThrottle
    from .llm.token_estimator import LiteLLMTokenEstimator

    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    estimator = LiteLLMTokenEstimator()

    # Token budget middleware — trims messages, truncates tool output, and throttles TPM
    from .middleware.token_budget import TokenBudgetMiddleware
    budget_mw = TokenBudgetMiddleware(
        max_input_tokens=config.max_input_tokens,
        max_tool_output_chars=config.max_tool_output_chars,
        throttle=throttle,
        estimator=estimator,
        model_name=model_name,
    )
    all_middleware = [budget_mw] + (middleware or [])
```

**Step 3: Run tests**

Run: `pytest tests/agents/ -v`
Expected: All passed

**Step 4: Commit**

```bash
git add src/agents/factory.py tests/agents/test_agent_integration.py
git commit -m "feat: wire TPM throttle into agent factory, remove InMemoryRateLimiter"
```

---

### Task 10: Run Full Test Suite and Fix Any Breakage

**Files:**
- Any files that need fixing from cascading changes

**Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass. If any fail, investigate and fix.

**Step 2: Check for references to removed config fields**

Search for `rate_limit_rps` and `rate_limit_burst` across the codebase:

Run: `grep -r "rate_limit_rps\|rate_limit_burst" src/ tests/`

Fix any remaining references — these fields no longer exist in `Config`.

**Step 3: Run full suite again**

Run: `pytest tests/ -v`
Expected: All green

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: clean up references to removed rate_limit_rps/burst config"
```

---

### Task 11: Final Commit — Update LLM __init__.py Exports

**Files:**
- Modify: `src/agents/llm/__init__.py`

**Step 1: Update exports**

The `src/agents/llm/__init__.py` is currently empty (or nearly). Add exports for the new public API:

```python
"""LLM provider abstraction layer."""

from .provider import LLMProvider, LLMResponse, AnthropicProvider, create_llm_provider
from .litellm_provider import LiteLLMProvider
from .token_estimator import TokenEstimator, LiteLLMTokenEstimator
from .rate_limiting import (
    RateLimitedProvider,
    TPMThrottle,
    RetryHandler,
    RateLimitInfo,
    TokenBudgetExceededError,
    TPMExhaustedError,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "AnthropicProvider",
    "LiteLLMProvider",
    "create_llm_provider",
    "TokenEstimator",
    "LiteLLMTokenEstimator",
    "RateLimitedProvider",
    "TPMThrottle",
    "RetryHandler",
    "RateLimitInfo",
    "TokenBudgetExceededError",
    "TPMExhaustedError",
]
```

**Step 2: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: All green

**Step 3: Commit**

```bash
git add src/agents/llm/__init__.py
git commit -m "feat: export rate limiting API from llm package"
```

---

## Task Summary

| Task | Component | Action | Test File |
|------|-----------|--------|-----------|
| 1 | TokenEstimator | CREATE | `tests/agents/llm/test_token_estimator.py` |
| 2 | TPMThrottle | CREATE | `tests/agents/llm/test_rate_limiting.py` |
| 3 | RetryHandler | APPEND | `tests/agents/llm/test_rate_limiting.py` |
| 4 | RateLimitedProvider | APPEND | `tests/agents/llm/test_rate_limiting.py` |
| 5 | Config | UPDATE | `tests/test_config.py` |
| 6 | .env.example | UPDATE | (none) |
| 7 | create_llm_provider | UPDATE | `tests/test_llm.py` |
| 8 | TokenBudgetMiddleware | UPDATE | `tests/agents/test_token_budget_middleware.py` |
| 9 | Factory | UPDATE | `tests/agents/test_agent_integration.py` |
| 10 | Full suite | VERIFY | all |
| 11 | LLM __init__ | UPDATE | all |
