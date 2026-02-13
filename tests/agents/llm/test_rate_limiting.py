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

    def test_wait_if_needed_allows_oversized_call_on_empty_window(self):
        """Should allow a call through when estimated exceeds limit but window is empty.

        Without this fix, the throttle loops forever because there are
        no entries to expire, yet estimated_tokens > effective_limit.
        """
        throttle = TPMThrottle(max_tpm=1000, safety_factor=1.0)
        # estimated 5000 >> limit 1000, but window is empty
        waited = throttle.wait_if_needed(5000)
        assert waited == 0.0

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
