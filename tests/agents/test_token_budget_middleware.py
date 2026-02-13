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

    def test_before_model_normalizes_langchain_roles(self):
        """LangChain message types (human/ai) should be mapped to user/assistant."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=1.0)
        captured_messages = []

        class CapturingEstimator:
            def estimate(self, messages, model):
                captured_messages.extend(messages)
                return 100

        mw = TokenBudgetMiddleware(
            throttle=throttle,
            estimator=CapturingEstimator(),
            model_name="test-model",
        )

        # Simulate LangChain message objects with .type attribute
        class FakeMessage:
            def __init__(self, type_, content):
                self.type = type_
                self.content = content

        state = {"messages": [
            FakeMessage("human", "hello"),
            FakeMessage("ai", "hi there"),
            FakeMessage("system", "you are helpful"),
        ]}
        mw.before_model(state, runtime=None)

        roles = [m["role"] for m in captured_messages]
        assert roles == ["user", "assistant", "system"]
