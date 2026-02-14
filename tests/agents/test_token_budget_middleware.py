"""Tests for TokenBudgetMiddleware TPM throttle integration."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.middleware.token_budget import TokenBudgetMiddleware
from agents.middleware.budget import BudgetTracker, BudgetExceededError
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


class TestTokenBudgetMiddlewareAfterModel:
    """Tests for after_model hook — records actual usage post-call."""

    def _make_middleware(self, throttle=None, estimator=None, budget_tracker=None):
        return TokenBudgetMiddleware(
            throttle=throttle or TPMThrottle(max_tpm=100000, safety_factor=1.0),
            estimator=estimator or FakeEstimator(value=500),
            model_name="test-model",
            budget_tracker=budget_tracker,
        )

    def test_after_model_records_actual_usage_to_throttle(self):
        """after_model extracts usage_metadata from AIMessage and records to throttle."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        mw = self._make_middleware(throttle=throttle)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        mw.after_model(state, runtime=None)

        assert throttle.current_usage == 280

    def test_after_model_records_to_budget_tracker(self):
        """after_model feeds usage into BudgetTracker when present."""
        tracker = BudgetTracker(max_tokens=50000)
        mw = self._make_middleware(budget_tracker=tracker)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        mw.after_model(state, runtime=None)

        assert tracker.total_tokens == 400
        assert tracker.total_prompt_tokens == 300
        assert tracker.total_completion_tokens == 100

    def test_after_model_raises_on_budget_exceeded(self):
        """after_model raises BudgetExceededError when budget is blown."""
        tracker = BudgetTracker(max_tokens=100)
        mw = self._make_middleware(budget_tracker=tracker)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 80, "output_tokens": 50, "total_tokens": 130},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        with pytest.raises(BudgetExceededError):
            mw.after_model(state, runtime=None)

    def test_after_model_falls_back_to_estimate(self):
        """When AIMessage has no usage_metadata, fall back to last estimate."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        estimator = FakeEstimator(value=750)
        mw = self._make_middleware(throttle=throttle, estimator=estimator)

        # Simulate before_model storing the estimate
        state_before = {"messages": [{"role": "user", "content": "hi"}]}
        mw.before_model(state_before, runtime=None)

        # AIMessage without usage_metadata
        ai_msg = AIMessage(content="response")
        state_after = {"messages": [HumanMessage(content="hi"), ai_msg]}

        mw.after_model(state_after, runtime=None)

        # Should have recorded the estimate (750)
        assert throttle.current_usage == 750

    def test_after_model_no_messages_is_noop(self):
        """after_model with empty messages does nothing."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        mw = self._make_middleware(throttle=throttle)

        state = {"messages": []}
        mw.after_model(state, runtime=None)

        assert throttle.current_usage == 0

    def test_after_model_no_ai_message_is_noop(self):
        """after_model with no AIMessage at end does nothing."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        mw = self._make_middleware(throttle=throttle)

        state = {"messages": [HumanMessage(content="hi")]}
        mw.after_model(state, runtime=None)

        assert throttle.current_usage == 0

    def test_after_model_without_throttle_is_noop(self):
        """after_model with no throttle configured does nothing."""
        mw = TokenBudgetMiddleware(max_input_tokens=5000)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        # Should not raise
        mw.after_model(state, runtime=None)


class TestTokenBudgetMiddlewareRoundTrip:
    """Integration test: before_model + after_model across multiple turns."""

    def test_multiple_turns_accumulate_in_throttle_and_tracker(self):
        """Simulate 3 agent turns and verify cumulative tracking."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        tracker = BudgetTracker(max_tokens=50000)
        estimator = FakeEstimator(value=500)
        mw = TokenBudgetMiddleware(
            throttle=throttle,
            estimator=estimator,
            model_name="test-model",
            budget_tracker=tracker,
        )

        # Turn 1: 200 tokens
        state1 = {"messages": [{"role": "user", "content": "turn 1"}]}
        mw.before_model(state1, runtime=None)
        ai1 = AIMessage(
            content="resp 1",
            usage_metadata={"input_tokens": 150, "output_tokens": 50, "total_tokens": 200},
        )
        mw.after_model({"messages": [HumanMessage(content="turn 1"), ai1]}, runtime=None)

        # Turn 2: 400 tokens
        state2 = {"messages": [
            HumanMessage(content="turn 1"), ai1,
            HumanMessage(content="turn 2"),
        ]}
        mw.before_model(state2, runtime=None)
        ai2 = AIMessage(
            content="resp 2",
            usage_metadata={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400},
        )
        mw.after_model(
            {"messages": [HumanMessage(content="turn 1"), ai1, HumanMessage(content="turn 2"), ai2]},
            runtime=None,
        )

        # Turn 3: 600 tokens
        state3 = {"messages": [
            HumanMessage(content="turn 1"), ai1,
            HumanMessage(content="turn 2"), ai2,
            HumanMessage(content="turn 3"),
        ]}
        mw.before_model(state3, runtime=None)
        ai3 = AIMessage(
            content="resp 3",
            usage_metadata={"input_tokens": 450, "output_tokens": 150, "total_tokens": 600},
        )
        mw.after_model(
            {"messages": [
                HumanMessage(content="turn 1"), ai1,
                HumanMessage(content="turn 2"), ai2,
                HumanMessage(content="turn 3"), ai3,
            ]},
            runtime=None,
        )

        # Throttle: 200 + 400 + 600 = 1200
        assert throttle.current_usage == 1200

        # Tracker: same total
        assert tracker.total_tokens == 1200
        assert tracker.total_prompt_tokens == 900  # 150 + 300 + 450
        assert tracker.total_completion_tokens == 300  # 50 + 100 + 150
