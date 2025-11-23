"""Tests for budget tracking and cost monitoring."""

import pytest
from unittest.mock import Mock
from src.agents.middleware import (
    BudgetTracker,
    TokenUsage,
    BudgetExceededError,
    extract_token_usage_from_response,
)
from src.agents.factory import create_contextualizer_agent_with_budget


class TestTokenUsage:
    """Test TokenUsage dataclass."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage instance."""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            operation="test_op",
        )

        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 1500
        assert usage.operation == "test_op"

    def test_token_usage_cost_estimate(self):
        """Test cost estimation for Claude Sonnet 4.5 pricing."""
        usage = TokenUsage(
            prompt_tokens=1_000_000,  # 1M input tokens
            completion_tokens=1_000_000,  # 1M output tokens
            total_tokens=2_000_000,
        )

        # Pricing: $3/1M input, $15/1M output
        # Expected: $3 + $15 = $18
        assert usage.cost_estimate == 18.0

    def test_token_usage_small_cost(self):
        """Test cost estimation for small usage."""
        usage = TokenUsage(
            prompt_tokens=1000,  # 0.001M tokens
            completion_tokens=500,  # 0.0005M tokens
        )

        # Expected: (1000/1M * $3) + (500/1M * $15)
        # = $0.003 + $0.0075 = $0.0105
        expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(usage.cost_estimate - expected_cost) < 0.0001


class TestBudgetTracker:
    """Test BudgetTracker functionality."""

    def test_tracker_initialization(self):
        """Test creating BudgetTracker with defaults."""
        tracker = BudgetTracker()

        assert tracker.max_tokens == 50000
        assert tracker.max_cost_usd == 5.0
        assert tracker.warn_threshold == 0.8
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_tracker_custom_limits(self):
        """Test creating tracker with custom limits."""
        tracker = BudgetTracker(max_tokens=10000, max_cost_usd=1.0, warn_threshold=0.5)

        assert tracker.max_tokens == 10000
        assert tracker.max_cost_usd == 1.0
        assert tracker.warn_threshold == 0.5

    def test_add_usage_basic(self):
        """Test adding token usage."""
        tracker = BudgetTracker(max_tokens=10000)

        usage = tracker.add_usage(
            prompt_tokens=500, completion_tokens=200, operation="test"
        )

        assert usage.prompt_tokens == 500
        assert usage.completion_tokens == 200
        assert usage.total_tokens == 700
        assert tracker.total_tokens == 700

    def test_add_multiple_usages(self):
        """Test adding multiple token usages."""
        tracker = BudgetTracker(max_tokens=10000)

        tracker.add_usage(prompt_tokens=500, completion_tokens=200, operation="op1")
        tracker.add_usage(prompt_tokens=1000, completion_tokens=400, operation="op2")
        tracker.add_usage(prompt_tokens=300, completion_tokens=100, operation="op3")

        # Total: (500+200) + (1000+400) + (300+100) = 2500
        assert tracker.total_tokens == 2500
        assert tracker.total_prompt_tokens == 1800
        assert tracker.total_completion_tokens == 700

    def test_remaining_tokens(self):
        """Test remaining token calculation."""
        tracker = BudgetTracker(max_tokens=10000)

        tracker.add_usage(prompt_tokens=3000, completion_tokens=1000, operation="test")

        assert tracker.remaining_tokens == 6000  # 10000 - 4000
        assert tracker.usage_percentage == 0.4  # 4000/10000

    def test_usage_percentage(self):
        """Test usage percentage calculation."""
        tracker = BudgetTracker(max_tokens=10000)

        tracker.add_usage(prompt_tokens=5000, completion_tokens=2000, operation="test")

        assert tracker.usage_percentage == 0.7  # 7000/10000

    def test_is_over_budget_tokens(self):
        """Test budget check based on tokens."""
        tracker = BudgetTracker(max_tokens=5000, max_cost_usd=10.0)

        # Under budget
        tracker.add_usage(prompt_tokens=2000, completion_tokens=1000, operation="test")
        assert not tracker.is_over_budget()

        # Over budget (tokens)
        tracker.add_usage(prompt_tokens=3000, completion_tokens=1000, operation="test2")
        assert tracker.is_over_budget()  # 7000 > 5000

    def test_is_over_budget_cost(self):
        """Test budget check based on cost."""
        tracker = BudgetTracker(max_tokens=1_000_000, max_cost_usd=0.01)

        # Add usage that exceeds cost limit but not token limit
        # 10k input + 5k output = (10k/1M * $3) + (5k/1M * $15) = $0.03 + $0.075 = $0.105
        tracker.add_usage(prompt_tokens=10000, completion_tokens=5000, operation="test")

        assert tracker.is_over_budget()  # $0.105 > $0.01

    def test_check_budget_raises_exception(self):
        """Test that check_budget raises exception when over budget."""
        tracker = BudgetTracker(max_tokens=1000)

        tracker.add_usage(prompt_tokens=800, completion_tokens=300, operation="test")

        with pytest.raises(BudgetExceededError, match="Budget exceeded"):
            tracker.check_budget()

    def test_get_summary(self):
        """Test getting usage summary."""
        tracker = BudgetTracker(max_tokens=10000)

        tracker.add_usage(prompt_tokens=1000, completion_tokens=500, operation="scan")
        tracker.add_usage(prompt_tokens=2000, completion_tokens=800, operation="analyze")

        summary = tracker.get_summary()

        assert summary["total_tokens"] == 4300
        assert summary["prompt_tokens"] == 3000
        assert summary["completion_tokens"] == 1300
        assert summary["operations_count"] == 2
        assert summary["is_over_budget"] is False
        assert "breakdown_by_operation" in summary

    def test_breakdown_by_operation(self):
        """Test usage breakdown by operation."""
        tracker = BudgetTracker(max_tokens=10000)

        tracker.add_usage(prompt_tokens=1000, completion_tokens=500, operation="scan")
        tracker.add_usage(prompt_tokens=2000, completion_tokens=800, operation="analyze")
        tracker.add_usage(prompt_tokens=500, completion_tokens=200, operation="scan")

        summary = tracker.get_summary()
        breakdown = summary["breakdown_by_operation"]

        assert "scan" in breakdown
        assert "analyze" in breakdown

        # scan called twice
        assert breakdown["scan"]["count"] == 2
        assert breakdown["scan"]["total_tokens"] == 1500 + 700  # 2200

        # analyze called once
        assert breakdown["analyze"]["count"] == 1
        assert breakdown["analyze"]["total_tokens"] == 2800

    def test_reset(self):
        """Test resetting tracker."""
        tracker = BudgetTracker(max_tokens=10000)

        tracker.add_usage(prompt_tokens=1000, completion_tokens=500, operation="test")
        assert tracker.total_tokens == 1500

        tracker.reset()

        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0
        assert len(tracker._usages) == 0

    def test_warning_at_threshold(self, capsys):
        """Test warning message at threshold."""
        tracker = BudgetTracker(max_tokens=10000, warn_threshold=0.8)

        # Below threshold - no warning
        tracker.add_usage(prompt_tokens=3000, completion_tokens=2000, operation="test1")
        captured = capsys.readouterr()
        assert "Budget warning" not in captured.out

        # At/above threshold - warning
        tracker.add_usage(prompt_tokens=2000, completion_tokens=1500, operation="test2")
        captured = capsys.readouterr()
        assert "Budget warning" in captured.out
        assert "85.0%" in captured.out  # 8500/10000

    def test_warning_only_once(self, capsys):
        """Test that warning is only shown once."""
        tracker = BudgetTracker(max_tokens=10000, warn_threshold=0.5)

        # First time crossing threshold
        tracker.add_usage(prompt_tokens=3000, completion_tokens=2500, operation="test1")
        captured = capsys.readouterr()
        assert "Budget warning" in captured.out

        # Second time - no warning
        tracker.add_usage(prompt_tokens=1000, completion_tokens=500, operation="test2")
        captured = capsys.readouterr()
        assert "Budget warning" not in captured.out


class TestExtractTokenUsage:
    """Test token usage extraction from LLM responses."""

    def test_extract_from_response_metadata(self):
        """Test extracting usage from response_metadata."""
        mock_response = Mock()
        mock_response.response_metadata = {
            "usage": {"input_tokens": 1000, "output_tokens": 500}
        }

        usage = extract_token_usage_from_response(mock_response)

        assert usage is not None
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 1500

    def test_extract_from_usage_metadata(self):
        """Test extracting usage from usage_metadata attribute."""
        mock_response = Mock()
        mock_response.usage_metadata = {"input_tokens": 2000, "output_tokens": 800}
        # No response_metadata
        mock_response.response_metadata = {}

        usage = extract_token_usage_from_response(mock_response)

        assert usage is not None
        assert usage.prompt_tokens == 2000
        assert usage.completion_tokens == 800

    def test_extract_no_usage_data(self):
        """Test extraction when no usage data available."""
        mock_response = Mock()
        # No metadata attributes
        del mock_response.response_metadata
        del mock_response.usage_metadata

        usage = extract_token_usage_from_response(mock_response)

        assert usage is None

    def test_extract_empty_usage(self):
        """Test extraction with empty usage data."""
        mock_response = Mock()
        mock_response.response_metadata = {"usage": {}}

        usage = extract_token_usage_from_response(mock_response)

        assert usage is not None
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestBudgetIntegration:
    """Test budget integration with agent factory."""

    def test_create_agent_with_budget(self):
        """Test creating agent with budget tracker."""
        agent, tracker = create_contextualizer_agent_with_budget(max_tokens=30000)

        assert agent is not None
        assert tracker is not None
        assert isinstance(tracker, BudgetTracker)
        assert tracker.max_tokens == 30000

    def test_create_agent_with_custom_budget(self):
        """Test creating agent with custom budget limits."""
        agent, tracker = create_contextualizer_agent_with_budget(
            max_tokens=20000, max_cost_usd=2.0
        )

        assert tracker.max_tokens == 20000
        assert tracker.max_cost_usd == 2.0

    def test_tracker_initially_empty(self):
        """Test that tracker starts with zero usage."""
        agent, tracker = create_contextualizer_agent_with_budget()

        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0
        assert not tracker.is_over_budget()


class TestCostCalculation:
    """Test cost calculation accuracy."""

    def test_cost_calculation_accuracy(self):
        """Test that cost calculations match expected values."""
        test_cases = [
            # (input_tokens, output_tokens, expected_cost)
            (1000, 500, 0.0105),  # (1k/1M * 3) + (0.5k/1M * 15)
            (5000, 2000, 0.045),  # (5k/1M * 3) + (2k/1M * 15)
            (10000, 5000, 0.105),  # (10k/1M * 3) + (5k/1M * 15)
            (100000, 50000, 1.05),  # (100k/1M * 3) + (50k/1M * 15)
        ]

        for input_tokens, output_tokens, expected in test_cases:
            usage = TokenUsage(
                prompt_tokens=input_tokens, completion_tokens=output_tokens
            )
            assert abs(usage.cost_estimate - expected) < 0.0001

    def test_typical_session_cost(self):
        """Test cost for typical agent session."""
        tracker = BudgetTracker()

        # Typical session:
        # - scan_structure: minimal (no LLM)
        # - extract_metadata: minimal (no LLM)
        # - analyze_code: 3k input, 1k output
        # - generate_context: 5k input, 2k output

        tracker.add_usage(prompt_tokens=3000, completion_tokens=1000, operation="analyze")
        tracker.add_usage(prompt_tokens=5000, completion_tokens=2000, operation="generate")

        # Total tokens: 11k
        # Cost: (8k/1M * 3) + (3k/1M * 15) = 0.024 + 0.045 = $0.069
        expected_cost = (8000 / 1_000_000) * 3.0 + (3000 / 1_000_000) * 15.0

        assert tracker.total_tokens == 11000
        assert abs(tracker.total_cost - expected_cost) < 0.001

    def test_budget_enforcement_prevents_overrun(self):
        """Test that budget enforcement can prevent cost overruns."""
        # Set very low budget
        tracker = BudgetTracker(max_tokens=5000, max_cost_usd=0.05)

        # Add moderate usage
        tracker.add_usage(prompt_tokens=3000, completion_tokens=1500, operation="test")

        # Should be under budget
        assert not tracker.is_over_budget()

        # Add more - should exceed
        tracker.add_usage(prompt_tokens=2000, completion_tokens=1000, operation="test2")

        # Now over budget (tokens)
        assert tracker.is_over_budget()

        # check_budget should raise
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()
