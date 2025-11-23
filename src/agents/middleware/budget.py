"""Token budget tracking for agent sessions."""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    """Track token usage for a single operation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    operation: str = ""

    @property
    def cost_estimate(self) -> float:
        """Estimate cost in USD based on Claude Sonnet 4.5 pricing.

        Pricing as of 2025-01:
        - Input: $3.00 per 1M tokens
        - Output: $15.00 per 1M tokens
        """
        input_cost = (self.prompt_tokens / 1_000_000) * 3.00
        output_cost = (self.completion_tokens / 1_000_000) * 15.00
        return input_cost + output_cost


@dataclass
class BudgetTracker:
    """Track and enforce token budgets for agent sessions.

    This tracker monitors token usage across agent invocations and can
    enforce limits to prevent cost overruns. It's designed to work with
    LangChain's callback system.

    Args:
        max_tokens: Maximum total tokens allowed for the session
        max_cost_usd: Maximum cost in USD allowed for the session
        warn_threshold: Percentage of budget at which to warn (0.0-1.0)

    Example:
        ```python
        from src.agents.middleware.budget import BudgetTracker

        # Create tracker with 50k token limit
        tracker = BudgetTracker(max_tokens=50000, max_cost_usd=5.0)

        # Track usage manually
        tracker.add_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            operation="scan_structure"
        )

        # Check if over budget
        if tracker.is_over_budget():
            print(f"Budget exceeded! Used {tracker.total_tokens} tokens")

        # Get summary
        print(tracker.get_summary())
        ```

    Note:
        Cost estimates are based on Claude Sonnet 4.5 pricing and may
        not be accurate for other models.
    """

    max_tokens: int = 50000
    max_cost_usd: float = 5.0
    warn_threshold: float = 0.8
    _usages: list[TokenUsage] = field(default_factory=list)
    _warned: bool = False

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all operations."""
        return sum(u.total_tokens for u in self._usages)

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens used."""
        return sum(u.prompt_tokens for u in self._usages)

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens used."""
        return sum(u.completion_tokens for u in self._usages)

    @property
    def total_cost(self) -> float:
        """Total estimated cost in USD."""
        return sum(u.cost_estimate for u in self._usages)

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining in budget."""
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def remaining_cost(self) -> float:
        """Cost remaining in budget (USD)."""
        return max(0.0, self.max_cost_usd - self.total_cost)

    @property
    def usage_percentage(self) -> float:
        """Percentage of token budget used (0.0-1.0)."""
        if self.max_tokens == 0:
            return 0.0
        return min(1.0, self.total_tokens / self.max_tokens)

    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        operation: str = "unknown",
    ) -> TokenUsage:
        """Record token usage for an operation.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            operation: Name of the operation (for tracking)

        Returns:
            TokenUsage object for this operation
        """
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            operation=operation,
        )
        self._usages.append(usage)

        # Check if we should warn
        if not self._warned and self.usage_percentage >= self.warn_threshold:
            self._warned = True
            print(
                f"\n⚠️  Budget warning: {self.usage_percentage * 100:.1f}% of token budget used "
                f"({self.total_tokens:,} / {self.max_tokens:,} tokens)"
            )

        return usage

    def is_over_budget(self) -> bool:
        """Check if token or cost budget has been exceeded.

        Returns:
            True if either token or cost limit is exceeded
        """
        return self.total_tokens > self.max_tokens or self.total_cost > self.max_cost_usd

    def check_budget(self) -> None:
        """Check budget and raise error if exceeded.

        Raises:
            BudgetExceededError: If budget has been exceeded
        """
        if self.is_over_budget():
            raise BudgetExceededError(
                f"Budget exceeded! Used {self.total_tokens:,} tokens "
                f"(limit: {self.max_tokens:,}) and ${self.total_cost:.2f} "
                f"(limit: ${self.max_cost_usd:.2f})"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "remaining_tokens": self.remaining_tokens,
            "remaining_cost_usd": round(self.remaining_cost, 4),
            "usage_percentage": round(self.usage_percentage * 100, 1),
            "operations_count": len(self._usages),
            "is_over_budget": self.is_over_budget(),
            "breakdown_by_operation": self._get_breakdown(),
        }

    def _get_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get token usage breakdown by operation."""
        breakdown = {}
        for usage in self._usages:
            if usage.operation not in breakdown:
                breakdown[usage.operation] = {
                    "count": 0,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                }
            breakdown[usage.operation]["count"] += 1
            breakdown[usage.operation]["total_tokens"] += usage.total_tokens
            breakdown[usage.operation]["prompt_tokens"] += usage.prompt_tokens
            breakdown[usage.operation]["completion_tokens"] += usage.completion_tokens
            breakdown[usage.operation]["cost_usd"] += usage.cost_estimate

        # Round costs for readability
        for op in breakdown.values():
            op["cost_usd"] = round(op["cost_usd"], 4)

        return breakdown

    def print_summary(self) -> None:
        """Print a formatted summary of token usage."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TOKEN BUDGET SUMMARY")
        print("=" * 60)
        print(f"Total Tokens:       {summary['total_tokens']:,} / {self.max_tokens:,}")
        print(f"  - Prompt:         {summary['prompt_tokens']:,}")
        print(f"  - Completion:     {summary['completion_tokens']:,}")
        print(f"Total Cost:         ${summary['total_cost_usd']:.4f} / ${self.max_cost_usd:.2f}")
        print(f"Usage:              {summary['usage_percentage']:.1f}%")
        print(f"Operations:         {summary['operations_count']}")
        print(f"Over Budget:        {'Yes ❌' if summary['is_over_budget'] else 'No ✅'}")

        if summary["breakdown_by_operation"]:
            print("\nBreakdown by Operation:")
            print("-" * 60)
            for op_name, stats in summary["breakdown_by_operation"].items():
                print(
                    f"  {op_name}:"
                    f" {stats['total_tokens']:,} tokens "
                    f"(${stats['cost_usd']:.4f}) "
                    f"× {stats['count']}"
                )

        print("=" * 60 + "\n")

    def reset(self) -> None:
        """Reset all usage tracking."""
        self._usages = []
        self._warned = False


class BudgetExceededError(Exception):
    """Raised when token or cost budget is exceeded."""

    pass


def extract_token_usage_from_response(response: Any) -> Optional[TokenUsage]:
    """Extract token usage from an LLM response.

    Args:
        response: Response object from LangChain model invocation

    Returns:
        TokenUsage object if usage data is available, None otherwise

    Example:
        ```python
        response = llm.invoke("Hello")
        usage = extract_token_usage_from_response(response)
        if usage:
            tracker.add_usage(
                usage.prompt_tokens,
                usage.completion_tokens,
                "llm_call"
            )
        ```
    """
    # Try to extract usage from response metadata
    if hasattr(response, "response_metadata"):
        metadata = response.response_metadata
        if "usage" in metadata:
            usage_data = metadata["usage"]
            return TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get(
                    "input_tokens", 0
                ) + usage_data.get("output_tokens", 0),
            )

    # Try alternative metadata structure
    if hasattr(response, "usage_metadata"):
        usage_data = response.usage_metadata
        return TokenUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

    return None
