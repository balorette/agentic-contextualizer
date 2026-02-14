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
        throttle: Optional["TPMThrottle"] = None,
        estimator: Optional["TokenEstimator"] = None,
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
                    raw_role = m.get("role", "user")
                else:
                    raw_role = getattr(m, "type", "user")
                # Map LangChain types to OpenAI-style roles
                role = {"human": "user", "ai": "assistant"}.get(raw_role, raw_role)
                content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", str(m))
                msg_dicts.append({"role": role, "content": content})
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
                    # Truncate as a plain string with a clear marker so
                    # downstream code doesn't try to parse broken JSON.
                    result.content = (
                        f"[truncated tool output — {len(serialized)} chars, limit {self.max_tool_output_chars}]\n"
                        + serialized[: self.max_tool_output_chars]
                    )

        return result
