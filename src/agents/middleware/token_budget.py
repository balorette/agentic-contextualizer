"""Agent middleware for token budget control."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.factory import AgentMiddleware
from langchain_core.messages import trim_messages


class TokenBudgetMiddleware(AgentMiddleware):
    """Middleware that controls input token usage.

    Two mechanisms:
    - before_model: trims old messages to stay under max_input_tokens
    - wrap_tool_call: truncates tool output to max_tool_output_chars
    """

    def __init__(
        self,
        max_input_tokens: int | None = None,
        max_tool_output_chars: int = 12000,
    ):
        self.max_input_tokens = max_input_tokens
        self.max_tool_output_chars = max_tool_output_chars

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        """Trim conversation history before each LLM call."""
        if not self.max_input_tokens:
            return None

        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return None

        trimmed = trim_messages(
            messages,
            max_tokens=self.max_input_tokens,
            token_counter="approximate",
            strategy="last",
            include_system=True,
        )

        if len(trimmed) < len(messages):
            return {"messages": trimmed}
        return None

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
