# Roadmap: Human-in-the-Loop Approval Gates

**Status:** Not yet implemented
**Prerequisite:** `create_contextualizer_agent_with_checkpointer()` (state persistence)

## Goal

Allow users to approve or reject expensive LLM operations (analyze_code,
generate_context, refine_context) before they execute.

## Current State

- `create_approval_tool()` wrapper exists in `src/agents/middleware/human_in_the_loop.py`
- `get_expensive_tool_approval()` and `should_approve_expensive_operation()` are implemented
- The factory function accepts `require_approval_for` but does NOT wire the wrappers
- No tools currently call `interrupt()` by default

## Implementation Plan

1. In the factory, wrap tools listed in `require_approval_for` with `create_approval_tool()`
2. Pass wrapped tools to `create_agent()` instead of raw tools
3. Add tests for the interrupt/resume flow
4. Update CLI to handle `__interrupt__` state and prompt the user

## References

- `src/agents/middleware/human_in_the_loop.py` - existing wrapper infrastructure
- `src/agents/factory.py` - factory function with `require_approval_for` parameter
- LangGraph interrupt docs: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
