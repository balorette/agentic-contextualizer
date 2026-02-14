# Agent-Mode Usage Caps Fix — Design

**Date:** 2026-02-14
**Status:** Approved
**Problem:** Usage caps (TPM throttle + session budget) don't work in agent mode when going through LiteLLM.

## Root Cause

Two separate LLM call paths exist:

1. **Pipeline mode** (`create_llm_provider` -> `RateLimitedProvider` -> `LiteLLMProvider`): Records actual token usage after each call via `throttle.record_usage()`. Works correctly.

2. **Agent mode** (`build_chat_model` -> `ChatLiteLLM` -> LangChain agent loop): The `TokenBudgetMiddleware` calls `throttle.wait_if_needed(estimated)` in `before_model()` but never records actual usage after the call. The TPM sliding window stays empty. `BudgetTracker` is also disconnected — returned to the caller but not wired into the agent loop.

## Approach: `after_model` Hook (Approach 1)

Use LangChain's `AgentMiddleware.after_model()` hook to extract actual token usage from the `AIMessage.usage_metadata` field after each model call.

### Why This Approach

- `after_model()` is a first-class hook in the `AgentMiddleware` API
- `usage_metadata` is LangChain's standard token usage format (`input_tokens`, `output_tokens`, `total_tokens`), populated by all LangChain chat model wrappers including `ChatLiteLLM`
- Keeps all budget logic in one middleware class
- Minimal changes to existing code

### Alternatives Considered

- **`wrap_model_call` interceptor**: More control but more complex; requires implementing sync version (base raises `NotImplementedError`), would need to relocate `before_model` throttle logic.
- **LangChain callback handler**: Framework-standard but lives outside middleware stack, harder to share `throttle` and `budget_tracker` instances.

## Design

### 1. `TokenBudgetMiddleware` Changes

**New `after_model()` method:**
- Extract last `AIMessage` from state messages
- Read `usage_metadata.total_tokens`, `input_tokens`, `output_tokens`
- Call `self.throttle.record_usage(actual_total)` for TPM tracking
- Call `self.budget_tracker.add_usage(input, output, operation)` for session caps
- Raise `BudgetExceededError` if session budget exceeded

**New `budget_tracker` parameter:**
- Optional `BudgetTracker` instance passed to `__init__`
- When present, `after_model` records cumulative usage and enforces limits

**Fallback for missing `usage_metadata`:**
- Store estimate from `before_model` as `self._last_estimate`
- If `usage_metadata` is absent, use `_last_estimate` for `throttle.record_usage()`
- Log warning when falling back to estimate

### 2. `build_token_middleware()` Changes

Accept optional `BudgetTracker` parameter, pass through to `TokenBudgetMiddleware`.

### 3. Factory Wiring

`create_contextualizer_agent_with_budget()` and `create_scoped_agent_with_budget()`:
- Create `BudgetTracker`
- Pass it into `build_token_middleware()`
- Still return the tracker to the caller for summary/printing

### 4. Files Changed

| File | Change |
|---|---|
| `src/agents/middleware/token_budget.py` | Add `after_model()`, `budget_tracker` param, `_last_estimate` |
| `src/agents/llm/chat_model_factory.py` | `build_token_middleware()` accepts `BudgetTracker` |
| `src/agents/factory.py` | Wire `BudgetTracker` into middleware |
| `src/agents/scoper/agent.py` | Wire `BudgetTracker` into middleware |
| `tests/` | Unit tests for after_model, fallback, budget enforcement |

### 5. Unchanged

- `RateLimitedProvider` (pipeline mode) — already works
- `before_model()` logic
- `wrap_tool_call()` logic
- `BudgetTracker` class itself
- Non-budget factory functions
