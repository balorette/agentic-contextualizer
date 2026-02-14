# Scoped Agent Token Limit Fixes

**Date**: 2026-02-14
**Status**: Approved
**Branch**: feat/litellm-enablement

## Problem

Agent mode (`--mode agent`) fails with token limit errors on larger questions. Root cause analysis identified five issues:

1. **No `max_input_tokens` default** — `TokenBudgetMiddleware.before_model()` never trims because `max_input_tokens` defaults to `None`. Message history grows unbounded as the agent loops through 10-20 tool calls.
2. **`generate_scoped_context` requires LLM to re-emit file contents** — The tool accepts `relevant_files: dict[str, str]`, forcing the LLM to output all file contents as tool call arguments. Same content appears twice in context (tool results + tool call args).
3. **`max_output_tokens` too small** — Default of 4096 is insufficient for complex tool calls with structured arguments.
4. **Token estimator receives provider-prefixed model name** — `litellm.token_counter()` can't parse `"anthropic:claude-sonnet-4-5-20250929"`, falls back to inaccurate 4-chars/token estimate.
5. **Dual TPM throttles** — Agent loop and generation tool create independent `TPMThrottle` instances that don't know about each other's usage.

Pipeline mode is unaffected — it has careful truncation at every stage.

## Approach

Five targeted fixes. No new files, classes, or abstractions. Four files modified.

## Fix 1: Set `max_input_tokens` default

**File**: `config.py`

Change `max_input_tokens` default from `None` to `128_000`. This activates the existing `trim_messages()` logic in `TokenBudgetMiddleware.before_model()` that was never firing. Conservative enough for Claude and GPT-4 class models. Users can override via `LLM_MAX_INPUT_TOKENS` env var.

**Impact**: Single most impactful change. Prevents unbounded message history growth.

## Fix 2: `generate_scoped_context` takes paths, not contents

**File**: `scoper/agent.py`

Change tool signature from:
```python
relevant_files: dict[str, str]   # LLM must output all file contents
```
to:
```python
relevant_file_paths: list[str]   # LLM outputs just paths
```

The tool reads files via `backend` (already in the closure) and applies the same truncation limits that `ScopedGenerator._format_files()` uses. Update the system prompt to instruct the agent to pass paths.

**Rationale**: The agent's editorial value is in deciding WHICH files matter, not physically shuttling data. Expressing file selection as a path list is cheap; re-emitting 60K+ chars of content is expensive and error-prone.

## Fix 3: Raise `max_output_tokens` default

**File**: `config.py`

Change default from `4096` to `16384`. Gives the LLM room for structured tool call arguments and longer generation outputs without being wasteful.

## Fix 4: Strip provider prefix in token estimator

**File**: `llm/chat_model_factory.py`

In `build_token_middleware()`, strip the provider prefix before passing `model_name` to `TokenBudgetMiddleware`. Use existing `_strip_provider_prefix()` from `provider.py:178`.

Ensures `litellm.token_counter()` receives `"claude-sonnet-4-5-20250929"` instead of `"anthropic:claude-sonnet-4-5-20250929"`, producing accurate estimates instead of falling back to character-based approximation.

## Fix 5: Share TPM throttle between agent and generation provider

**File**: `scoper/agent.py`, `llm/chat_model_factory.py`, `llm/provider.py`

Create one `TPMThrottle` instance and share it between:
- `build_token_middleware()` (agent loop rate limiting)
- `create_llm_provider()` (generation tool rate limiting)

Add optional `throttle` parameter to both factory functions. When provided, use the shared instance instead of creating a new one.

## Files Modified

| File | Changes |
|------|---------|
| `config.py` | Default `max_input_tokens=128_000`, `max_output_tokens=16384` |
| `scoper/agent.py` | Tool signature (paths not contents), system prompt update, shared throttle wiring |
| `llm/chat_model_factory.py` | Strip prefix for estimator, accept optional `throttle` param |
| `llm/provider.py` | `create_llm_provider()` accepts optional `throttle` param |

## What stays the same

- Pipeline mode: completely untouched (separate code path)
- All existing tools: `read_file`, `grep_in_files`, `search_for_files`, etc.
- `ScopedAnalyzer` and `ScopedGenerator` internals
- Config loading from env vars (just new defaults)
- Rate limiting and retry logic internals
