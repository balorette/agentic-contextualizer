# Code Review Refactoring Design

**Date:** 2026-02-13
**Status:** Approved
**Source:** `docs/findings/refactoring-prompt.md` (consolidated from two independent code reviews)

## Goal

Improve structure, clarity, and correctness across the codebase. No new features. All existing tests must continue to pass.

---

## Priority 1 — Extract shared chat model builder

**Problem:** `factory.py:create_contextualizer_agent()` and `scoper/agent.py:create_scoped_agent()` both implement ~60 lines of identical LLM setup: LiteLLM-vs-standard branching, kwargs construction, API key resolution, debug logging, and rate limiter/middleware setup. The scoper is missing error handling present in the factory.

**Design:**

Create `src/agents/llm/chat_model_factory.py` with:

```python
def build_chat_model(
    config: Config,
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    use_litellm: bool = False,
    debug: bool = False,
) -> BaseChatModel:
```

Encapsulates:
- LiteLLM auto-detection (`use_litellm or base_url or config.llm_provider == "litellm"`)
- API key resolution via `_resolve_api_key_for_model`
- `ChatLiteLLM` kwargs (temperature, retries, timeout, max_tokens)
- LiteLLM debug env var (`LITELLM_LOG`)
- `init_chat_model` fallback with 401/404 error handling
- Consistent debug output

Also extract duplicated TPMThrottle + TokenBudgetMiddleware setup into a shared helper (e.g. `build_token_middleware`).

Both factories reduce to: `model = build_chat_model(config, ...)`.

**Files changed:**
- New: `src/agents/llm/chat_model_factory.py`
- Modified: `src/agents/factory.py`, `src/agents/scoper/agent.py`

---

## Priority 2 — Refactor CLI orchestration in `main.py`

**Problem:** `generate`, `refine`, and `scope` each repeat: CLI overrides → config loading → API validation → repo resolution → agent setup → invoke/stream → exception handling (~45 duplicated lines x3).

**Design:** Three helpers in `main.py`:

1. **`_prepare_config(provider, model) -> Config | None`** — Builds overrides dict, calls `Config.from_env()`, validates API key. Returns `None` (printing error) on validation failure.

2. **`_resolve_repo_with_error_handling(source, debug) -> ContextManager`** — Wraps `resolve_repo()` with shared `CalledProcessError` / `TimeoutExpired` / `ValueError` handling.

3. **`_run_agent(agent, user_message, agent_config, stream, debug) -> int`** — Wraps invoke-vs-stream pattern (TTY detection) and common `except Exception` handler. Returns exit code.

Command handlers become thin ~15-line functions composing these helpers.

**Files changed:**
- Modified: `src/main.py`

---

## Priority 3 — Narrow exception handling

**Problem:** 5 tool functions in `repository_tools.py` use `except Exception as e: return {"error": ...}` with no logging or differentiation.

**Design:**
- Add specific catches for known failures: `yaml.YAMLError`, `OSError`, `ValueError`
- Keep final `except Exception` as safety net, but log full traceback via `logger.exception()` before returning safe message
- In `main.py:_extract_repo_from_context`, replace silent `pass` with `logger.warning()`

**Files changed:**
- Modified: `src/agents/tools/repository_tools.py`, `src/main.py`

---

## Priority 4 — Lazy config fallback in `repository_tools.py`

**Problem:** `_default_config = Config.from_env()` runs at import time. Env changes after import are ignored.

**Design:** Lazy resolution in `_get_config()`:

```python
_default_config: Config | None = None

def _get_config() -> Config:
    cfg = _tool_config.get(None)
    if cfg is not None:
        return cfg
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config
```

**Files changed:**
- Modified: `src/agents/tools/repository_tools.py`

---

## Priority 5 — Unify ignore lists

**Problem:** `Config.DEFAULT_IGNORED_DIRS` (8 entries, list) and `discovery.py:IGNORED_DIRS` (14 entries, set) are maintained independently with different contents.

**Design:**
- Expand `Config.DEFAULT_IGNORED_DIRS` to include all 14 entries
- Convert to `frozenset` for O(1) lookup
- Replace `discovery.py:IGNORED_DIRS` with reference to `Config.DEFAULT_IGNORED_DIRS`

**Files changed:**
- Modified: `src/agents/config.py`, `src/agents/scoper/discovery.py`

---

## Priority 6 — Fix comment mismatch

**Problem:** `AnthropicProvider._rl()` comment says "once every 10 seconds" but `requests_per_second=0.7` = ~1.43 sec/request (7x off).

**Fix:** Correct comment to `# ~0.7 req/sec, approx one request every ~1.4 seconds`.

**Files changed:**
- Modified: `src/agents/llm/provider.py`

---

## Priority 7 — Fix ContextVar typing

**Problem:** `_tool_config: ContextVar[Config]` with `default=None` is a type mismatch.

**Fix:** Change to `ContextVar[Config | None]`.

**Files changed:**
- Modified: `src/agents/tools/repository_tools.py`

---

## Priority 8 — Clarify HITL API semantics

**Problem:** `create_contextualizer_agent_with_hitl()` suggests functional HITL, but tools don't call `interrupt()`. The function is a stub that only validates checkpointer presence.

**Design:**
- Rename to `create_contextualizer_agent_with_checkpointer()`
- Update docstring to clearly state it enables state persistence, not automatic approval gates
- Add `# ROADMAP:` comment referencing future HITL implementation
- Create `docs/roadmap/hitl-approval-gates.md` documenting intended future behavior
- Update all imports/references to old name

**Files changed:**
- Modified: `src/agents/factory.py`, any files importing the old name
- New: `docs/roadmap/hitl-approval-gates.md`

---

## Constraints

- No new features or expanded functionality
- No changes to public CLI interface or output formats
- All existing tests must pass
- One logical change per commit
- Run `pytest` after each priority group
