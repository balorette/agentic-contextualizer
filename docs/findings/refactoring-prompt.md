# Refactoring Agent Prompt — Agentic Contextualizer

You are a senior Python refactoring agent. Your task is to improve the **agentic-contextualizer** codebase based on the consolidated findings from two independent code reviews. The codebase is a LangChain/LangGraph-based tool that scans repositories and generates markdown context files for AI coding agents.

Apply the fixes below in priority order. Keep changes minimal and surgical — do not introduce new features, only improve structure, clarity, and correctness. Run the existing test suite after each logical group of changes to confirm nothing breaks.

---

## Priority 1 — Extract shared chat model builder

**Problem:** `create_contextualizer_agent()` and `create_scoped_agent()` both independently implement model/provider selection, LiteLLM-vs-standard branching, API key resolution, `litellm_kwargs` construction, rate limiter setup, and fallback `init_chat_model` creation. This duplicated logic increases the change surface and risks behavioral skew between the two agent types.

**What to do:**
- Create a shared internal utility, e.g. `src/agents/llm/chat_model_factory.py`, that encapsulates:
  - LiteLLM decision logic
  - API key resolution
  - Rate limiter setup
  - `ChatLiteLLM` kwargs construction
  - Fallback `init_chat_model` creation
- Expose a single function like `build_chat_model(config, model_name, base_url, api_key, use_litellm, debug) -> BaseChatModel`.
- Call this from both agent factories, removing the duplicated blocks.

---

## Priority 2 — Refactor CLI orchestration in `main.py`

**Problem:** The `generate`, `refine`, and `scope` command handlers each repeat similar steps: CLI overrides, config loading, API validation, repo resolution, agent invocation (stream vs. invoke), and exception handling. This creates parallel logic branches that can drift.

**What to do:**
- Introduce reusable orchestration helpers:
  - `_build_cli_overrides(provider, model)` — merges CLI flags into config.
  - `_handle_repo_resolution(source, debug)` — resolves the repo path with standard error handling.
  - `_run_agent_execution(agent, user_message, agent_config, stream, debug)` — wraps the invoke-vs-stream pattern and common exception handling.
- Refactor command functions to be thin and declarative, delegating to these helpers.

---

## Priority 3 — Narrow exception handling and add structured error context

**Problem:** Broad `except Exception` blocks in parser/tool boundaries suppress stack information and return generic strings. Examples include frontmatter extraction silently swallowing all exceptions and tool wrappers returning opaque error strings.

**What to do:**
- Replace broad `except Exception` with specific exception types where the failure modes are known (e.g. `yaml.YAMLError`, `IOError`, `ValueError`).
- For unexpected exceptions, log the full traceback via the logger before returning a safe tool-facing message.
- Never silently `pass` in parsing helpers — at minimum log a warning.

---

## Priority 4 — Eliminate import-time fallback config in `repository_tools.py`

**Problem:** `repository_tools.py` creates `_default_config = Config.from_env()` at import time. If env/config changes after module import (e.g. in tests or subprocesses), the fallback becomes stale.

**What to do:**
- Resolve the fallback config lazily inside `_get_config()` instead of at module import time.
- Alternatively, require explicit config injection before tool execution and fail fast if missing.

---

## Priority 5 — Unify ignore-list / scan-policy configuration

**Problem:** Ignored directory defaults exist in multiple places (`Config.DEFAULT_IGNORED_DIRS` and scoper discovery constants) with different sets, leading to inconsistent scan behavior.

**What to do:**
- Centralize the ignore policy in one authoritative location (e.g. on `Config`).
- Have all scan/discovery paths reference that single source instead of maintaining independent lists.

---

## Priority 6 — Fix comment/documentation mismatches

**Problem:** In `AnthropicProvider._rl()`, the comment says "once every 10 seconds" while `requests_per_second=0.7` is ~1.4 seconds/request. This kind of documentation drift erodes trust.

**What to do:**
- Correct the inline comment to match the effective rate.
- Consider reading rate limits from `Config` for consistency.

---

## Priority 7 — Fix `ContextVar` typing

**Problem:** `_tool_config` is declared as `ContextVar[Config]` but defaulted to `None` and retrieved with a `None` fallback.

**What to do:**
- Change the type annotation to `ContextVar[Config | None]` to match actual usage.

---

## Priority 8 — Clarify HITL API semantics

**Problem:** `create_contextualizer_agent_with_hitl()` suggests enabled human-in-the-loop behavior, but tools don't call `interrupt()` by default — the behavior is conditional and mostly advisory.

**What to do:**
- Either implement wrapper-based approval integration, or rename/deprecate to indicate "HITL prerequisites only" to reduce surprise.

---

## Constraints

- **Do not** add new features or expand functionality.
- **Do not** change the public CLI interface or output formats.
- **Preserve** all existing tests — they must continue to pass.
- **Keep** changes focused and reviewable; one logical change per commit.
- Run `pytest` after each priority group to catch regressions early.
