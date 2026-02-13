# Code Review Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply 8 surgical refactoring improvements from consolidated code reviews — eliminate duplication, narrow error handling, fix typing and comments, unify config, and clarify the HITL API.

**Architecture:** Extract shared helpers from duplicated agent factory and CLI code; centralize config constants; fix typing, comments, and exception handling at module boundaries. No new features.

**Tech Stack:** Python 3.x, LangChain/LangGraph, pytest, uv

---

### Task 1: Extract shared chat model builder (Priority 1)

**Files:**
- Create: `src/agents/llm/chat_model_factory.py`
- Modify: `src/agents/llm/__init__.py`
- Modify: `src/agents/factory.py:111-194` (LLM setup block)
- Modify: `src/agents/scoper/agent.py:139-188` (LLM setup block)
- Test: `tests/agents/llm/test_chat_model_factory.py`

**Step 1: Write the failing tests**

Create `tests/agents/llm/test_chat_model_factory.py`:

```python
"""Tests for shared chat model builder."""

import pytest
from unittest.mock import patch, MagicMock


class TestBuildChatModel:
    """Tests for build_chat_model factory function."""

    @patch("src.agents.llm.chat_model_factory.init_chat_model")
    def test_standard_model_uses_init_chat_model(self, mock_init):
        """Standard (non-LiteLLM) models should use init_chat_model."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_init.return_value = MagicMock()
        config = Config(anthropic_api_key="test-key")

        result = build_chat_model(
            config=config,
            model_name="anthropic:claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        mock_init.assert_called_once()
        assert result is not None

    @patch("src.agents.llm.chat_model_factory.ChatLiteLLM")
    def test_litellm_flag_uses_chat_litellm(self, mock_litellm_cls):
        """use_litellm=True should use ChatLiteLLM."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(anthropic_api_key="test-key")

        result = build_chat_model(
            config=config,
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-key",
            use_litellm=True,
        )

        mock_litellm_cls.assert_called_once()
        assert result is not None

    @patch("src.agents.llm.chat_model_factory.ChatLiteLLM")
    def test_base_url_triggers_litellm(self, mock_litellm_cls):
        """Providing base_url should auto-trigger LiteLLM path."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(anthropic_api_key="test-key")

        build_chat_model(
            config=config,
            model_name="gpt-4o",
            base_url="https://gateway.example.com",
            api_key="test-key",
        )

        mock_litellm_cls.assert_called_once()

    @patch("src.agents.llm.chat_model_factory.ChatLiteLLM")
    def test_litellm_provider_config_triggers_litellm(self, mock_litellm_cls):
        """config.llm_provider == 'litellm' should trigger LiteLLM path."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(llm_provider="litellm", anthropic_api_key="test-key")

        build_chat_model(
            config=config,
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        mock_litellm_cls.assert_called_once()

    @patch("src.agents.llm.chat_model_factory.ChatLiteLLM")
    def test_litellm_kwargs_include_config_values(self, mock_litellm_cls):
        """LiteLLM kwargs should include config values for retries, timeout, max_tokens."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(
            anthropic_api_key="test-key",
            max_retries=5,
            timeout=120,
            max_output_tokens=8192,
        )

        build_chat_model(
            config=config,
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-key",
            use_litellm=True,
        )

        call_kwargs = mock_litellm_cls.call_args[1]
        assert call_kwargs["max_retries"] == 5
        assert call_kwargs["request_timeout"] == 120
        assert call_kwargs["max_tokens"] == 8192

    @patch("src.agents.llm.chat_model_factory.init_chat_model")
    def test_init_chat_model_404_raises_runtime_error(self, mock_init):
        """init_chat_model 404 error should raise RuntimeError with LiteLLM suggestion."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_init.side_effect = Exception("404 Not Found")
        config = Config(anthropic_api_key="test-key")

        with pytest.raises(RuntimeError, match="litellm"):
            build_chat_model(
                config=config,
                model_name="anthropic:claude-3-5-sonnet-20241022",
                api_key="test-key",
            )


class TestBuildTokenMiddleware:
    """Tests for shared middleware builder."""

    def test_build_token_middleware_returns_middleware(self):
        """Should return a TokenBudgetMiddleware instance."""
        from src.agents.llm.chat_model_factory import build_token_middleware
        from src.agents.config import Config
        from src.agents.middleware.token_budget import TokenBudgetMiddleware

        config = Config()
        mw = build_token_middleware(config, "claude-3-5-sonnet-20241022")

        assert isinstance(mw, TokenBudgetMiddleware)

    def test_build_token_middleware_uses_config_tpm(self):
        """Should use config.max_tpm for throttle setup."""
        from src.agents.llm.chat_model_factory import build_token_middleware
        from src.agents.config import Config

        config = Config(max_tpm=50000, tpm_safety_factor=0.9)
        mw = build_token_middleware(config, "test-model")

        # Middleware should be created without error
        assert mw is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/llm/test_chat_model_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agents.llm.chat_model_factory'`

**Step 3: Implement `chat_model_factory.py`**

Create `src/agents/llm/chat_model_factory.py`:

```python
"""Shared chat model and middleware builder for agent factories.

Consolidates the duplicated LLM setup logic from factory.py and scoper/agent.py
into a single authoritative location.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ..config import Config
from .provider import _resolve_api_key_for_model

if TYPE_CHECKING:
    from ..middleware.token_budget import TokenBudgetMiddleware


def _format_model_name_for_langchain(model_name: str) -> str:
    """Format model name for LangChain's init_chat_model.

    LangChain expects format: "provider:model"
    """
    if ":" in model_name:
        return model_name

    if model_name.startswith("gpt-") or model_name.startswith("o1"):
        return f"openai:{model_name}"
    elif model_name.startswith("claude"):
        return f"anthropic:{model_name}"
    elif model_name.startswith("gemini") or model_name.startswith("vertex"):
        return f"google-genai:{model_name}"
    else:
        return f"anthropic:{model_name}"


def build_chat_model(
    config: Config,
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    use_litellm: bool = False,
    debug: bool = False,
) -> BaseChatModel:
    """Build a chat model instance from config and options.

    Encapsulates LiteLLM-vs-standard branching, API key resolution,
    kwargs construction, and error handling.

    Args:
        config: Application configuration
        model_name: LLM model identifier (with or without provider prefix)
        base_url: Optional custom API endpoint URL
        api_key: Optional API key (resolved from config if not provided)
        use_litellm: Force use of ChatLiteLLM
        debug: Enable verbose logging

    Returns:
        Configured BaseChatModel instance
    """
    should_use_litellm = (
        use_litellm
        or base_url is not None
        or config.llm_provider == "litellm"
    )

    if not api_key:
        api_key = _resolve_api_key_for_model(model_name, config)

    if debug:
        print(f"[DEBUG] Creating LangChain model:")
        print(f"  - Original model_name: {model_name}")
        print(f"  - Using LiteLLM: {should_use_litellm}")
        print(f"  - base_url: {base_url or 'None'}")
        print(f"  - api_key: {'***' + api_key[-4:] if api_key else 'None'}")

    if should_use_litellm:
        return _build_litellm_model(config, model_name, base_url, api_key, debug)
    else:
        return _build_standard_model(model_name, base_url, api_key, debug)


def _build_litellm_model(
    config: Config,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    debug: bool,
) -> BaseChatModel:
    """Build a ChatLiteLLM model instance."""
    from langchain_litellm import ChatLiteLLM

    if debug:
        import os
        os.environ["LITELLM_LOG"] = "DEBUG"

    litellm_kwargs: dict = {
        "model": model_name,
        "temperature": 0.0,
    }

    if api_key:
        litellm_kwargs["api_key"] = api_key
    if base_url:
        litellm_kwargs["api_base"] = base_url
    if config.max_retries:
        litellm_kwargs["max_retries"] = config.max_retries
    if config.timeout:
        litellm_kwargs["request_timeout"] = config.timeout
    if config.max_output_tokens:
        litellm_kwargs["max_tokens"] = config.max_output_tokens

    if debug:
        print(f"  - ChatLiteLLM kwargs:")
        for k, v in litellm_kwargs.items():
            if k == "api_key":
                print(f"      {k}: ***{v[-4:] if v else 'None'}")
            else:
                print(f"      {k}: {v}")

    model = ChatLiteLLM(**litellm_kwargs)

    if debug:
        print(f"  - Created ChatLiteLLM successfully")

    return model


def _build_standard_model(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    debug: bool,
) -> BaseChatModel:
    """Build a standard LangChain chat model via init_chat_model."""
    formatted_model_name = _format_model_name_for_langchain(model_name)

    model_kwargs: dict = {}
    if base_url:
        model_kwargs["base_url"] = base_url
    if api_key:
        model_kwargs["api_key"] = api_key

    if debug:
        print(f"  - Formatted model_name: {formatted_model_name}")

    try:
        return init_chat_model(formatted_model_name, **model_kwargs)
    except Exception as e:
        if "404" in str(e) or "401" in str(e):
            raise RuntimeError(
                f"Failed to initialize model '{formatted_model_name}'.\n"
                f"For custom LiteLLM gateways, set LLM_PROVIDER=litellm in .env\n"
                f"Original error: {e}"
            ) from e
        raise


def build_token_middleware(
    config: Config,
    model_name: str,
) -> "TokenBudgetMiddleware":
    """Build a TokenBudgetMiddleware with TPM throttle from config.

    Args:
        config: Application configuration
        model_name: Model name for token estimation

    Returns:
        Configured TokenBudgetMiddleware
    """
    from .rate_limiting import TPMThrottle
    from .token_estimator import LiteLLMTokenEstimator
    from ..middleware.token_budget import TokenBudgetMiddleware

    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    estimator = LiteLLMTokenEstimator()

    return TokenBudgetMiddleware(
        max_input_tokens=config.max_input_tokens,
        max_tool_output_chars=config.max_tool_output_chars,
        throttle=throttle,
        estimator=estimator,
        model_name=model_name,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/llm/test_chat_model_factory.py -v`
Expected: All PASS

**Step 5: Update `__init__.py` exports**

In `src/agents/llm/__init__.py`, add:
```python
from .chat_model_factory import build_chat_model, build_token_middleware
```

And add `"build_chat_model"` and `"build_token_middleware"` to `__all__`.

**Step 6: Refactor `factory.py` to use `build_chat_model`**

Replace lines 111-222 of `src/agents/factory.py` with:

```python
    from .llm.chat_model_factory import build_chat_model, build_token_middleware

    config = Config.from_env()
    model = build_chat_model(
        config=config,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        use_litellm=use_litellm,
        debug=debug,
    )
```

For middleware (lines 206-222):
```python
    budget_mw = build_token_middleware(config, model_name)
    all_middleware = [budget_mw] + (middleware or [])
```

Remove the now-unused `_format_model_name_for_langchain` from `factory.py` (it's now in `chat_model_factory.py`).

**Step 7: Refactor `scoper/agent.py` to use `build_chat_model`**

Replace lines 139-188 of `src/agents/scoper/agent.py` with:

```python
    from ..llm.chat_model_factory import build_chat_model, build_token_middleware

    config = Config.from_env()
    model = build_chat_model(
        config=config,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        use_litellm=use_litellm,
        debug=debug,
    )
```

For middleware (lines 262-277):
```python
    budget_mw = build_token_middleware(config, model_name)
```

Remove unused imports (`init_chat_model`, `_format_model_name_for_langchain`).

**Step 8: Fix any imports referencing `_format_model_name_for_langchain` from factory**

Update `src/agents/scoper/agent.py` line 20: remove the import of `_format_model_name_for_langchain` from `..factory`.

**Step 9: Run full test suite**

Run: `uv run pytest -v`
Expected: All 315+ tests PASS

**Step 10: Commit**

```bash
git add src/agents/llm/chat_model_factory.py src/agents/llm/__init__.py \
        src/agents/factory.py src/agents/scoper/agent.py \
        tests/agents/llm/test_chat_model_factory.py
git commit -m "refactor: extract shared chat model builder into chat_model_factory.py

Consolidates duplicated LLM setup logic from factory.py and
scoper/agent.py into build_chat_model() and build_token_middleware().
Eliminates ~60 lines of duplication per factory."
```

---

### Task 2: Refactor CLI orchestration in `main.py` (Priority 2)

**Files:**
- Modify: `src/agents/main.py`
- Test: `tests/test_cli.py` (existing tests must pass)

**Step 1: Write failing tests for new helpers**

Add to `tests/test_cli.py`:

```python
class TestPrepareConfig:
    """Test _prepare_config helper."""

    @patch("src.agents.main.Config.from_env")
    def test_returns_config_on_valid_key(self, mock_from_env):
        from src.agents.main import _prepare_config
        from unittest.mock import Mock

        config = Mock()
        config.anthropic_api_key = "test-key"
        config.api_key = "test-key"
        config.model_name = "claude-3-5-sonnet-20241022"
        config.llm_provider = "anthropic"
        config.api_base_url = None
        mock_from_env.return_value = config

        result = _prepare_config(None, None)
        assert result is not None

    @patch("src.agents.main.Config.from_env")
    def test_returns_none_on_missing_key(self, mock_from_env):
        from src.agents.main import _prepare_config
        from unittest.mock import Mock

        config = Mock()
        config.anthropic_api_key = None
        config.api_key = None
        config.openai_api_key = None
        config.google_api_key = None
        config.model_name = "claude-3-5-sonnet-20241022"
        config.llm_provider = "anthropic"
        config.api_base_url = None
        mock_from_env.return_value = config

        result = _prepare_config(None, None)
        assert result is None

    @patch("src.agents.main.Config.from_env")
    def test_passes_provider_override(self, mock_from_env):
        from src.agents.main import _prepare_config
        from unittest.mock import Mock

        config = Mock()
        config.anthropic_api_key = "key"
        config.api_key = "key"
        config.model_name = "gpt-4o"
        config.llm_provider = "litellm"
        config.api_base_url = None
        config.openai_api_key = "key"
        mock_from_env.return_value = config

        _prepare_config("litellm", "gpt-4o")
        call_kwargs = mock_from_env.call_args[1]
        assert call_kwargs["cli_overrides"]["llm_provider"] == "litellm"
        assert call_kwargs["cli_overrides"]["model_name"] == "gpt-4o"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::TestPrepareConfig -v`
Expected: FAIL — `ImportError: cannot import name '_prepare_config'`

**Step 3: Implement helpers in `main.py`**

Add after `_validate_api_key()` in `main.py`:

```python
def _prepare_config(provider: str | None, model: str | None) -> Config | None:
    """Build config with CLI overrides and validate API key.

    Returns Config on success, None on validation failure (error already printed).
    """
    cli_overrides = {}
    if provider:
        cli_overrides["llm_provider"] = provider
    if model:
        cli_overrides["model_name"] = model

    config = Config.from_env(cli_overrides=cli_overrides)

    is_valid, error_msg = _validate_api_key(config)
    if not is_valid:
        click.echo(error_msg, err=True)
        return None
    return config


def _run_agent(agent, user_message: str, agent_config: dict, stream: bool, debug: bool) -> int:
    """Execute an agent with invoke or stream, returning exit code.

    Handles TTY detection for streaming and common exception handling.
    """
    try:
        if stream:
            from .streaming import stream_agent_execution, simple_stream_agent_execution
            import sys

            if sys.stdout.isatty():
                stream_agent_execution(
                    agent,
                    messages=[{"role": "user", "content": user_message}],
                    config=agent_config,
                    verbose=debug,
                )
            else:
                simple_stream_agent_execution(
                    agent,
                    messages=[{"role": "user", "content": user_message}],
                    config=agent_config,
                )
        else:
            click.echo("\n🔄 Agent executing...")
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=agent_config,
            )
            final_message = result.get("messages", [])[-1]
            output_content = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )
            click.echo("\n📋 Agent Response:")
            click.echo(output_content)
            click.echo("\n✅ Agent execution complete")

        return 0

    except Exception as e:
        click.echo(f"\n❌ Agent execution failed: {e}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        return 1
```

**Step 4: Refactor command handlers to use helpers**

Refactor `generate`, `refine`, `scope` commands to use `_prepare_config()` and `_run_agent()`. Each command handler should shrink significantly. The `_generate_agent_mode`, `_refine_agent_mode`, and `_scope_agent_mode` functions should all use `_run_agent()` for the invoke/stream block.

For example, `_generate_agent_mode` becomes:

```python
def _generate_agent_mode(repo: Path, summary: str, config: Config, debug: bool, stream: bool) -> int:
    from .factory import create_contextualizer_agent
    from .memory import create_checkpointer, create_agent_config
    from .observability import configure_tracing, is_tracing_enabled
    from .tools.repository_tools import set_tool_config
    from .llm.provider import _resolve_api_key_for_model

    click.echo(f"🤖 Agent mode: Analyzing repository: {repo}")
    if stream:
        click.echo("   Streaming: Enabled")

    set_tool_config(config)
    configure_tracing()
    checkpointer = create_checkpointer()
    api_key = _resolve_api_key_for_model(config.model_name, config)

    agent = create_contextualizer_agent(
        model_name=config.model_name,
        checkpointer=checkpointer,
        debug=debug,
        base_url=config.api_base_url,
        api_key=api_key,
    )

    agent_config = create_agent_config(str(repo))
    user_message = f"Generate context for {repo}. User description: {summary}"

    click.echo(f"   Thread ID: {agent_config['configurable']['thread_id']}")
    if is_tracing_enabled():
        click.echo(f"   Tracing: Enabled")

    return _run_agent(agent, user_message, agent_config, stream, debug)
```

Apply the same pattern to `_refine_agent_mode` and `_scope_agent_mode`.

Also refactor the top-level `generate()` and `scope()` to use `_prepare_config()`:

```python
def generate(source, summary, output, mode, provider, model, debug, stream):
    config = _prepare_config(provider, model)
    if config is None:
        return 1

    try:
        with resolve_repo(source) as repo:
            if mode == "agent":
                return _generate_agent_mode(repo, summary, config, debug, stream)
            else:
                return _generate_pipeline_mode(repo, summary, config)
    except subprocess.CalledProcessError as e:
        # ... same error handling ...
```

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/agents/main.py tests/test_cli.py
git commit -m "refactor: extract CLI orchestration helpers in main.py

Add _prepare_config() and _run_agent() to eliminate duplicated
config/validation/invocation patterns across generate, refine, and scope."
```

---

### Task 3: Narrow exception handling (Priority 3)

**Files:**
- Modify: `src/agents/tools/repository_tools.py:130-131, 154-155, 199-200, 251-252, 277-278`
- Modify: `src/agents/main.py:43-45`
- Test: `tests/agents/tools/test_repository_tools.py` (existing tests must pass)

**Step 1: Add specific exception catches and logging**

In `src/agents/tools/repository_tools.py`, add a logger at module level:

```python
import logging

logger = logging.getLogger(__name__)
```

Then update each tool's exception handler. For example, `scan_structure`:

```python
    except (OSError, ValueError) as e:
        return {"error": f"Failed to scan repository: {str(e)}"}
    except Exception as e:
        logger.exception("Unexpected error in scan_structure")
        return {"error": f"Failed to scan repository: {str(e)}"}
```

Apply similar narrowing to all 5 tool functions:
- `scan_structure`: catch `OSError`, `ValueError`
- `extract_metadata`: catch `OSError`, `ValueError`
- `analyze_code`: catch `OSError`, `ValueError`, `RuntimeError`
- `generate_context`: catch `OSError`, `ValueError`, `RuntimeError`
- `refine_context`: catch `OSError`, `ValueError`, `RuntimeError`

In `src/agents/main.py:_extract_repo_from_context`, replace the silent `pass`:

```python
    except yaml.YAMLError:
        logger.warning("Failed to parse frontmatter from %s", context_path)
    except OSError as e:
        logger.warning("Failed to read context file %s: %s", context_path, e)
    except Exception:
        logger.warning("Unexpected error extracting repo from %s", context_path, exc_info=True)
```

Add `import logging` and `logger = logging.getLogger(__name__)` at top of `main.py`.

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (existing tests check for `"error" in result` which still holds)

**Step 3: Commit**

```bash
git add src/agents/tools/repository_tools.py src/agents/main.py
git commit -m "fix: narrow exception handling and add structured error logging

Replace broad except Exception with specific catches where failure
modes are known. Log unexpected exceptions via logger.exception()
before returning safe tool-facing messages."
```

---

### Task 4: Lazy config fallback (Priority 4)

**Files:**
- Modify: `src/agents/tools/repository_tools.py:18-47`
- Test: `tests/agents/tools/test_repository_tools.py` (add test)

**Step 1: Write failing test**

Add to `tests/agents/tools/test_repository_tools.py`:

```python
class TestGetConfig:
    """Tests for _get_config lazy fallback."""

    def test_get_config_returns_context_var_if_set(self):
        """_get_config should prefer the ContextVar value."""
        from src.agents.tools.repository_tools import _get_config, set_tool_config
        from src.agents.config import Config

        config = Config(model_name="test-model")
        set_tool_config(config)

        result = _get_config()
        assert result.model_name == "test-model"

    def test_get_config_lazy_creates_default(self):
        """_get_config should lazily create fallback on first call when ContextVar unset."""
        from src.agents.tools import repository_tools
        from src.agents.tools.repository_tools import _get_config

        # Reset the module-level state
        repository_tools._default_config = None
        repository_tools._tool_config = repository_tools.ContextVar(
            'tool_config', default=None
        )

        result = _get_config()
        assert result is not None
        # Should have been lazily created
        assert repository_tools._default_config is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/tools/test_repository_tools.py::TestGetConfig -v`
Expected: FAIL (module-level `_default_config` is eagerly created)

**Step 3: Implement lazy config**

In `src/agents/tools/repository_tools.py`, change lines 16-47 to:

```python
_tool_config: ContextVar[Config | None] = ContextVar('tool_config', default=None)

# Lazily initialized fallback (was import-time; now deferred)
_default_config: Config | None = None


def set_tool_config(config: Config) -> None:
    """Set the config to be used by tools in this context."""
    _tool_config.set(config)


def _get_config() -> Config:
    """Get current config from context or lazily create fallback."""
    config = _tool_config.get(None)
    if config is not None:
        return config
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config
```

Note: This also fixes Priority 7 (ContextVar typing) since we're changing `ContextVar[Config]` to `ContextVar[Config | None]`.

**Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agents/tools/repository_tools.py tests/agents/tools/test_repository_tools.py
git commit -m "fix: lazy config fallback and correct ContextVar typing

Replace import-time Config.from_env() with lazy initialization in
_get_config(). Fix ContextVar[Config] -> ContextVar[Config | None]
to match actual usage."
```

---

### Task 5: Unify ignore lists (Priority 5)

**Files:**
- Modify: `src/agents/config.py:13-22, 57, 84`
- Modify: `src/agents/scoper/discovery.py:59-63, 94`
- Test: `tests/scoper/test_discovery.py`, `tests/test_config.py` (existing)

**Step 1: Write failing test**

Add to `tests/test_config.py`:

```python
def test_default_ignored_dirs_is_frozenset():
    """DEFAULT_IGNORED_DIRS should be a frozenset for O(1) lookup."""
    from src.agents.config import DEFAULT_IGNORED_DIRS
    assert isinstance(DEFAULT_IGNORED_DIRS, frozenset)

def test_default_ignored_dirs_includes_cache_dirs():
    """DEFAULT_IGNORED_DIRS should include mypy_cache, ruff_cache, etc."""
    from src.agents.config import DEFAULT_IGNORED_DIRS
    assert ".mypy_cache" in DEFAULT_IGNORED_DIRS
    assert ".ruff_cache" in DEFAULT_IGNORED_DIRS
    assert ".tox" in DEFAULT_IGNORED_DIRS
    assert ".nox" in DEFAULT_IGNORED_DIRS
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py::test_default_ignored_dirs_is_frozenset -v`
Expected: FAIL — `assert isinstance(DEFAULT_IGNORED_DIRS, frozenset)` fails (it's a list)

**Step 3: Expand and convert `DEFAULT_IGNORED_DIRS`**

In `src/agents/config.py`, change lines 13-22 to:

```python
DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "egg-info", ".egg-info", ".tox", ".nox",
})
```

Update `from_env()` line 84: change `ignored_dirs = DEFAULT_IGNORED_DIRS.copy()` to `ignored_dirs = list(DEFAULT_IGNORED_DIRS)` (since frozenset has no `.copy()` that returns a list).

Update `Config.ignored_dirs` field default: `Field(default_factory=lambda: list(DEFAULT_IGNORED_DIRS))`.

**Step 4: Update `discovery.py` to reference Config**

In `src/agents/scoper/discovery.py`, replace lines 58-63:

```python
from ..config import DEFAULT_IGNORED_DIRS
```

Remove the module-level `IGNORED_DIRS` constant. Update line 94 reference:

```python
dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORED_DIRS and not d.endswith(".egg-info")]
```

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/agents/config.py src/agents/scoper/discovery.py \
        tests/test_config.py
git commit -m "refactor: unify ignore-list into single DEFAULT_IGNORED_DIRS frozenset

Expand Config.DEFAULT_IGNORED_DIRS to 14 entries (adding mypy_cache,
ruff_cache, egg-info, tox, nox). Convert to frozenset for O(1) lookup.
Remove duplicate IGNORED_DIRS from discovery.py."
```

---

### Task 6: Fix comment mismatch (Priority 6)

**Files:**
- Modify: `src/agents/llm/provider.py:44`

**Step 1: Fix the comment**

Change line 44 from:
```python
            requests_per_second=0.7,  # <-- Super slow! We can only make a request once every 10 seconds!!
```
to:
```python
            requests_per_second=0.7,  # ~0.7 req/sec, approx one request every ~1.4 seconds
```

**Step 2: Run tests**

Run: `uv run pytest tests/agents/llm/ tests/test_llm.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/agents/llm/provider.py
git commit -m "fix: correct rate limiter comment to match actual rate

The comment said 'once every 10 seconds' but 0.7 req/sec is ~1.4
seconds per request."
```

---

### Task 7: ContextVar typing (Priority 7)

**Already handled in Task 4.** The `ContextVar[Config]` → `ContextVar[Config | None]` fix is included in the lazy config refactoring. No separate task needed.

---

### Task 8: Clarify HITL API (Priority 8)

**Files:**
- Modify: `src/agents/factory.py:305-394`
- Modify: `docs/API_REFERENCE.md` (update import references)
- Create: `docs/roadmap/hitl-approval-gates.md`

**Step 1: Write test for renamed function**

Add to `tests/agents/test_agent_integration.py`:

```python
class TestAgentWithCheckpointer:
    """Test create_contextualizer_agent_with_checkpointer."""

    def test_requires_checkpointer(self):
        from src.agents.factory import create_contextualizer_agent_with_checkpointer

        with pytest.raises(ValueError, match="checkpointer is required"):
            create_contextualizer_agent_with_checkpointer(checkpointer=None)

    def test_creates_agent_with_checkpointer(self):
        from src.agents.factory import create_contextualizer_agent_with_checkpointer
        from src.agents.memory import create_checkpointer

        checkpointer = create_checkpointer()
        agent = create_contextualizer_agent_with_checkpointer(
            checkpointer=checkpointer
        )
        assert agent is not None
        assert hasattr(agent, "invoke")
```

**Step 2: Run test to verify failure**

Run: `uv run pytest tests/agents/test_agent_integration.py::TestAgentWithCheckpointer -v`
Expected: FAIL — `ImportError: cannot import name 'create_contextualizer_agent_with_checkpointer'`

**Step 3: Rename function in `factory.py`**

Rename `create_contextualizer_agent_with_hitl` → `create_contextualizer_agent_with_checkpointer`.

Update the docstring:

```python
def create_contextualizer_agent_with_checkpointer(
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    debug: bool = False,
    require_approval_for: Optional[list[str]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
):
    """Create agent with state persistence via checkpointer.

    This creates an agent with checkpointing enabled, which is a
    prerequisite for future human-in-the-loop approval gates.

    Note: This does NOT currently implement automatic approval gates.
    Tools do not call interrupt() by default. See docs/roadmap/hitl-approval-gates.md
    for the planned HITL implementation.

    # ROADMAP: Wire create_approval_tool() wrappers for tools listed in
    # require_approval_for. See docs/roadmap/hitl-approval-gates.md

    Args:
        model_name: LLM model identifier
        checkpointer: Checkpointer for state persistence (REQUIRED)
        debug: Enable verbose logging
        require_approval_for: Reserved for future HITL implementation
        ...
    """
```

**Step 4: Create roadmap doc**

Create `docs/roadmap/hitl-approval-gates.md`:

```markdown
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

- `src/agents/middleware/human_in_the_loop.py` — existing wrapper infrastructure
- `src/agents/factory.py` — factory function with `require_approval_for` parameter
- LangGraph interrupt docs: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
```

**Step 5: Update `docs/API_REFERENCE.md`**

Search-and-replace `create_contextualizer_agent_with_hitl` → `create_contextualizer_agent_with_checkpointer` in the API reference.

**Step 6: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/agents/factory.py docs/roadmap/hitl-approval-gates.md \
        docs/API_REFERENCE.md tests/agents/test_agent_integration.py
git commit -m "refactor: rename HITL factory to clarify it only enables checkpointing

Rename create_contextualizer_agent_with_hitl to
create_contextualizer_agent_with_checkpointer. Add roadmap doc
for future HITL approval gate implementation."
```

---

### Task 9: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 2: Quick smoke test**

Run: `uv run python -c "from src.agents.factory import create_contextualizer_agent; print('factory OK')"`
Run: `uv run python -c "from src.agents.scoper.agent import create_scoped_agent; print('scoper OK')"`
Run: `uv run python -c "from src.agents.llm.chat_model_factory import build_chat_model; print('chat_model_factory OK')"`

Expected: All print OK with no import errors.
