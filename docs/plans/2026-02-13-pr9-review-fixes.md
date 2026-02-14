# PR #9 Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all 10 Copilot review comments on PR #9 (feat/litellm-enablement), fixing bugs, test/doc mismatches, and security concerns.

**Architecture:** These are independent fixes across config, provider, tools, tests, and example code. Each task is self-contained. No new abstractions needed — just aligning existing code with documented behavior and fixing inconsistencies.

**Tech Stack:** Python, pytest, LiteLLM, LangChain, Pydantic

---

### Task 1: Fix `_flatten_tree` repo-root prefix in paths

**Context:** `_flatten_tree()` in `repository_tools.py` starts from the root node, so all paths include the repo name as prefix (e.g., `myrepo/src/main.py`). Downstream consumers expect repo-relative paths like `src/main.py`.

**Files:**
- Modify: `src/agents/tools/repository_tools.py:81-91` (fix `_flatten_tree`)
- Modify: `src/agents/tools/repository_tools.py:108-109` (fix call site in `scan_structure`)
- Test: `tests/agents/tools/test_repository_tools.py`

**Step 1: Write a failing test for repo-relative paths**

Add a test in `tests/agents/tools/test_repository_tools.py` that verifies `_flatten_tree` returns paths without the root directory prefix:

```python
def test_flatten_tree_returns_relative_paths():
    """_flatten_tree should return repo-relative paths without root dir name."""
    from agents.tools.repository_tools import _flatten_tree

    tree = {
        "name": "my-repo",
        "type": "directory",
        "children": [
            {"name": "README.md", "type": "file"},
            {
                "name": "src",
                "type": "directory",
                "children": [
                    {"name": "main.py", "type": "file"},
                ],
            },
        ],
    }
    result = _flatten_tree(tree)
    assert "README.md" in result
    assert "src/main.py" in result
    # Must NOT have repo name prefix
    assert not any(p.startswith("my-repo/") for p in result)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/tools/test_repository_tools.py::test_flatten_tree_returns_relative_paths -v`
Expected: FAIL — paths currently include `my-repo/` prefix

**Step 3: Fix `_flatten_tree` to skip root node**

In `src/agents/tools/repository_tools.py`, change `_flatten_tree` to start from children of root:

```python
def _flatten_tree(node: dict, prefix: str = "") -> list[str]:
    """Flatten nested tree dict into a compact list of repo-relative paths."""
    paths = []
    name = node.get("name", "")
    if node.get("type") == "file":
        path = f"{prefix}/{name}" if prefix else name
        paths.append(path)
    elif node.get("type") == "directory":
        # For the root node (no prefix), don't include the root dir name in paths
        child_prefix = f"{prefix}/{name}" if prefix else ""
        for child in node.get("children", []):
            paths.extend(_flatten_tree(child, child_prefix))
    return paths
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/tools/test_repository_tools.py::test_flatten_tree_returns_relative_paths -v`
Expected: PASS

**Step 5: Run full repository_tools test suite**

Run: `pytest tests/agents/tools/test_repository_tools.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/agents/tools/repository_tools.py tests/agents/tools/test_repository_tools.py
git commit -m "fix: _flatten_tree returns repo-relative paths without root prefix"
```

---

### Task 2: Fix `list_key_files` to accept flat file list

**Context:** `scan_structure` now returns a flat `file_list` (list of strings), but `list_key_files` expects a nested tree dict with `type`/`name`/`children`. The agent has both tools so the LLM will try to pipe one into the other and fail.

**Files:**
- Modify: `src/agents/tools/exploration_tools.py:10-55`
- Test: `tests/agents/tools/` (existing or new test)

**Step 1: Write a failing test for flat list input**

```python
def test_list_key_files_accepts_flat_list():
    """list_key_files should work with a flat file list from scan_structure."""
    from agents.tools.exploration_tools import list_key_files

    flat_list = [
        "README.md",
        "pyproject.toml",
        "src/main.py",
        "src/__init__.py",
        "docs/guide.md",
        "Dockerfile",
    ]
    # list_key_files is a @tool, invoke it
    result = list_key_files.invoke({"file_list": flat_list})
    assert "README.md" in result["docs"]
    assert "pyproject.toml" in result["configs"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/tools/test_exploration_tools.py::test_list_key_files_accepts_flat_list -v`
Expected: FAIL — current signature expects `file_tree: dict`

**Step 3: Update `list_key_files` to accept a flat file list**

In `src/agents/tools/exploration_tools.py`, change the tool to accept `file_list: list[str]` instead of `file_tree: dict`:

```python
@tool
def list_key_files(file_list: list[str]) -> dict[str, list[str]]:
    """Categorize key files (configs, entry_points, docs) from a file list.

    Args:
        file_list: List of repo-relative file paths from scan_structure tool

    Returns:
        Dictionary with configs, entry_points, docs, and all_key_files lists.
    """
    found_files = {
        "configs": [],
        "entry_points": [],
        "docs": [],
    }

    for file_path in file_list:
        file_name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path

        for category, patterns in KEY_FILE_PATTERNS.items():
            for pattern in patterns:
                if file_name == pattern or file_path.endswith(pattern):
                    found_files[category].append(file_path)
                    break

    all_key_files = []
    for category_files in found_files.values():
        all_key_files.extend(category_files)

    return {
        **found_files,
        "all_key_files": sorted(set(all_key_files)),
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/agents/tools/test_exploration_tools.py::test_list_key_files_accepts_flat_list -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: All pass (update any existing tests that pass a tree dict)

**Step 6: Commit**

```bash
git add src/agents/tools/exploration_tools.py tests/agents/tools/
git commit -m "fix: list_key_files accepts flat file_list from scan_structure"
```

---

### Task 3: Fix double `Config.from_env()` in `create_scoped_agent`

**Context:** `create_scoped_agent` in `src/agents/scoper/agent.py` loads `Config.from_env()` at line 140, then loads it *again* at line 204 to create the `ScopedGenerator`'s LLM provider. The second load ignores `model_name`, `base_url`, and `api_key` passed into the function, so the generator could use different credentials than the agent's chat model.

**Files:**
- Modify: `src/agents/scoper/agent.py:203-206`

**Step 1: Write a failing test**

```python
def test_scoped_agent_generator_uses_passed_config(mocker):
    """Generator LLM provider should use the same config as the agent model."""
    mock_create_agent = mocker.patch("agents.scoper.agent.create_agent")
    mock_create_agent.return_value = mocker.Mock()
    mock_create_llm = mocker.patch("agents.scoper.agent.create_llm_provider")
    mock_create_llm.return_value = mocker.Mock()
    mocker.patch("agents.scoper.agent.Config.from_env")

    from agents.scoper.agent import create_scoped_agent
    create_scoped_agent(
        repo_path="/tmp/test",
        model_name="gpt-4o",
        api_key="test-key-123",
        base_url="https://custom.api",
        use_litellm=True,
    )

    # create_llm_provider should be called with the SAME config used for the agent,
    # not a fresh Config.from_env()
    calls = agents.scoper.agent.Config.from_env.call_count
    assert calls == 1, f"Config.from_env() called {calls} times, expected 1"
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `Config.from_env()` is called twice

**Step 3: Remove duplicate `Config.from_env()` call**

In `src/agents/scoper/agent.py`, remove the second `Config.from_env()` at line 204 and reuse the existing `config` variable from line 140:

Change lines 203-206 from:
```python
    # Create the generation tool (needs LLM and output config)
    config = Config.from_env()
    llm_provider = create_llm_provider(config)
    generator = ScopedGenerator(llm_provider, output_dir)
```
To:
```python
    # Create the generation tool (needs LLM and output config)
    # Reuse the config from above to ensure consistent credentials
    llm_provider = create_llm_provider(config)
    generator = ScopedGenerator(llm_provider, output_dir)
```

**Step 4: Run tests**

Run: `pytest tests/scoper/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/agents/scoper/agent.py
git commit -m "fix: remove duplicate Config.from_env() in create_scoped_agent"
```

---

### Task 4: Fix `generate_structured()` docstring and test mismatch

**Context:** `LiteLLMProvider.generate_structured()` docstring claims "Uses provider-specific structured output when available, falls back to JSON mode" but the code always uses JSON mode. The test `test_litellm_provider_generate_structured` expects a `NotImplementedError` retry/fallback that doesn't exist.

**Files:**
- Modify: `src/agents/llm/litellm_provider.py:96-99` (fix docstring)
- Modify: `tests/test_llm.py:104-137` (fix test)

**Step 1: Fix the docstring to match actual behavior**

In `src/agents/llm/litellm_provider.py`, change lines 96-99 from:
```python
        """Generate structured output using Pydantic schema.

        Uses provider-specific structured output when available,
        falls back to JSON mode + parsing for others.
```
To:
```python
        """Generate structured output using Pydantic schema.

        Uses JSON mode for broad compatibility across providers,
        then parses the response against the provided Pydantic schema.
```

**Step 2: Fix the test to match actual JSON-only behavior**

In `tests/test_llm.py`, replace `test_litellm_provider_generate_structured` with:

```python
def test_litellm_provider_generate_structured(mocker):
    """Test LiteLLMProvider.generate_structured() uses JSON mode."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        count: int

    mock_response = mocker.Mock()
    mock_response.choices = [
        mocker.Mock(message=mocker.Mock(content='{"name": "test", "count": 42}'))
    ]

    mock_completion = mocker.patch("litellm.completion", return_value=mock_response)

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")
    result = provider.generate_structured("Generate data", schema=TestSchema)

    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.count == 42

    # Verify JSON mode was used
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}
```

**Step 3: Run tests**

Run: `pytest tests/test_llm.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/agents/llm/litellm_provider.py tests/test_llm.py
git commit -m "fix: align generate_structured docstring and test with JSON-only implementation"
```

---

### Task 5: Auto-switch provider to `litellm` when `LLM_BASE_URL` is set

**Context:** `.env.example` line 27 says "When set, LLM_PROVIDER is automatically treated as litellm" but `Config.from_env()` doesn't implement this auto-switch.

**Files:**
- Modify: `src/agents/config.py:90-97`
- Test: `tests/test_config.py`

**Step 1: Write a failing test**

```python
def test_base_url_auto_switches_to_litellm(monkeypatch):
    """Setting LLM_BASE_URL should auto-switch provider to litellm."""
    monkeypatch.setenv("LLM_BASE_URL", "https://my-gateway.example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    config = Config.from_env()
    assert config.llm_provider == "litellm"
    assert config.api_base_url == "https://my-gateway.example.com"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_base_url_auto_switches_to_litellm -v`
Expected: FAIL — `config.llm_provider` is `"anthropic"` (the default)

**Step 3: Add auto-switch logic in `Config.from_env()`**

In `src/agents/config.py`, after the `base_url` line and before building `config_dict`, add:

```python
        # Auto-switch to litellm when a custom base URL is set (as documented in .env.example)
        llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
        if base_url and llm_provider == "anthropic" and not os.getenv("LLM_PROVIDER"):
            llm_provider = "litellm"
```

Then update the `config_dict` to use the local `llm_provider` variable instead of reading from env again:
```python
        config_dict = {
            "llm_provider": llm_provider,
            ...
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/agents/config.py tests/test_config.py
git commit -m "fix: auto-switch provider to litellm when LLM_BASE_URL is set"
```

---

### Task 6: Add `google_api_key` to gateway validation

**Context:** In `_validate_api_key()` in `main.py`, gateway mode checks `openai_api_key`, `anthropic_api_key`, and `api_key` but not `google_api_key`. Error message also only mentions OPENAI/ANTHROPIC.

**Files:**
- Modify: `src/agents/main.py:67-69`
- Test: `tests/test_cli.py`

**Step 1: Write a failing test**

```python
def test_validate_api_key_gateway_with_google_key():
    """Gateway mode should accept google_api_key as valid."""
    from agents.main import _validate_api_key
    config = Config(
        llm_provider="litellm",
        model_name="gemini-1.5-pro",
        api_base_url="https://gateway.example.com",
        google_api_key="test-google-key",
    )
    is_valid, error_msg = _validate_api_key(config)
    assert is_valid is True
    assert error_msg == ""
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `google_api_key` is not checked, returns invalid

**Step 3: Fix the validation**

In `src/agents/main.py`, change line 67 from:
```python
            has_any_key = config.openai_api_key or config.anthropic_api_key or config.api_key
```
To:
```python
            has_any_key = config.openai_api_key or config.anthropic_api_key or config.google_api_key or config.api_key
```

And update the error message on line 69 from:
```python
                return False, "Error: No API key set. Set OPENAI_API_KEY or ANTHROPIC_API_KEY for gateway auth."
```
To:
```python
                return False, "Error: No API key set. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for gateway auth."
```

**Step 4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/agents/main.py tests/test_cli.py
git commit -m "fix: include google_api_key in gateway validation check"
```

---

### Task 7: Move `test_agent_config.py` to `scripts/`

**Context:** `test_agent_config.py` is a debug script at the repo root. It's named like a pytest test module, runs code at import time (loads env, prints config, makes API calls), and will be collected by pytest during test discovery.

**Files:**
- Move: `test_agent_config.py` → `scripts/debug_agent_config.py`
- Modify: add `if __name__ == "__main__":` guard

**Step 1: Move and fix the file**

Move `test_agent_config.py` to `scripts/debug_agent_config.py` and wrap all executable code in a `if __name__ == "__main__":` guard. Everything from line 14 onward (after the imports) should be inside the guard.

**Step 2: Verify pytest no longer collects it**

Run: `pytest --collect-only 2>&1 | grep debug_agent`
Expected: No output (not collected)

**Step 3: Commit**

```bash
git rm test_agent_config.py
git add scripts/debug_agent_config.py
git commit -m "fix: move debug script out of pytest collection path"
```

---

### Task 8: Add SSL warning to agent example

**Context:** `docs/examples/agent-example.py` allows `LLM_VERIFY_SSL=false` with no warning. This is a security concern (MITM risk).

**Files:**
- Modify: `docs/examples/agent-example.py:137-145`

**Step 1: Add warning when SSL verification is disabled**

In `docs/examples/agent-example.py`, change the SSL block to:

```python
    verify_ssl = _env_flag("LLM_VERIFY_SSL", default=True)
    http_client = None
    http_async_client = None
    if not verify_ssl:
        import warnings
        import httpx

        warnings.warn(
            "SSL verification is disabled (LLM_VERIFY_SSL=false). "
            "This exposes connections to MITM attacks. Only use for local development.",
            stacklevel=2,
        )
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
```

**Step 2: Commit**

```bash
git add docs/examples/agent-example.py
git commit -m "fix: add security warning when SSL verification is disabled in example"
```

---

### Task 9: Align example env var names with project

**Context:** `docs/examples/agent-example.py` uses `MODEL` and `LLM_API_KEY` env vars, while the project uses `MODEL_NAME` and provider-specific keys. This causes confusion.

**Files:**
- Modify: `docs/examples/agent-example.py:127-135`

**Step 1: Update env var names to match project conventions**

Change the env var reading block to:

```python
def create_customer_support_agent():
    model = os.getenv("MODEL_NAME") or os.getenv("MODEL")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    if not model:
        raise RuntimeError("Missing required env var: MODEL_NAME (or MODEL)")
    if not base_url:
        raise RuntimeError("Missing required env var: LLM_BASE_URL")
    if api_key is None:
        raise RuntimeError(
            "Missing required env var: OPENAI_API_KEY, ANTHROPIC_API_KEY, or LLM_API_KEY"
        )
```

**Step 2: Commit**

```bash
git add docs/examples/agent-example.py
git commit -m "fix: align example env var names with project conventions"
```

---

### Task 10: Final verification

**Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All pass

**Step 2: Commit any remaining fixes if tests fail**

---

## Summary of Changes

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `repository_tools.py` | `_flatten_tree` includes repo root in paths | Start from children of root |
| 2 | `exploration_tools.py` | `list_key_files` expects tree, gets flat list | Accept `list[str]` |
| 3 | `scoper/agent.py` | Double `Config.from_env()` | Remove duplicate |
| 4 | `litellm_provider.py` + `test_llm.py` | Docstring/test claim fallback that doesn't exist | Align with JSON-only impl |
| 5 | `config.py` | `LLM_BASE_URL` doesn't auto-switch to litellm | Add auto-switch logic |
| 6 | `main.py` | Gateway validation ignores `google_api_key` | Add to check |
| 7 | `test_agent_config.py` | Debug script collected by pytest | Move to `scripts/` |
| 8 | `agent-example.py` | SSL disable with no warning | Add `warnings.warn()` |
| 9 | `agent-example.py` | Inconsistent env var names | Align with project |
| 10 | — | Full regression check | Run test suite |
