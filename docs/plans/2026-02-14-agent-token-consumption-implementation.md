# Agent Token Consumption Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce scoped agent token consumption ~50% by passing tighter tool output limits from the agent factory to tool factories and lowering the middleware truncation default.

**Architecture:** The tool factory functions (`create_file_tools`, `create_search_tools`) gain optional limit parameters with backward-compatible defaults. The scoped agent factory passes tighter values. The middleware default drops from 12k to 6k chars. The system prompt adds token economy guidance.

**Tech Stack:** Python 3.x, LangChain, pytest, uv

---

### Task 1: Add limit parameters to `create_file_tools`

**Files:**
- Modify: `src/agents/tools/file.py:239-292`
- Test: `tests/scoper/test_tools.py`

**Context:** `create_file_tools(backend)` currently creates `read_file` and `search_for_files` tools with hardcoded defaults (`DEFAULT_MAX_CHARS=13_500` for read, `max_results=30` for search). We need optional parameters so the scoped agent can pass tighter values.

**Step 1: Write the failing tests**

Add these tests to `tests/scoper/test_tools.py` at the end of the `TestCreateFileTools` class (after line 244):

```python
def test_custom_max_chars(self, tmp_path):
    """Test that create_file_tools respects custom max_chars."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.py").write_text("x" * 500)

    backend = LocalFileBackend(repo)
    tools = create_file_tools(backend, max_chars=100)

    read_tool = next(t for t in tools if t.name == "read_file")
    result = read_tool.invoke({"file_path": "big.py"})

    assert result["truncated"] is True
    assert result["char_count"] == 100

def test_custom_max_search_results(self, tmp_path):
    """Test that create_file_tools respects custom max_search_results."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(10):
        (repo / f"match_{i}.py").write_text("keyword content")

    backend = LocalFileBackend(repo)
    tools = create_file_tools(backend, max_search_results=3)

    search_tool = next(t for t in tools if t.name == "search_for_files")
    result = search_tool.invoke({"keywords": ["match"]})

    assert result["total_found"] <= 3

def test_default_limits_unchanged(self, tmp_path):
    """Test that default limits are preserved when no overrides given."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test.py").write_text("content")

    backend = LocalFileBackend(repo)
    tools = create_file_tools(backend)

    read_tool = next(t for t in tools if t.name == "read_file")
    # Default max_chars should still be 13_500
    result = read_tool.invoke({"file_path": "test.py"})
    assert result["truncated"] is False  # 7 chars < 13500
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/scoper/test_tools.py::TestCreateFileTools::test_custom_max_chars tests/scoper/test_tools.py::TestCreateFileTools::test_custom_max_search_results tests/scoper/test_tools.py::TestCreateFileTools::test_default_limits_unchanged -v`
Expected: FAIL — `create_file_tools()` does not accept `max_chars` or `max_search_results`

**Step 3: Implement the changes**

In `src/agents/tools/file.py`, change the `create_file_tools` function signature (line 239) and both inner tool closures:

```python
def create_file_tools(
    backend: FileBackend,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_search_results: int = 30,
) -> list[BaseTool]:
    """Create file tools bound to a specific backend.

    Args:
        backend: File backend to bind tools to
        max_chars: Maximum characters for read_file (default: 13500)
        max_search_results: Maximum results for search_for_files (default: 30)

    Returns:
        List of LangChain tools ready for agent use
    """

    # Capture the factory-level defaults so inner tools use them
    _default_max_chars = max_chars
    _default_max_search = max_search_results

    @tool
    def read_file(file_path: str, max_chars: int = _default_max_chars) -> dict:
        """Read a file from the repository.

        Use this to examine files you've identified as potentially relevant.
        The file must be within the repository.

        Args:
            file_path: Relative path to file within the repository
            max_chars: Maximum characters to return (default: {_default_max_chars})

        Returns:
            Dictionary with:
            - content: File content (or None if unreadable)
            - path: The requested path
            - char_count: Number of characters returned
            - truncated: Whether content was truncated
            - error: Error message if file couldn't be read
        """
        result = read_file_content(backend, file_path, max_chars)
        return result.model_dump()

    @tool
    def search_for_files(keywords: list[str], max_results: int = _default_max_search) -> dict:
        """Search for files matching keywords in the repository.

        Use this as your first step to find candidate files.
        Searches both filenames/paths and file contents.

        Args:
            keywords: Keywords to search for (case-insensitive)
            max_results: Maximum number of results (default: {_default_max_search})

        Returns:
            Dictionary with:
            - matches: List of matching files with path, match_type, score
            - total_found: Number of matches found
            - keywords_used: Keywords that were searched
            - error: Error message if search failed
        """
        result = search_files(backend, keywords, max_results)
        return result.model_dump()

    return [read_file, search_for_files]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scoper/test_tools.py::TestCreateFileTools -v`
Expected: ALL PASS (3 new + 2 existing)

**Step 5: Commit**

```bash
git add src/agents/tools/file.py tests/scoper/test_tools.py
git commit -m "feat: add limit parameters to create_file_tools"
```

---

### Task 2: Add limit parameters to `create_search_tools`

**Files:**
- Modify: `src/agents/tools/search.py:401-461`
- Test: `tests/test_search_tools.py`

**Context:** `create_search_tools(backend)` currently creates `grep_in_files` and `find_code_definitions` with hardcoded defaults (`max_results=50` for both, `context_lines=2` for grep). We need optional parameters.

**Step 1: Write the failing tests**

Add these tests to `tests/test_search_tools.py` at the end of the `TestCreateSearchTools` class (after line 514):

```python
def test_custom_grep_limits(self, tmp_path):
    """Test that create_search_tools respects custom grep limits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = [f"match line {i}" for i in range(20)]
    (repo / "data.py").write_text("\n".join(lines))

    backend = LocalFileBackend(repo)
    tools = create_search_tools(backend, max_grep_results=5, context_lines=1)

    grep_tool = next(t for t in tools if t.name == "grep_in_files")
    result = grep_tool.invoke({"pattern": "match"})

    assert len(result["matches"]) == 5
    # With context_lines=1, each match should have at most 1 context line
    for match in result["matches"]:
        assert len(match["context_before"]) <= 1
        assert len(match["context_after"]) <= 1

def test_custom_find_defs_limit(self, tmp_path):
    """Test that create_search_tools respects custom find_definitions limit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = [f"def func_{i}(): pass" for i in range(20)]
    (repo / "funcs.py").write_text("\n".join(lines))

    backend = LocalFileBackend(repo)
    tools = create_search_tools(backend, max_def_results=3)

    find_tool = next(t for t in tools if t.name == "find_code_definitions")
    result = find_tool.invoke({"name": "func"})

    assert len(result["definitions"]) == 3

def test_default_search_limits_unchanged(self, tmp_path):
    """Test that default limits are preserved when no overrides given."""
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = [f"match line {i}" for i in range(60)]
    (repo / "data.py").write_text("\n".join(lines))

    backend = LocalFileBackend(repo)
    tools = create_search_tools(backend)

    grep_tool = next(t for t in tools if t.name == "grep_in_files")
    result = grep_tool.invoke({"pattern": "match"})

    # Default max_results is 50
    assert len(result["matches"]) == 50
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search_tools.py::TestCreateSearchTools::test_custom_grep_limits tests/test_search_tools.py::TestCreateSearchTools::test_custom_find_defs_limit tests/test_search_tools.py::TestCreateSearchTools::test_default_search_limits_unchanged -v`
Expected: FAIL — `create_search_tools()` does not accept `max_grep_results`, `context_lines`, or `max_def_results`

**Step 3: Implement the changes**

In `src/agents/tools/search.py`, change the `create_search_tools` function (line 401):

```python
def create_search_tools(
    backend: FileBackend,
    max_grep_results: int = 50,
    max_def_results: int = 50,
    context_lines: int = 2,
) -> list[BaseTool]:
    """Create code search tools bound to a specific backend.

    Args:
        backend: File backend to bind tools to
        max_grep_results: Default max results for grep_in_files (default: 50)
        max_def_results: Default max results for find_code_definitions (default: 50)
        context_lines: Lines of context around grep matches (default: 2)

    Returns:
        List of LangChain tools ready for agent use
    """

    _default_grep_max = max_grep_results
    _default_def_max = max_def_results
    _default_context_lines = context_lines

    @tool
    def grep_in_files(
        pattern: str, path: str | None = None, max_results: int = _default_grep_max
    ) -> dict:
        """Search for a regex pattern in repository files.

        Returns matches with file paths, line numbers, and surrounding context.
        Use this to find specific code patterns, function calls, error messages,
        or any text across the codebase.

        Args:
            pattern: Regex pattern to search for (case-insensitive)
            path: Optional specific file to search (searches all files if None)
            max_results: Maximum matches to return (default: {_default_grep_max})

        Returns:
            Dictionary with:
            - matches: List of {path, line_num, content, context_before, context_after}
            - total_matches: Total matches found (may exceed max_results)
            - pattern: The pattern that was searched
            - files_searched: Number of files searched
            - error: Error message if search failed
        """
        result = grep_pattern(backend, pattern, path, max_results, _default_context_lines)
        return result.model_dump()

    @tool
    def find_code_definitions(
        name: str, def_type: str | None = None, max_results: int = _default_def_max
    ) -> dict:
        """Find function, class, method, or variable definitions by name.

        Searches Python and JavaScript/TypeScript files for definitions
        matching the given name. Supports partial matching.

        Args:
            name: Name to search for (partial match, case-insensitive)
            def_type: Optional filter: "function", "class", "method", "variable", or None for all
            max_results: Maximum definitions to return (default: {_default_def_max})

        Returns:
            Dictionary with:
            - definitions: List of {name, def_type, path, line_num, line_end, signature}
            - name_searched: The name that was searched
            - files_searched: Number of files searched
            - error: Error message if search failed
        """
        result = find_definitions(backend, name, def_type, max_results)
        return result.model_dump()

    return [grep_in_files, find_code_definitions]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search_tools.py::TestCreateSearchTools -v`
Expected: ALL PASS (3 new + 3 existing)

**Step 5: Commit**

```bash
git add src/agents/tools/search.py tests/test_search_tools.py
git commit -m "feat: add limit parameters to create_search_tools"
```

---

### Task 3: Lower `max_tool_output_chars` default and wire agent limits

**Files:**
- Modify: `src/agents/config.py:47,121`
- Modify: `src/agents/scoper/agent.py:169-171`
- Test: `tests/test_config.py`
- Test: `tests/scoper/test_scoped_agent.py`

**Context:** The middleware `max_tool_output_chars` default is 12,000 chars. Lower it to 6,000. Also wire the scoped agent factory to pass tighter tool limits: `max_chars=8000`, `max_search_results=15`, `max_grep_results=15`, `max_def_results=15`, `context_lines=1`.

**Step 1: Write the failing tests**

In `tests/test_config.py`, add a test (find the test class for config defaults):

```python
def test_max_tool_output_chars_default_is_6000():
    """max_tool_output_chars should default to 6000."""
    config = Config()
    assert config.max_tool_output_chars == 6000
```

In `tests/scoper/test_scoped_agent.py`, add a test in `TestCreateScopedAgent` (after the `test_config_from_env_called_once` test):

```python
def test_passes_tight_tool_limits(
    self,
    sample_repo,
    mock_config,
    mock_llm,
    mock_create_agent,
    mock_init_chat_model,
):
    """Test that agent factory passes tighter limits to tool factories."""
    from src.agents.scoper.agent import create_scoped_agent

    with patch("src.agents.scoper.agent.create_file_tools") as mock_file_tools, \
         patch("src.agents.scoper.agent.create_search_tools") as mock_search_tools, \
         patch("src.agents.scoper.agent.create_analysis_tools") as mock_analysis_tools:
        mock_file_tools.return_value = []
        mock_search_tools.return_value = []
        mock_analysis_tools.return_value = []

        create_scoped_agent(sample_repo)

        # File tools should get tighter limits
        mock_file_tools.assert_called_once()
        ft_kwargs = mock_file_tools.call_args
        assert ft_kwargs[1].get("max_chars") == 8000 or (len(ft_kwargs[0]) > 1 and ft_kwargs[0][1] == 8000)

        # Search tools should get tighter limits
        mock_search_tools.assert_called_once()
        st_kwargs = mock_search_tools.call_args
        assert st_kwargs[1].get("max_grep_results") == 15
        assert st_kwargs[1].get("context_lines") == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::test_max_tool_output_chars_default_is_6000 tests/scoper/test_scoped_agent.py::TestCreateScopedAgent::test_passes_tight_tool_limits -v`
Expected: FAIL — config default is 12000, agent doesn't pass limits

**Step 3: Implement the changes**

**3a. In `src/agents/config.py`:**

Line 47 — change default:
```python
    max_tool_output_chars: int = Field(default=6000)
```

Line 121 — change `from_env` fallback:
```python
            "max_tool_output_chars": _parse_int(os.getenv("MAX_TOOL_OUTPUT_CHARS"), 6000),
```

**3b. In `src/agents/scoper/agent.py`:**

Lines 168-171 — pass tighter limits to tool factories:
```python
    # Create file, analysis, and code search tools bound to backend
    # Use tighter limits for agent mode to reduce token consumption
    file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
    analysis_tools = create_analysis_tools(backend)
    code_search_tools = create_search_tools(
        backend, max_grep_results=15, max_def_results=15, context_lines=1
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/scoper/test_scoped_agent.py -v`
Expected: ALL PASS

Then run the full suite to check for regressions:

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (351+ tests)

**Step 5: Commit**

```bash
git add src/agents/config.py src/agents/scoper/agent.py tests/test_config.py tests/scoper/test_scoped_agent.py
git commit -m "feat: lower tool output limits for agent mode and middleware default"
```

---

### Task 4: Add system prompt token economy guidance

**Files:**
- Modify: `src/agents/scoper/agent.py:24-96` (the `SCOPED_AGENT_SYSTEM_PROMPT` string)
- Test: `tests/scoper/test_scoped_agent.py`

**Context:** The system prompt currently has no guidance about keeping tool outputs small. Add a short "Token Economy" section advising the agent to use small `max_results` values.

**Step 1: Write the failing test**

In `tests/scoper/test_scoped_agent.py`, add to `TestScopedAgentSystemPrompt`:

```python
def test_prompt_includes_token_economy(self):
    """Test that system prompt includes token economy guidance."""
    from src.agents.scoper.agent import SCOPED_AGENT_SYSTEM_PROMPT

    assert "Token Economy" in SCOPED_AGENT_SYSTEM_PROMPT or "token" in SCOPED_AGENT_SYSTEM_PROMPT.lower()
    # Should mention keeping results small
    assert "max_results" in SCOPED_AGENT_SYSTEM_PROMPT
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoper/test_scoped_agent.py::TestScopedAgentSystemPrompt::test_prompt_includes_token_economy -v`
Expected: FAIL — prompt doesn't mention token economy or max_results

**Step 3: Implement the change**

In `src/agents/scoper/agent.py`, add a new section to `SCOPED_AGENT_SYSTEM_PROMPT` right before the `## Important Notes` section (before line 88). Insert:

```
## Token Economy

Every tool result is added to the conversation and sent with each subsequent API call. Large results compound quickly.

- **grep_in_files**: Start with `max_results=5`. Only increase if you need more matches.
- **search_for_files**: Start with `max_results=5`. Narrow with more specific keywords rather than increasing results.
- **find_code_definitions**: Start with `max_results=5`.
- **read_file**: Large files are automatically truncated. If you only need a section, note the relevant lines and move on.
- Prefer targeted grep searches over reading entire files.
- If a tool returns too many results, refine the query instead of reading them all.

```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scoper/test_scoped_agent.py::TestScopedAgentSystemPrompt -v`
Expected: ALL PASS (3 existing + 1 new)

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/agents/scoper/agent.py tests/scoper/test_scoped_agent.py
git commit -m "feat: add token economy guidance to scoped agent system prompt"
```

---

### Task 5: End-to-end verification

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (355+ tests)

**Step 2: Verify backward compatibility**

Check that calling `create_file_tools(backend)` and `create_search_tools(backend)` without extra args still works by running existing tests:

Run: `uv run pytest tests/scoper/test_tools.py::TestCreateFileTools::test_creates_tools tests/scoper/test_tools.py::TestCreateFileTools::test_tools_are_bound_to_backend tests/test_search_tools.py::TestCreateSearchTools::test_creates_two_tools tests/test_search_tools.py::TestCreateSearchTools::test_grep_tool_invocation -v`
Expected: ALL PASS — existing callers unaffected

**Step 3: Verify config default**

Run: `uv run python -c "from src.agents.config import Config; c = Config(); print(f'max_tool_output_chars={c.max_tool_output_chars}'); assert c.max_tool_output_chars == 6000"`
Expected: Prints `max_tool_output_chars=6000`

**Step 4: Done**

All 4 fixes implemented, tested, and committed. Ready for branch finishing.
