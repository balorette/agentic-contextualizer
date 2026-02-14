# Scoped Agent Token Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix agent mode token limit errors with five targeted changes across four files.

**Architecture:** Surgical fixes to existing code — set missing defaults, change one tool signature from content-passing to path-passing, fix model name handling in estimator, share TPM throttle instance. No new files or abstractions.

**Tech Stack:** Python 3.11+, pytest, LangChain, LiteLLM

---

### Task 1: Set `max_input_tokens` and `max_output_tokens` defaults in Config

**Files:**
- Modify: `src/agents/config.py:45-46`
- Test: `tests/test_config.py`

**Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_config_default_max_input_tokens():
    """max_input_tokens should default to 128000 to enable message trimming."""
    config = Config(api_key="test")
    assert config.max_input_tokens == 128_000


def test_config_default_max_output_tokens():
    """max_output_tokens should default to 16384 for complex tool call args."""
    config = Config(api_key="test")
    assert config.max_output_tokens == 16384


def test_config_max_input_tokens_from_env(monkeypatch):
    """LLM_MAX_INPUT_TOKENS env var should override default."""
    monkeypatch.setenv("LLM_MAX_INPUT_TOKENS", "64000")
    config = Config.from_env()
    assert config.max_input_tokens == 64000
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py::test_config_default_max_input_tokens tests/test_config.py::test_config_default_max_output_tokens tests/test_config.py::test_config_max_input_tokens_from_env -v`

Expected: FAIL — `max_input_tokens` is `None`, `max_output_tokens` is `4096`

**Step 3: Implement the fix**

In `src/agents/config.py`, change lines 45-46:

```python
# Before:
max_output_tokens: Optional[int] = Field(default=4096)
max_input_tokens: Optional[int] = Field(default=None)

# After:
max_output_tokens: Optional[int] = Field(default=16384)
max_input_tokens: Optional[int] = Field(default=128_000)
```

Also update `from_env()` line 119-120 to handle the new default correctly. The existing `_parse_int` logic with the `if os.getenv(...)` guard already handles this — when the env var is unset, it won't override the field default. Verify this by reading the code. The current line:

```python
"max_input_tokens": _parse_int(os.getenv("LLM_MAX_INPUT_TOKENS"), None) if os.getenv("LLM_MAX_INPUT_TOKENS") else None,
```

Must change to use the new default as fallback:

```python
"max_input_tokens": _parse_int(os.getenv("LLM_MAX_INPUT_TOKENS"), 128_000) if os.getenv("LLM_MAX_INPUT_TOKENS") else 128_000,
"max_output_tokens": _parse_int(os.getenv("LLM_MAX_OUTPUT_TOKENS"), 16384),
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`

Expected: ALL PASS (including existing tests)

**Step 5: Commit**

```bash
git add src/agents/config.py tests/test_config.py
git commit -m "fix: set max_input_tokens=128k and max_output_tokens=16k defaults

Activates TokenBudgetMiddleware message trimming that was never firing
because max_input_tokens defaulted to None. Raises max_output_tokens
from 4096 to 16384 for complex tool call arguments."
```

---

### Task 2: Strip provider prefix in token estimator model name

**Files:**
- Modify: `src/agents/llm/chat_model_factory.py:163-189`
- Test: `tests/agents/llm/test_chat_model_factory.py`

**Step 1: Write the failing test**

Add to `tests/agents/llm/test_chat_model_factory.py`:

```python
def test_build_token_middleware_strips_provider_prefix():
    """Token estimator should receive model name without provider prefix.

    litellm.token_counter() fails on 'anthropic:claude-...' format,
    falling back to inaccurate char-based estimation.
    """
    from src.agents.llm.chat_model_factory import build_token_middleware
    from src.agents.config import Config

    config = Config()
    mw = build_token_middleware(config, "anthropic:claude-sonnet-4-5-20250929")

    # model_name stored on the middleware should NOT have the prefix
    assert mw.model_name == "claude-sonnet-4-5-20250929"
    assert ":" not in mw.model_name
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/llm/test_chat_model_factory.py::test_build_token_middleware_strips_provider_prefix -v`

Expected: FAIL — `mw.model_name` is `"anthropic:claude-sonnet-4-5-20250929"`

**Step 3: Implement the fix**

In `src/agents/llm/chat_model_factory.py`, modify `build_token_middleware()`:

```python
def build_token_middleware(
    config: Config,
    model_name: str,
) -> "TokenBudgetMiddleware":
    from .rate_limiting import TPMThrottle
    from .token_estimator import LiteLLMTokenEstimator
    from ..middleware.token_budget import TokenBudgetMiddleware
    from .provider import _strip_provider_prefix  # Add this import

    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    estimator = LiteLLMTokenEstimator()

    return TokenBudgetMiddleware(
        max_input_tokens=config.max_input_tokens,
        max_tool_output_chars=config.max_tool_output_chars,
        throttle=throttle,
        estimator=estimator,
        model_name=_strip_provider_prefix(model_name),  # Strip prefix
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/agents/llm/test_chat_model_factory.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/agents/llm/chat_model_factory.py tests/agents/llm/test_chat_model_factory.py
git commit -m "fix: strip provider prefix before token estimation

litellm.token_counter() cannot parse 'anthropic:model-name' format,
causing fallback to inaccurate 4-chars/token estimation."
```

---

### Task 3: Share TPM throttle between agent model and generation provider

**Files:**
- Modify: `src/agents/llm/chat_model_factory.py:163-189` (add `throttle` param)
- Modify: `src/agents/llm/provider.py:213-256` (add `throttle` param)
- Modify: `src/agents/scoper/agent.py:92-241` (wire shared throttle)
- Test: `tests/agents/llm/test_chat_model_factory.py`
- Test: `tests/scoper/test_scoped_agent.py`

**Step 1: Write the failing tests**

Add to `tests/agents/llm/test_chat_model_factory.py`:

```python
def test_build_token_middleware_accepts_shared_throttle():
    """Should use provided throttle instead of creating a new one."""
    from src.agents.llm.chat_model_factory import build_token_middleware
    from src.agents.llm.rate_limiting import TPMThrottle
    from src.agents.config import Config

    shared_throttle = TPMThrottle(max_tpm=50000, safety_factor=0.9)
    config = Config()
    mw = build_token_middleware(config, "test-model", throttle=shared_throttle)

    assert mw.throttle is shared_throttle
```

Add to `tests/agents/llm/test_rate_limiting.py` (in `TestRateLimitedProvider`):

```python
def test_create_llm_provider_accepts_shared_throttle(self):
    """create_llm_provider should use provided throttle when given."""
    from src.agents.llm.provider import create_llm_provider
    from src.agents.llm.rate_limiting import TPMThrottle, RateLimitedProvider
    from src.agents.config import Config

    shared_throttle = TPMThrottle(max_tpm=50000, safety_factor=0.9)
    config = Config(api_key="test-key", anthropic_api_key="test-key")
    provider = create_llm_provider(config, throttle=shared_throttle)

    assert isinstance(provider, RateLimitedProvider)
    assert provider.throttle is shared_throttle
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/agents/llm/test_chat_model_factory.py::test_build_token_middleware_accepts_shared_throttle tests/agents/llm/test_rate_limiting.py::TestRateLimitedProvider::test_create_llm_provider_accepts_shared_throttle -v`

Expected: FAIL — `build_token_middleware()` and `create_llm_provider()` don't accept `throttle` param

**Step 3: Implement the fix**

In `src/agents/llm/chat_model_factory.py`, add `throttle` param to `build_token_middleware()`:

```python
def build_token_middleware(
    config: Config,
    model_name: str,
    throttle: "TPMThrottle | None" = None,
) -> "TokenBudgetMiddleware":
    from .rate_limiting import TPMThrottle
    from .token_estimator import LiteLLMTokenEstimator
    from ..middleware.token_budget import TokenBudgetMiddleware
    from .provider import _strip_provider_prefix

    if throttle is None:
        throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    estimator = LiteLLMTokenEstimator()

    return TokenBudgetMiddleware(
        max_input_tokens=config.max_input_tokens,
        max_tool_output_chars=config.max_tool_output_chars,
        throttle=throttle,
        estimator=estimator,
        model_name=_strip_provider_prefix(model_name),
    )
```

In `src/agents/llm/provider.py`, add `throttle` param to `create_llm_provider()`:

```python
def create_llm_provider(config: "Config", throttle: "TPMThrottle | None" = None) -> LLMProvider:
    from .rate_limiting import RateLimitedProvider, TPMThrottle, RetryHandler
    from .token_estimator import LiteLLMTokenEstimator

    # ... (existing inner provider creation logic unchanged) ...

    if throttle is None:
        throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)

    return RateLimitedProvider(
        provider=inner,
        throttle=throttle,
        estimator=LiteLLMTokenEstimator(),
        retry_handler=RetryHandler(config.retry_max_attempts, config.retry_initial_wait),
        max_tokens_per_call=config.max_tokens_per_call,
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/agents/llm/test_chat_model_factory.py tests/agents/llm/test_rate_limiting.py -v`

Expected: ALL PASS

**Step 5: Wire shared throttle in scoped agent factory**

In `src/agents/scoper/agent.py`, inside `create_scoped_agent()`, create one throttle and share it:

```python
from ..llm.rate_limiting import TPMThrottle

# Create shared throttle instance
throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)

# Pass to both middleware and generation provider
budget_mw = build_token_middleware(config, model_name, throttle=throttle)
llm_provider = create_llm_provider(config, throttle=throttle)
```

**Step 6: Run full test suite to verify no regressions**

Run: `pytest tests/scoper/ tests/agents/llm/ -v`

Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/agents/llm/chat_model_factory.py src/agents/llm/provider.py src/agents/scoper/agent.py tests/agents/llm/test_chat_model_factory.py tests/agents/llm/test_rate_limiting.py
git commit -m "fix: share TPM throttle between agent loop and generation provider

Both LLM paths in scoped agent now use the same TPMThrottle instance,
preventing them from independently exceeding rate limits."
```

---

### Task 4: Redesign `generate_scoped_context` tool to accept paths instead of contents

**Files:**
- Modify: `src/agents/scoper/agent.py:21-89` (system prompt), `src/agents/scoper/agent.py:166-224` (tool)
- Test: `tests/scoper/test_scoped_agent.py`

**Step 1: Write the failing test**

Add to `tests/scoper/test_scoped_agent.py`:

```python
class TestGenerateScopedContextTool:
    """Tests for the generate_scoped_context tool signature and behavior."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository with files."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "auth.py").write_text(
            "def login(user, password):\n    return authenticate(user, password)\n"
        )
        (repo / "src" / "models.py").write_text(
            "class User:\n    name: str\n    email: str\n"
        )
        return repo

    @pytest.fixture
    def mock_generator(self):
        """Mock ScopedGenerator to avoid real LLM calls."""
        with patch("src.agents.scoper.agent.ScopedGenerator") as mock_cls:
            generator_instance = MagicMock()
            generator_instance.generate.return_value = Path("/tmp/output.md")
            mock_cls.return_value = generator_instance
            yield generator_instance

    @pytest.fixture
    def mock_create_agent(self):
        with patch("src.agents.scoper.agent.create_agent") as mock:
            mock.return_value = MagicMock()
            yield mock

    @pytest.fixture
    def mock_init_chat_model(self):
        with patch("src.agents.llm.chat_model_factory.init_chat_model") as mock:
            yield mock

    def test_generate_tool_accepts_paths_not_contents(
        self,
        sample_repo,
        mock_generator,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """generate_scoped_context should accept file paths, not file contents.

        The tool reads files itself via the backend — the LLM should not
        have to re-emit file contents as tool call arguments.
        """
        from src.agents.scoper.agent import create_scoped_agent

        create_scoped_agent(sample_repo, config=Config(api_key="test"))

        # Extract the generate_scoped_context tool from the tools passed to create_agent
        call_kwargs = mock_create_agent.call_args[1]
        tools = call_kwargs["tools"]
        gen_tool = next(t for t in tools if t.name == "generate_scoped_context")

        # Check tool schema: should have relevant_file_paths (list[str]),
        # NOT relevant_files (dict[str, str])
        schema = gen_tool.args_schema.model_json_schema()
        props = schema["properties"]
        assert "relevant_file_paths" in props, "Tool should accept relevant_file_paths"
        assert "relevant_files" not in props, "Tool should NOT accept relevant_files dict"
        assert props["relevant_file_paths"]["type"] == "array"

    def test_generate_tool_reads_files_via_backend(
        self,
        sample_repo,
        mock_generator,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """Tool should read file contents from backend using the provided paths."""
        from src.agents.scoper.agent import create_scoped_agent

        create_scoped_agent(sample_repo, config=Config(api_key="test"))

        call_kwargs = mock_create_agent.call_args[1]
        tools = call_kwargs["tools"]
        gen_tool = next(t for t in tools if t.name == "generate_scoped_context")

        # Invoke the tool with file paths
        result = gen_tool.invoke({
            "question": "How does auth work?",
            "relevant_file_paths": ["src/auth.py", "src/models.py"],
            "insights": "Auth uses password-based login",
        })

        # Generator should have been called with actual file contents
        mock_generator.generate.assert_called_once()
        call_kwargs = mock_generator.generate.call_args[1]
        relevant_files = call_kwargs["relevant_files"]
        assert "src/auth.py" in relevant_files
        assert "def login" in relevant_files["src/auth.py"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/scoper/test_scoped_agent.py::TestGenerateScopedContextTool -v`

Expected: FAIL — tool still has `relevant_files: dict[str, str]` signature

**Step 3: Implement the tool signature change**

In `src/agents/scoper/agent.py`, replace the `generate_scoped_context` tool definition (lines 166-224):

```python
    @tool
    def generate_scoped_context(
        question: str,
        relevant_file_paths: list[str],
        insights: str,
        code_references: list[dict] | None = None,
    ) -> dict:
        """Generate the final scoped context markdown file.

        Call this when you have gathered sufficient context to answer the question.
        Pass the PATHS of relevant files — the tool reads their contents automatically.

        Args:
            question: The original scope question being answered
            relevant_file_paths: List of file paths the agent determined are relevant
            insights: Your analysis and insights about the code
            code_references: Optional list of code reference dicts with keys:
                - path: File path
                - line_start: Starting line number
                - line_end: Optional ending line number
                - description: Brief description of what this code does

        Returns:
            Dictionary with:
            - output_path: Path to generated markdown file
            - error: Error message if generation failed
        """
        try:
            # Read file contents via backend
            relevant_files = {}
            for file_path in relevant_file_paths:
                content = backend.read_file(file_path)
                if content is not None:
                    relevant_files[file_path] = content

            # Convert code reference dicts to CodeReference objects
            refs = None
            if code_references:
                refs = [
                    CodeReference(
                        path=ref["path"],
                        line_start=ref["line_start"],
                        line_end=ref.get("line_end"),
                        description=ref["description"],
                    )
                    for ref in code_references
                ]

            output_path = generator.generate(
                repo_name=repo_path.name,
                question=question,
                relevant_files=relevant_files,
                insights=insights,
                model_name=config.model_name,
                source_repo=str(repo_path),
                code_references=refs,
            )
            return {
                "output_path": str(output_path),
                "error": None,
            }
        except Exception as e:
            return {
                "output_path": None,
                "error": str(e),
            }
```

**Step 4: Update the system prompt**

In `src/agents/scoper/agent.py`, update `SCOPED_AGENT_SYSTEM_PROMPT` (lines 21-89). Key changes:

1. Tool description for `generate_scoped_context` — change "relevant files and their contents" to "file paths"
2. Step 4 guidance — tell agent to pass paths, not contents:

```
### Step 4: Generate
When you have sufficient context (typically 5-15 relevant files), use `generate_scoped_context` with:
- The list of **file paths** you found relevant (the tool reads contents automatically)
- Your analysis and insights
- Code references with specific line numbers
```

3. Add a guideline:
```
- **File Paths**: When calling generate_scoped_context, pass file paths — NOT file contents. The tool reads files automatically.
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/scoper/test_scoped_agent.py -v`

Expected: ALL PASS

**Step 6: Run full test suite for regressions**

Run: `pytest tests/ -v`

Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/agents/scoper/agent.py tests/scoper/test_scoped_agent.py
git commit -m "fix: generate_scoped_context accepts file paths instead of contents

The LLM no longer needs to re-emit file contents as tool call arguments.
The tool reads files via the backend, applying the same truncation limits
as pipeline mode. Eliminates context doubling in agent mode."
```

---

### Task 5: Verify end-to-end and run full test suite

**Files:**
- None (verification only)

**Step 1: Run the complete test suite**

Run: `pytest tests/ -v --tb=short`

Expected: ALL PASS

**Step 2: Verify the existing `test_config_tpm_defaults` test still passes**

The existing test at `tests/test_config.py:66-73` checks:
```python
def test_config_tpm_defaults():
    config = Config(api_key="test")
    assert config.max_tokens_per_call is None
```

This should still pass since we only changed `max_input_tokens` and `max_output_tokens`.

**Step 3: Spot-check with debug output (manual)**

If a test environment with API key is available:

```bash
python -m agents.main scope /path/to/repo -q "some question" --mode agent --debug
```

Verify in debug output:
- `max_input_tokens: 128000` appears in config
- `max_output_tokens: 16384` appears in config
- Token estimator receives model name without `:` prefix
- No token limit errors on a question that previously failed

**Step 4: Final commit (if any test fixups needed)**

```bash
git add -A
git commit -m "fix: address test fixups from token limit fixes"
```
