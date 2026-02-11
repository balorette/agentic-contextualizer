# LiteLLM Multi-Provider Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add support for 100+ LLM providers through LiteLLM while maintaining backward compatibility with existing Anthropic integration.

**Architecture:** Dual-mode provider system - keep existing `AnthropicProvider` as default, add new `LiteLLMProvider` as opt-in alternative. Factory function routes to appropriate provider based on config. CLI flags override environment variables.

**Tech Stack:** Python 3.11+, LiteLLM, Pydantic, pytest, pytest-mock

---

## Task 1: Install LiteLLM Dependency

**Files:**
- Modify: `pyproject.toml:7-19`

**Step 1: Add litellm to dependencies**

```toml
dependencies = [
    "langchain>=1.0.0",
    "langchain-anthropic>=0.1.0",
    "langchain-core>=0.3.0",
    "langgraph>=0.2.0",
    "langsmith>=0.1.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "pathspec>=0.12.1",
    "rich>=13.0.0",
    "litellm>=1.0.0",
]
```

**Step 2: Install the dependency**

Run: `uv pip install -e ".[dev]"`
Expected: Package installed successfully

**Step 3: Verify installation**

Run: `python -c "import litellm; print(litellm.__version__)"`
Expected: Version number printed (e.g., "1.52.0")

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add litellm for multi-provider support"
```

---

## Task 2: Update Config Class - Add Provider Fields

**Files:**
- Modify: `src/agents/config.py:25-34`
- Test: `tests/test_config.py`

**Step 1: Write test for new config fields**

Add to `tests/test_config.py`:

```python
def test_config_from_env_provider_keys(monkeypatch):
    """Test loading provider-specific API keys from environment."""
    monkeypatch.setenv("LLM_PROVIDER", "litellm")
    monkeypatch.setenv("MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    config = Config.from_env()

    assert config.llm_provider == "litellm"
    assert config.model_name == "gpt-4o"
    assert config.openai_api_key == "sk-test-openai"
    assert config.google_api_key == "test-google-key"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_from_env_provider_keys -v`
Expected: FAIL - AttributeError: 'Config' object has no attribute 'openai_api_key'

**Step 3: Add new fields to Config class**

In `src/agents/config.py`, update the Config class:

```python
class Config(BaseModel):
    """Application configuration."""

    # LLM Settings
    llm_provider: str = Field(default="anthropic")
    model_name: str = Field(default="claude-3-5-sonnet-20241022")
    api_key: Optional[str] = Field(default=None)
    api_base_url: Optional[str] = Field(default=None)

    # Provider-specific API keys
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)

    max_retries: int = Field(default=3)
    timeout: int = Field(default=60)
```

**Step 4: Update from_env() to load provider keys**

In `src/agents/config.py`, update the `from_env()` method's config_dict:

```python
config_dict = {
    "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
    "model_name": os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022"),
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "api_base_url": os.getenv("ANTHROPIC_BASE_URL"),

    # Provider-specific keys
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "google_api_key": os.getenv("GOOGLE_API_KEY"),

    "max_retries": _parse_int(os.getenv("LLM_MAX_RETRIES"), 3),
    "timeout": _parse_int(os.getenv("LLM_TIMEOUT"), 60),
    "max_file_size": _parse_int(os.getenv("MAX_FILE_SIZE"), 1_000_000),
    "ignored_dirs": ignored_dirs,
    "output_dir": Path(output_dir_env) if output_dir_env else Path("contexts"),
}
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_config_from_env_provider_keys -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/agents/config.py tests/test_config.py
git commit -m "feat(config): add provider-specific API key fields"
```

---

## Task 3: Add CLI Override Support to Config

**Files:**
- Modify: `src/agents/config.py:42-69`
- Test: `tests/test_config.py`

**Step 1: Write test for CLI overrides**

Add to `tests/test_config.py`:

```python
def test_config_cli_overrides(monkeypatch):
    """Test CLI overrides take precedence over env vars."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL_NAME", "claude-3-5-sonnet-20241022")

    config = Config.from_env(cli_overrides={
        "llm_provider": "litellm",
        "model_name": "gpt-4o",
    })

    assert config.llm_provider == "litellm"
    assert config.model_name == "gpt-4o"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_cli_overrides -v`
Expected: FAIL - TypeError: from_env() got an unexpected keyword argument 'cli_overrides'

**Step 3: Update from_env() signature and implementation**

In `src/agents/config.py`, update the method:

```python
@classmethod
def from_env(cls, cli_overrides: Optional[dict] = None) -> "Config":
    """Load configuration from environment variables with optional CLI overrides.

    Args:
        cli_overrides: Dict of config fields to override (from CLI flags)

    Returns:
        Config instance with merged settings
    """
    # ... existing code to build config_dict ...

    # Apply CLI overrides
    if cli_overrides:
        config_dict.update({k: v for k, v in cli_overrides.items() if v is not None})

    return cls(**config_dict)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_config_cli_overrides -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/config.py tests/test_config.py
git commit -m "feat(config): add CLI override support"
```

---

## Task 4: Create LiteLLMProvider Class - Basic Structure

**Files:**
- Create: `src/agents/llm/litellm_provider.py`
- Test: `tests/test_llm.py`

**Step 1: Write test for LiteLLMProvider initialization**

Add to `tests/test_llm.py`:

```python
from src.agents.llm.litellm_provider import LiteLLMProvider

def test_litellm_provider_init():
    """Test LiteLLMProvider initialization stores config."""
    provider = LiteLLMProvider(
        model_name="gpt-4o",
        api_key="test-key",
        base_url="https://custom.api",
        max_retries=5,
        timeout=120,
    )

    assert provider.model_name == "gpt-4o"
    assert provider.api_key == "test-key"
    assert provider.base_url == "https://custom.api"
    assert provider.max_retries == 5
    assert provider.timeout == 120
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_litellm_provider_init -v`
Expected: FAIL - ModuleNotFoundError: No module named 'src.agents.llm.litellm_provider'

**Step 3: Create LiteLLMProvider class file**

Create `src/agents/llm/litellm_provider.py`:

```python
"""LiteLLM provider for multi-provider LLM support."""

from typing import Optional, Type
from pydantic import BaseModel
import litellm
from .provider import LLMProvider, LLMResponse


class LiteLLMProvider(LLMProvider):
    """LLM provider using LiteLLM for multi-provider support."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """Initialize LiteLLM provider.

        Args:
            model_name: LiteLLM model identifier (e.g., gpt-4o, claude-3-5-sonnet-20241022)
            api_key: API key for the provider (optional for local models)
            base_url: Custom API endpoint URL (maps to api_base)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate response using LiteLLM."""
        raise NotImplementedError("To be implemented in next task")

    def generate_structured(
        self, prompt: str, system: Optional[str] = None, schema: Type[BaseModel] = None
    ) -> BaseModel:
        """Generate structured output using Pydantic schema."""
        raise NotImplementedError("To be implemented in later task")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py::test_litellm_provider_init -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/llm/litellm_provider.py tests/test_llm.py
git commit -m "feat(llm): add LiteLLMProvider class skeleton"
```

---

## Task 5: Implement LiteLLMProvider.generate() Method

**Files:**
- Modify: `src/agents/llm/litellm_provider.py:33-51`
- Test: `tests/test_llm.py`

**Step 1: Write test for generate() method**

Add to `tests/test_llm.py`:

```python
def test_litellm_provider_generate(mocker):
    """Test LiteLLMProvider.generate() with mocked litellm."""
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock(message=mocker.Mock(content="Test response"))]
    mock_response.usage = mocker.Mock(total_tokens=100)

    mock_completion = mocker.patch("litellm.completion", return_value=mock_response)

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")
    response = provider.generate("Test prompt", system="Test system")

    assert response.content == "Test response"
    assert response.model == "gpt-4o"
    assert response.tokens_used == 100

    # Verify litellm.completion was called correctly
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"] == "Test system"
    assert call_kwargs["messages"][1]["role"] == "user"
    assert call_kwargs["messages"][1]["content"] == "Test prompt"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_litellm_provider_generate -v`
Expected: FAIL - NotImplementedError

**Step 3: Implement generate() method**

In `src/agents/llm/litellm_provider.py`, replace the generate() method:

```python
def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
    """Generate response using LiteLLM.

    Args:
        prompt: User prompt
        system: Optional system prompt

    Returns:
        LLMResponse with generated content

    Raises:
        RuntimeError: If generation fails with clear error message
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        response = litellm.completion(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model_name,
            tokens_used=response.usage.total_tokens if hasattr(response, "usage") else None,
        )
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {str(e)}") from e
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py::test_litellm_provider_generate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/llm/litellm_provider.py tests/test_llm.py
git commit -m "feat(llm): implement LiteLLMProvider.generate()"
```

---

## Task 6: Add Error Handling with Provider Detection

**Files:**
- Modify: `src/agents/llm/litellm_provider.py`
- Test: `tests/test_llm.py`

**Step 1: Write test for authentication error handling**

Add to `tests/test_llm.py`:

```python
def test_litellm_provider_auth_error(mocker):
    """Test LiteLLMProvider handles auth errors with clear messages."""
    import litellm

    mocker.patch(
        "litellm.completion",
        side_effect=litellm.AuthenticationError("Invalid API key")
    )

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="bad-key")

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("Test")

    assert "OpenAI authentication failed" in str(exc_info.value)
    assert "OPENAI_API_KEY" in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_litellm_provider_auth_error -v`
Expected: FAIL - RuntimeError message doesn't match expected format

**Step 3: Add _detect_provider() helper method**

In `src/agents/llm/litellm_provider.py`, add after the generate() method:

```python
def _detect_provider(self, model_name: str) -> str:
    """Detect provider from model name for error messages."""
    if model_name.startswith("gpt-") or model_name.startswith("o1"):
        return "OpenAI"
    elif model_name.startswith("claude"):
        return "Anthropic"
    elif model_name.startswith("gemini"):
        return "Google"
    elif model_name.startswith("ollama"):
        return "Ollama"
    else:
        return "LLM Provider"
```

**Step 4: Update generate() error handling**

In `src/agents/llm/litellm_provider.py`, update the except block in generate():

```python
    except litellm.AuthenticationError as e:
        provider = self._detect_provider(self.model_name)
        raise RuntimeError(
            f"{provider} authentication failed. "
            f"Set {provider.upper()}_API_KEY in your .env file"
        ) from e
    except litellm.RateLimitError as e:
        raise RuntimeError(f"Rate limit exceeded for {self.model_name}") from e
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {str(e)}") from e
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_llm.py::test_litellm_provider_auth_error -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/agents/llm/litellm_provider.py tests/test_llm.py
git commit -m "feat(llm): add provider-specific error handling"
```

---

## Task 7: Implement generate_structured() Method

**Files:**
- Modify: `src/agents/llm/litellm_provider.py`
- Test: `tests/test_llm.py`

**Step 1: Write test for generate_structured()**

Add to `tests/test_llm.py`:

```python
from pydantic import BaseModel

class TestSchema(BaseModel):
    name: str
    count: int

def test_litellm_provider_generate_structured(mocker):
    """Test LiteLLMProvider.generate_structured() with JSON fallback."""
    import json

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_litellm_provider_generate_structured -v`
Expected: FAIL - NotImplementedError

**Step 3: Implement generate_structured() method**

In `src/agents/llm/litellm_provider.py`, replace generate_structured():

```python
def generate_structured(
    self, prompt: str, system: Optional[str] = None, schema: Type[BaseModel] = None
) -> BaseModel:
    """Generate structured output using Pydantic schema.

    Uses LiteLLM's response_format for providers that support it,
    falls back to JSON mode + parsing for others.

    Args:
        prompt: User prompt
        system: Optional system prompt
        schema: Pydantic model class defining output structure

    Returns:
        Instance of the provided Pydantic schema

    Raises:
        RuntimeError: If generation fails
        ValueError: If schema is not provided
    """
    if schema is None:
        raise ValueError("schema parameter is required")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        # Try structured output if provider supports it
        try:
            kwargs["response_format"] = schema
            response = litellm.completion(**kwargs)
            return response
        except (NotImplementedError, AttributeError):
            # Fall back to JSON mode + parsing
            kwargs["response_format"] = {"type": "json_object"}
            response = litellm.completion(**kwargs)
            import json
            data = json.loads(response.choices[0].message.content)
            return schema(**data)

    except Exception as e:
        raise RuntimeError(f"Structured generation failed: {str(e)}") from e
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py::test_litellm_provider_generate_structured -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/llm/litellm_provider.py tests/test_llm.py
git commit -m "feat(llm): implement generate_structured() with JSON fallback"
```

---

## Task 8: Create Provider Factory Function

**Files:**
- Modify: `src/agents/llm/provider.py` (add at end)
- Test: `tests/test_llm.py`

**Step 1: Write test for factory creating LiteLLMProvider**

Add to `tests/test_llm.py`:

```python
from src.agents.llm.provider import create_llm_provider
from src.agents.config import Config

def test_create_llm_provider_litellm():
    """Test factory creates LiteLLMProvider for litellm config."""
    config = Config(
        llm_provider="litellm",
        model_name="gpt-4o",
        openai_api_key="test-key"
    )
    provider = create_llm_provider(config)

    assert isinstance(provider, LiteLLMProvider)
    assert provider.model_name == "gpt-4o"
    assert provider.api_key == "test-key"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_create_llm_provider_litellm -v`
Expected: FAIL - ImportError: cannot import name 'create_llm_provider'

**Step 3: Write test for factory creating AnthropicProvider**

Add to `tests/test_llm.py`:

```python
def test_create_llm_provider_anthropic():
    """Test factory creates AnthropicProvider for default config."""
    config = Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key"
    )
    provider = create_llm_provider(config)

    assert isinstance(provider, AnthropicProvider)
```

**Step 4: Run test to verify it fails**

Run: `pytest tests/test_llm.py::test_create_llm_provider_anthropic -v`
Expected: FAIL - ImportError: cannot import name 'create_llm_provider'

**Step 5: Implement _resolve_api_key_for_model() helper**

Add to end of `src/agents/llm/provider.py`:

```python
def _resolve_api_key_for_model(model_name: str, config: "Config") -> Optional[str]:
    """Resolve which API key to use based on model name.

    Args:
        model_name: LiteLLM model identifier
        config: Application configuration

    Returns:
        Appropriate API key or None for local models
    """
    if model_name.startswith("gpt-") or model_name.startswith("o1"):
        return config.openai_api_key
    elif model_name.startswith("claude"):
        return config.anthropic_api_key or config.api_key
    elif model_name.startswith("gemini") or model_name.startswith("vertex"):
        return config.google_api_key
    elif model_name.startswith("ollama") or model_name.startswith("lmstudio"):
        return None  # Local models don't need API keys
    else:
        # For other providers, return None
        return None
```

**Step 6: Implement create_llm_provider() factory**

Add to end of `src/agents/llm/provider.py` (after _resolve_api_key_for_model):

```python
def create_llm_provider(config: "Config") -> LLMProvider:
    """Factory function to create the appropriate LLM provider.

    Args:
        config: Application configuration

    Returns:
        Configured LLM provider instance (AnthropicProvider or LiteLLMProvider)
    """
    if config.llm_provider == "litellm":
        from .litellm_provider import LiteLLMProvider

        # Determine which API key to use based on model
        api_key = _resolve_api_key_for_model(config.model_name, config)

        return LiteLLMProvider(
            model_name=config.model_name,
            api_key=api_key,
            base_url=config.api_base_url,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )
    else:  # "anthropic" or default
        return AnthropicProvider(
            model_name=config.model_name,
            api_key=config.anthropic_api_key or config.api_key,
            base_url=config.api_base_url,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )
```

**Step 7: Add Config import at top of provider.py**

At the top of `src/agents/llm/provider.py`, add:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
```

**Step 8: Run tests to verify they pass**

Run: `pytest tests/test_llm.py::test_create_llm_provider_litellm tests/test_llm.py::test_create_llm_provider_anthropic -v`
Expected: Both PASS

**Step 9: Commit**

```bash
git add src/agents/llm/provider.py tests/test_llm.py
git commit -m "feat(llm): add provider factory function"
```

---

## Task 9: Update main.py Pipeline Mode - Generate Command

**Files:**
- Modify: `src/agents/main.py:131-133`

**Step 1: Update imports**

At top of `src/agents/main.py`, add:

```python
from .llm.provider import create_llm_provider
```

**Step 2: Replace AnthropicProvider in _generate_pipeline_mode**

In `src/agents/main.py`, find line 131-133 and replace:

```python
# Before
llm = AnthropicProvider(config.model_name, config.api_key, base_url=config.api_base_url)

# After
llm = create_llm_provider(config)
```

**Step 3: Test manually**

Run: `python -m agents.main generate . -s "Test" --provider anthropic`
Expected: Should work with existing Anthropic setup

**Step 4: Commit**

```bash
git add src/agents/main.py
git commit -m "refactor(main): use provider factory in generate pipeline"
```

---

## Task 10: Update main.py Pipeline Mode - Refine and Scope

**Files:**
- Modify: `src/agents/main.py:268` (refine)
- Modify: `src/agents/main.py:488` (scope)

**Step 1: Update refine pipeline mode**

In `src/agents/main.py` line 268, replace:

```python
# Before
llm = AnthropicProvider(config.model_name, config.api_key, base_url=config.api_base_url)

# After
llm = create_llm_provider(config)
```

**Step 2: Update scope pipeline mode**

In `src/agents/main.py` line 488, replace:

```python
# Before
llm = AnthropicProvider(config.model_name, config.api_key, base_url=config.api_base_url)

# After
llm = create_llm_provider(config)
```

**Step 3: Commit**

```bash
git add src/agents/main.py
git commit -m "refactor(main): use provider factory in refine and scope pipelines"
```

---

## Task 11: Update repository_tools.py Provider

**Files:**
- Modify: `src/agents/tools/repository_tools.py:21-27`

**Step 1: Update imports**

At top of `src/agents/tools/repository_tools.py`, add:

```python
from ..llm.provider import create_llm_provider
```

**Step 2: Replace _get_llm_provider() implementation**

Replace the function:

```python
def _get_llm_provider() -> AnthropicProvider:
    """Get LLM provider instance from config.

    Returns:
        Configured Anthropic LLM provider
    """
    return AnthropicProvider(_config.model_name, _config.api_key, base_url=_config.api_base_url)
```

With:

```python
def _get_llm_provider() -> LLMProvider:
    """Get LLM provider instance from config.

    Returns:
        Configured LLM provider
    """
    return create_llm_provider(_config)
```

**Step 3: Update return type import**

Change import at top:

```python
# Before
from ..llm.provider import AnthropicProvider

# After
from ..llm.provider import LLMProvider, create_llm_provider
```

**Step 4: Commit**

```bash
git add src/agents/tools/repository_tools.py
git commit -m "refactor(tools): use provider factory in repository tools"
```

---

## Task 12: Update scoper/agent.py Provider

**Files:**
- Modify: `src/agents/scoper/agent.py:143`

**Step 1: Update imports**

At top of `src/agents/scoper/agent.py`, add:

```python
from ..llm.provider import create_llm_provider
```

**Step 2: Replace provider instantiation**

In the `create_scoped_agent` function around line 143, replace:

```python
# Before
config = Config.from_env()
llm_provider = AnthropicProvider(config.model_name, config.api_key, base_url=config.api_base_url)

# After
config = Config.from_env()
llm_provider = create_llm_provider(config)
```

**Step 3: Remove AnthropicProvider import**

Remove from imports at top:

```python
from ..llm.provider import AnthropicProvider
```

**Step 4: Commit**

```bash
git add src/agents/scoper/agent.py
git commit -m "refactor(scoper): use provider factory in scoped agent"
```

---

## Task 13: Add CLI Arguments - Generate Command

**Files:**
- Modify: `src/agents/main.py:55-68`

**Step 1: Add --provider and --model options**

In `src/agents/main.py`, update the generate command:

```python
@cli.command()
@click.argument("source")
@click.option("--summary", "-s", required=True, help="Brief description of the project")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--provider", help="LLM provider: anthropic or litellm")
@click.option("--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
@click.option("--debug", is_flag=True, help="Enable debug output for agent mode")
@click.option("--stream", is_flag=True, help="Enable streaming output for agent mode (real-time feedback)")
def generate(source: str, summary: str, output: str | None, mode: str, provider: str | None, model: str | None, debug: bool, stream: bool):
```

**Step 2: Build CLI overrides dict**

At the start of the generate function, add:

```python
# Build CLI overrides dict
cli_overrides = {}
if provider:
    cli_overrides["llm_provider"] = provider
if model:
    cli_overrides["model_name"] = model

# Load config with overrides
config = Config.from_env(cli_overrides=cli_overrides)
```

**Step 3: Replace existing config line**

Remove the old line:

```python
config = Config.from_env()
```

**Step 4: Commit**

```bash
git add src/agents/main.py
git commit -m "feat(cli): add --provider and --model flags to generate command"
```

---

## Task 14: Add CLI Arguments - Refine Command

**Files:**
- Modify: `src/agents/main.py:226-238`

**Step 1: Add options to refine command**

```python
@cli.command()
@click.argument("context_file", type=click.Path(exists=True))
@click.option("--request", "-r", required=True, help="What to change")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--provider", help="LLM provider: anthropic or litellm")
@click.option("--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
@click.option("--debug", is_flag=True, help="Enable debug output for agent mode")
@click.option("--stream", is_flag=True, help="Enable streaming output for agent mode (real-time feedback)")
def refine(context_file: str, request: str, mode: str, provider: str | None, model: str | None, debug: bool, stream: bool):
```

**Step 2: Add CLI overrides handling**

At start of refine function:

```python
# Build CLI overrides dict
cli_overrides = {}
if provider:
    cli_overrides["llm_provider"] = provider
if model:
    cli_overrides["model_name"] = model

config = Config.from_env(cli_overrides=cli_overrides)
```

**Step 3: Remove old config line**

Remove:

```python
config = Config.from_env()
```

**Step 4: Commit**

```bash
git add src/agents/main.py
git commit -m "feat(cli): add --provider and --model flags to refine command"
```

---

## Task 15: Add CLI Arguments - Scope Command

**Files:**
- Modify: `src/agents/main.py:362-375`

**Step 1: Add options to scope command**

```python
@cli.command()
@click.argument("source")
@click.option("--question", "-q", required=True, help="Question/topic to scope to")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--provider", help="LLM provider: anthropic or litellm")
@click.option("--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--stream", is_flag=True, help="Enable streaming output (agent mode)")
def scope(source: str, question: str, output: str | None, mode: str, provider: str | None, model: str | None, debug: bool, stream: bool):
```

**Step 2: Add CLI overrides handling**

At start of scope function:

```python
# Build CLI overrides dict
cli_overrides = {}
if provider:
    cli_overrides["llm_provider"] = provider
if model:
    cli_overrides["model_name"] = model

config = Config.from_env(cli_overrides=cli_overrides)
```

**Step 3: Remove old config line**

**Step 4: Commit**

```bash
git add src/agents/main.py
git commit -m "feat(cli): add --provider and --model flags to scope command"
```

---

## Task 16: Write Backward Compatibility Test

**Files:**
- Test: `tests/test_llm.py`

**Step 1: Write backward compatibility test**

Add to `tests/test_llm.py`:

```python
def test_legacy_config_still_works(monkeypatch):
    """Test that old .env files work unchanged."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
    # Don't set LLM_PROVIDER - should default to anthropic

    config = Config.from_env()
    provider = create_llm_provider(config)

    assert config.llm_provider == "anthropic"
    assert isinstance(provider, AnthropicProvider)
    assert provider.api_key == "test-key"
```

**Step 2: Run test**

Run: `pytest tests/test_llm.py::test_legacy_config_still_works -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_llm.py
git commit -m "test(llm): add backward compatibility test"
```

---

## Task 17: Update .env.example with Multi-Provider Config

**Files:**
- Modify: `.env.example`

**Step 1: Update .env.example**

Replace the LLM configuration section:

```bash
# LLM Provider Configuration
# Options: "anthropic" (default) or "litellm"
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3-5-sonnet-20241022

# Provider-specific API keys
ANTHROPIC_API_KEY=your_api_key_here
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...

# Optional: Custom API endpoint (for proxies or alternative providers)
# ANTHROPIC_BASE_URL=https://api.anthropic.com

# LLM Settings
LLM_MAX_RETRIES=3
LLM_TIMEOUT=60

# Scanner Settings
MAX_FILE_SIZE=1000000
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: update .env.example with multi-provider config"
```

---

## Task 18: Update README.md with Multi-Provider Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add Multi-Provider Support section**

After the Configuration section in README.md, add:

```markdown
## Multi-Provider Support

The tool supports multiple LLM providers through LiteLLM integration. Use any provider you have access to:

### OpenAI

```bash
# .env
LLM_PROVIDER=litellm
MODEL_NAME=gpt-4o
OPENAI_API_KEY=sk-...
```

### Google Gemini

```bash
# .env
LLM_PROVIDER=litellm
MODEL_NAME=gemini/gemini-2.0-flash-exp
GOOGLE_API_KEY=...
```

### Local Models (Ollama)

```bash
# .env
LLM_PROVIDER=litellm
MODEL_NAME=ollama/llama3
# No API key needed
```

### CLI Overrides

Override provider/model for a single command:

```bash
python -m agents.main generate /path/to/repo \
  --summary "Description" \
  --provider litellm \
  --model gpt-4o
```

### Supported Providers

LiteLLM supports 100+ providers including:
- OpenAI (GPT-4o, GPT-4, GPT-3.5)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus/Haiku)
- Google (Gemini, Vertex AI)
- AWS Bedrock
- Azure OpenAI
- Ollama (local models)
- And many more...

See [LiteLLM's provider docs](https://docs.litellm.ai/docs/providers) for the complete list.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add multi-provider support documentation"
```

---

## Task 19: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 2: Check test coverage**

Run: `pytest tests/ --cov=src/agents --cov-report=term-missing`
Expected: >80% coverage on modified files

**Step 3: If tests fail, fix and commit**

Address any failures, then:

```bash
git add <fixed-files>
git commit -m "fix: address test failures"
```

---

## Task 20: Manual Testing with OpenAI (Optional)

**Prerequisites:** OPENAI_API_KEY environment variable set

**Step 1: Test with OpenAI in pipeline mode**

```bash
LLM_PROVIDER=litellm MODEL_NAME=gpt-4o-mini \
python -m agents.main generate . \
  --summary "Test project" \
  --mode pipeline
```

Expected: Successfully generates context using GPT-4o-mini

**Step 2: Test with CLI override**

```bash
python -m agents.main generate . \
  --summary "Test project" \
  --provider litellm \
  --model gpt-4o-mini
```

Expected: Works without env vars set

**Step 3: Verify output**

Check `contexts/<repo-name>/context.md` exists and contains reasonable content.

---

## Task 21: Final Integration Test

**Step 1: Test backward compatibility**

```bash
# Use existing Anthropic setup (no changes to .env)
python -m agents.main generate . --summary "Test"
```

Expected: Works exactly as before

**Step 2: Test config precedence**

```bash
# Env says anthropic, CLI says litellm
LLM_PROVIDER=anthropic \
python -m agents.main generate . \
  --summary "Test" \
  --provider litellm \
  --model gpt-4o-mini
```

Expected: Uses LiteLLM (CLI overrides env)

**Step 3: Document results**

Note: All tests passing, ready for PR.

---

## Completion Checklist

- [ ] All tests passing
- [ ] Backward compatibility verified
- [ ] Documentation updated (README.md, .env.example)
- [ ] Manual testing completed (at least with Anthropic)
- [ ] Code committed in logical chunks
- [ ] Ready to create PR

## Notes for PR Description

**What:** Add multi-provider LLM support via LiteLLM

**Why:** Enable users to choose their preferred LLM provider (OpenAI, Google, local models) while maintaining full backward compatibility

**How:**
- Dual-mode system: AnthropicProvider (default) + LiteLLMProvider (opt-in)
- Factory function routes based on config
- CLI flags override environment variables
- 100+ providers supported out of the box

**Testing:**
- Unit tests for all new code
- Backward compatibility tests
- Manual testing with [providers tested]

**Breaking Changes:** None - fully backward compatible
