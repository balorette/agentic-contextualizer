# LiteLLM Multi-Provider Integration Design

**Date:** 2026-02-10
**Status:** Design Complete
**Goal:** Add support for multiple LLM providers through LiteLLM while maintaining backward compatibility

## Overview

Enable users to choose any LLM provider (OpenAI, Google, local models, etc.) based on their access and preferences. Current Anthropic-only integration will remain as the default, with LiteLLM as an opt-in alternative supporting 100+ providers.

## Architecture

### Dual-Mode Provider System

The system will support two LLM initialization modes:

1. **Legacy Mode** (default): Current `AnthropicProvider` - no changes to existing behavior
2. **LiteLLM Mode**: New `LiteLLMProvider` supporting all LiteLLM providers

**Provider Selection Logic:**
```
If LLM_PROVIDER is set and != "anthropic":
    → Use LiteLLMProvider
Else if LLM_PROVIDER == "anthropic" or not set:
    → Use AnthropicProvider (current behavior)
```

### Configuration Hierarchy

**Sources (in order of precedence):**
1. CLI flags (`--model`, `--provider`) - per-command overrides
2. Environment variables (`.env` file) - base configuration

**Key Configuration Fields:**
- `LLM_PROVIDER`: `"anthropic"` (default) or `"litellm"`
- `MODEL_NAME`: Model identifier (e.g., `gpt-4o`, `claude-3-5-sonnet-20241022`)
- Provider-specific API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc.

### Backward Compatibility

- Existing `.env` files continue to work unchanged
- `AnthropicProvider` remains the default code path
- No breaking changes to current behavior
- LiteLLM is opt-in via `LLM_PROVIDER=litellm`

## Implementation

### 1. New LiteLLMProvider Class

**File:** `src/agents/llm/litellm_provider.py`

```python
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

### 2. Configuration Updates

**File:** `src/agents/config.py`

```python
class Config(BaseModel):
    """Application configuration."""

    # LLM Settings
    llm_provider: str = Field(default="anthropic")  # "anthropic" or "litellm"
    model_name: str = Field(default="claude-3-5-sonnet-20241022")

    # Legacy (backward compat)
    api_key: Optional[str] = Field(default=None)
    api_base_url: Optional[str] = Field(default=None)

    # Provider-specific API keys
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)

    max_retries: int = Field(default=3)
    timeout: int = Field(default=60)

    # Scanner Settings
    max_file_size: int = Field(default=1_000_000)
    ignored_dirs: list[str] = Field(default_factory=lambda: DEFAULT_IGNORED_DIRS.copy())

    # Output Settings
    output_dir: Path = Field(default=Path("contexts"))

    @classmethod
    def from_env(cls, cli_overrides: Optional[dict] = None) -> "Config":
        """Load configuration from environment variables with optional CLI overrides.

        Args:
            cli_overrides: Dict of config fields to override (from CLI flags)

        Returns:
            Config instance with merged settings
        """
        def _parse_int(value: Optional[str], fallback: int) -> int:
            try:
                return int(value) if value is not None else fallback
            except ValueError:
                return fallback

        # Read ignored directories
        ignored_dirs = DEFAULT_IGNORED_DIRS.copy()
        extra_ignored = os.getenv("IGNORED_DIRS")
        if extra_ignored:
            ignored_dirs.extend([entry.strip() for entry in extra_ignored.split(",") if entry.strip()])

        output_dir_env = os.getenv("OUTPUT_DIR")

        # Build config dict from environment
        config_dict = {
            "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
            "model_name": os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022"),

            # Legacy keys (backward compat)
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

        # Apply CLI overrides
        if cli_overrides:
            config_dict.update({k: v for k, v in cli_overrides.items() if v is not None})

        return cls(**config_dict)
```

### 3. Provider Factory

**File:** `src/agents/llm/provider.py` (add to existing file)

```python
def create_llm_provider(config: Config) -> LLMProvider:
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


def _resolve_api_key_for_model(model_name: str, config: Config) -> Optional[str]:
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
        # For other providers, try to infer from model name or return None
        return None
```

### 4. Update Provider Instantiations

Replace direct `AnthropicProvider(...)` calls with `create_llm_provider(config)`:

**Locations to update:**
- `src/agents/main.py` - 3 pipeline mode functions
- `src/agents/tools/repository_tools.py` - `_get_llm_provider()`
- `src/agents/scoper/agent.py` - generation tool initialization

**Example update:**
```python
# Before
llm = AnthropicProvider(config.model_name, config.api_key, base_url=config.api_base_url)

# After
from .llm.provider import create_llm_provider
llm = create_llm_provider(config)
```

### 5. CLI Argument Updates

Add to all commands in `main.py`:

```python
@cli.command()
@click.argument("source")
@click.option("--summary", "-s", required=True, help="Brief description of the project")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--mode", "-m", type=click.Choice(["pipeline", "agent"]), default="pipeline")
@click.option("--provider", help="LLM provider: anthropic or litellm")  # NEW
@click.option("--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)")  # NEW
@click.option("--debug", is_flag=True, help="Enable debug output")
def generate(source: str, summary: str, output: str | None, mode: str, provider: str | None, model: str | None, debug: bool):
    """Generate context for a repository."""
    # Build CLI overrides dict
    cli_overrides = {}
    if provider:
        cli_overrides["llm_provider"] = provider
    if model:
        cli_overrides["model_name"] = model

    # Load config with overrides
    config = Config.from_env(cli_overrides=cli_overrides)

    # Rest of function continues as normal
    ...
```

### 6. Agent Mode Integration

Agent mode already uses `init_chat_model()` which supports LiteLLM models natively. No changes needed - just ensure `config.model_name` is passed correctly.

**Current code (factory.py):**
```python
model = init_chat_model(model_name, **model_kwargs)
```

This will work for LiteLLM models like `gpt-4o`, `gemini-2.0-flash-exp`, etc.

## Testing Strategy

### Unit Tests

**File:** `tests/test_llm.py`

```python
def test_litellm_provider_generate(mocker):
    """Test LiteLLMProvider.generate() with mocked litellm."""
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock(message=mocker.Mock(content="Test response"))]
    mock_response.usage.total_tokens = 100

    mocker.patch("litellm.completion", return_value=mock_response)

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")
    response = provider.generate("Test prompt")

    assert response.content == "Test response"
    assert response.tokens_used == 100


def test_create_llm_provider_anthropic():
    """Test factory creates AnthropicProvider for default config."""
    config = Config(llm_provider="anthropic", api_key="test-key")
    provider = create_llm_provider(config)

    assert isinstance(provider, AnthropicProvider)


def test_create_llm_provider_litellm():
    """Test factory creates LiteLLMProvider for litellm config."""
    config = Config(llm_provider="litellm", model_name="gpt-4o", openai_api_key="test-key")
    provider = create_llm_provider(config)

    assert isinstance(provider, LiteLLMProvider)


def test_config_cli_overrides():
    """Test CLI overrides take precedence over env vars."""
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["MODEL_NAME"] = "claude-3-5-sonnet-20241022"

    config = Config.from_env(cli_overrides={
        "llm_provider": "litellm",
        "model_name": "gpt-4o",
    })

    assert config.llm_provider == "litellm"
    assert config.model_name == "gpt-4o"
```

### Integration Tests (Optional)

```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
def test_litellm_openai_real():
    """Test real OpenAI call via LiteLLM."""
    provider = LiteLLMProvider(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response = provider.generate("Say hello")
    assert response.content
    assert response.tokens_used > 0
```

### Backward Compatibility Tests

```python
def test_legacy_env_vars_still_work():
    """Test that old .env files work unchanged."""
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["MODEL_NAME"] = "claude-3-5-sonnet-20241022"
    # Don't set LLM_PROVIDER

    config = Config.from_env()
    provider = create_llm_provider(config)

    assert isinstance(provider, AnthropicProvider)
    assert provider.api_key == "test-key"
```

## Documentation

### README.md Updates

Add new section after "Configuration":

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

### .env.example Updates

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

## Implementation Checklist

- [ ] Install LiteLLM dependency (`uv pip install litellm`)
- [ ] Create `LiteLLMProvider` class
- [ ] Add `create_llm_provider()` factory function
- [ ] Update `Config` class with new fields and `from_env()` method
- [ ] Add CLI arguments (`--provider`, `--model`) to all commands
- [ ] Update provider instantiations (main.py, repository_tools.py, scoper/agent.py)
- [ ] Write unit tests for `LiteLLMProvider` and factory
- [ ] Write config override tests
- [ ] Write backward compatibility tests
- [ ] Update README.md with multi-provider examples
- [ ] Update .env.example
- [ ] Update CLAUDE.md with architecture notes
- [ ] Manual testing with OpenAI, local Ollama
- [ ] Create PR with full test coverage

## Migration Path

**Phase 1: Add LiteLLM (This Design)**
- Implement dual-mode system
- Keep Anthropic as default
- Add opt-in LiteLLM support

**Phase 2: Encourage Migration (Future)**
- Update docs to promote LiteLLM
- Add helpful warnings/tips

**Phase 3: Deprecate Legacy (Far Future)**
- Mark `AnthropicProvider` as deprecated
- Migrate default to LiteLLM
- Eventually remove legacy code

## Open Questions

None - design is complete and ready for implementation.
