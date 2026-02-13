"""Tests for configuration loading."""

from pathlib import Path
from agents.config import Config, DEFAULT_IGNORED_DIRS


def test_config_from_env_overrides(monkeypatch, tmp_path):
    """Environment variables should override defaults."""
    monkeypatch.setenv("MODEL_NAME", "custom-model")
    monkeypatch.setenv("MAX_FILE_SIZE", "2048")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "contexts"))
    monkeypatch.setenv("IGNORED_DIRS", "coverage,.mypy_cache")

    config = Config.from_env()

    assert config.model_name == "custom-model"
    assert config.max_file_size == 2048
    assert Path(config.output_dir) == tmp_path / "contexts"

    # Ensure new ignored directories are appended to defaults
    assert set(DEFAULT_IGNORED_DIRS).issubset(set(config.ignored_dirs))
    assert "coverage" in config.ignored_dirs
    assert ".mypy_cache" in config.ignored_dirs


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


def test_base_url_auto_switches_to_litellm(monkeypatch):
    """Setting LLM_BASE_URL should auto-switch provider to litellm."""
    monkeypatch.setenv("LLM_BASE_URL", "https://my-gateway.example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    config = Config.from_env()
    assert config.llm_provider == "litellm"
    assert config.api_base_url == "https://my-gateway.example.com"


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
