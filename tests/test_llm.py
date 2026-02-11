"""Tests for LLM provider."""

import os
import pytest
from agents.llm.provider import AnthropicProvider, LLMResponse
from agents.config import Config
from src.agents.llm.litellm_provider import LiteLLMProvider


def test_anthropic_provider_initialization(config):
    """Test Anthropic provider can be initialized."""
    provider = AnthropicProvider(model_name=config.model_name, api_key=config.api_key)
    assert provider.model_name == config.model_name
    assert provider.client is not None


def _real_api_key_present() -> bool:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(api_key and api_key != "your_api_key_here")


@pytest.mark.skipif(
    not _real_api_key_present(), reason="Requires a real ANTHROPIC_API_KEY environment variable"
)
def test_anthropic_provider_generate():
    """Integration test for Anthropic provider (requires API key)."""
    config = Config.from_env()
    provider = AnthropicProvider(model_name=config.model_name, api_key=config.api_key)

    response = provider.generate(
        prompt="Say 'Hello, World!' and nothing else.", system="You are a helpful assistant."
    )

    assert isinstance(response, LLMResponse)
    assert "Hello" in response.content
    assert response.model == config.model_name


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
