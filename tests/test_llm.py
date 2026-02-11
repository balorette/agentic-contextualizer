"""Tests for LLM provider."""

import os
import pytest
from agents.llm.provider import AnthropicProvider, LLMResponse, create_llm_provider
from agents.llm.litellm_provider import LiteLLMProvider
from agents.config import Config


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


def test_litellm_provider_auth_error(mocker):
    """Test LiteLLMProvider handles auth errors with clear messages."""
    import litellm

    mocker.patch(
        "litellm.completion",
        side_effect=litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4o"
        )
    )

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="bad-key")

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("Test")

    assert "OpenAI authentication failed" in str(exc_info.value)
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_litellm_provider_generate_structured(mocker):
    """Test LiteLLMProvider.generate_structured() with JSON fallback."""
    import json
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        count: int

    # Mock to simulate fallback to JSON mode
    call_count = [0]

    def mock_completion(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call with response_format=schema raises NotImplementedError
            raise NotImplementedError("Structured output not supported")
        else:
            # Second call with JSON mode returns JSON string
            mock_response = mocker.Mock()
            mock_response.choices = [
                mocker.Mock(message=mocker.Mock(content='{"name": "test", "count": 42}'))
            ]
            return mock_response

    mocker.patch("litellm.completion", side_effect=mock_completion)

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")
    result = provider.generate_structured("Generate data", schema=TestSchema)

    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.count == 42


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


def test_create_llm_provider_anthropic():
    """Test factory creates AnthropicProvider for default config."""
    config = Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key"
    )
    provider = create_llm_provider(config)

    assert isinstance(provider, AnthropicProvider)


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
