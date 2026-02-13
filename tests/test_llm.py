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


from agents.llm.rate_limiting import RateLimitedProvider


def test_create_llm_provider_litellm():
    """Test factory creates RateLimitedProvider wrapping LiteLLMProvider."""
    config = Config(
        llm_provider="litellm",
        model_name="gpt-4o",
        openai_api_key="test-key"
    )
    provider = create_llm_provider(config)

    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, LiteLLMProvider)
    assert provider.provider.model_name == "gpt-4o"
    assert provider.provider.api_key == "test-key"


def test_create_llm_provider_anthropic():
    """Test factory creates RateLimitedProvider wrapping AnthropicProvider."""
    config = Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key"
    )
    provider = create_llm_provider(config)

    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, AnthropicProvider)


def test_legacy_config_still_works(monkeypatch):
    """Test that old .env files work unchanged."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    config = Config.from_env()
    provider = create_llm_provider(config)

    assert config.llm_provider == "anthropic"
    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, AnthropicProvider)
    assert provider.provider.api_key == "test-key"


def test_create_llm_provider_wraps_with_rate_limiting():
    """Factory should return a RateLimitedProvider with correct TPM config."""
    config = Config(
        llm_provider="litellm",
        model_name="gpt-4o",
        openai_api_key="test-key",
        max_tpm=50000,
        tpm_safety_factor=0.9,
    )
    provider = create_llm_provider(config)
    assert isinstance(provider, RateLimitedProvider)
    assert provider.throttle.max_tpm == 50000
    assert provider.throttle.effective_limit == 45000  # 50000 * 0.9


def test_create_llm_provider_anthropic_wrapped():
    """Anthropic provider should also be wrapped."""
    config = Config(
        llm_provider="anthropic",
        anthropic_api_key="test-key",
    )
    provider = create_llm_provider(config)
    assert isinstance(provider, RateLimitedProvider)
    assert isinstance(provider.provider, AnthropicProvider)
