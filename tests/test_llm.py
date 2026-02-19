"""Tests for LLM provider."""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from agents.llm.provider import AnthropicProvider, LLMResponse, create_llm_provider, coerce_content
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


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_litellm_provider_init(mock_chat_litellm_cls):
    """Test LiteLLMProvider initialization stores config and creates client."""
    mock_chat_litellm_cls.return_value = MagicMock()

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
    assert provider.client is not None

    # Verify ChatLiteLLM was constructed with correct kwargs
    call_kwargs = mock_chat_litellm_cls.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["api_base"] == "https://custom.api"
    assert call_kwargs["max_retries"] == 5
    assert call_kwargs["request_timeout"] == 120


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_litellm_provider_generate(mock_chat_litellm_cls):
    """Test LiteLLMProvider.generate() with mocked ChatLiteLLM."""
    mock_ai_message = AIMessage(
        content="Test response",
        usage_metadata={"total_tokens": 100, "input_tokens": 40, "output_tokens": 60},
    )
    mock_client = MagicMock()
    mock_client.invoke.return_value = mock_ai_message
    mock_chat_litellm_cls.return_value = mock_client

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")
    response = provider.generate("Test prompt", system="Test system")

    assert response.content == "Test response"
    assert response.model == "gpt-4o"
    assert response.tokens_used == 100

    # Verify invoke was called with LangChain messages
    mock_client.invoke.assert_called_once()
    messages = mock_client.invoke.call_args[0][0]
    assert len(messages) == 2
    assert messages[0].content == "Test system"
    assert messages[1].content == "Test prompt"


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_litellm_provider_generate_no_system(mock_chat_litellm_cls):
    """Test LiteLLMProvider.generate() without system prompt."""
    mock_ai_message = AIMessage(content="Response")
    mock_client = MagicMock()
    mock_client.invoke.return_value = mock_ai_message
    mock_chat_litellm_cls.return_value = mock_client

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")
    provider.generate("Just a prompt")

    messages = mock_client.invoke.call_args[0][0]
    assert len(messages) == 1
    assert messages[0].content == "Just a prompt"


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_litellm_provider_auth_error(mock_chat_litellm_cls):
    """Test LiteLLMProvider propagates errors as RuntimeError."""
    mock_client = MagicMock()
    mock_client.invoke.side_effect = Exception("Authentication failed: Invalid API key")
    mock_chat_litellm_cls.return_value = mock_client

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="bad-key")

    with pytest.raises(RuntimeError, match="LLM generation failed"):
        provider.generate("Test")


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_litellm_provider_generate_structured(mock_chat_litellm_cls):
    """Test LiteLLMProvider.generate_structured() uses with_structured_output."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        count: int

    expected_result = TestSchema(name="test", count=42)

    mock_structured_chain = MagicMock()
    mock_structured_chain.invoke.return_value = expected_result

    mock_client = MagicMock()
    mock_client.with_structured_output.return_value = mock_structured_chain
    mock_chat_litellm_cls.return_value = mock_client

    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")
    result = provider.generate_structured("Generate data", schema=TestSchema)

    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.count == 42

    # Verify with_structured_output was called with the schema class
    mock_client.with_structured_output.assert_called_once_with(TestSchema)
    mock_structured_chain.invoke.assert_called_once()


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_litellm_provider_generate_structured_requires_schema(mock_chat_litellm_cls):
    """Test generate_structured raises ValueError when schema is None."""
    mock_chat_litellm_cls.return_value = MagicMock()
    provider = LiteLLMProvider(model_name="gpt-4o", api_key="test-key")

    with pytest.raises(ValueError, match="schema parameter is required"):
        provider.generate_structured("Generate data")


from agents.llm.rate_limiting import RateLimitedProvider


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_create_llm_provider_litellm(mock_chat_litellm_cls):
    """Test factory creates RateLimitedProvider wrapping LiteLLMProvider."""
    mock_chat_litellm_cls.return_value = MagicMock()

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


@patch("agents.llm.litellm_provider.ChatLiteLLM")
def test_create_llm_provider_wraps_with_rate_limiting(mock_chat_litellm_cls):
    """Factory should return a RateLimitedProvider with correct TPM config."""
    mock_chat_litellm_cls.return_value = MagicMock()

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


def test_resolve_api_key_strips_provider_prefix():
    """Provider-prefixed model names (openai:gpt-4o) should resolve correctly."""
    config = Config(
        openai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
        google_api_key="google-key",
    )
    from agents.llm.provider import _resolve_api_key_for_model

    assert _resolve_api_key_for_model("openai:gpt-4o", config) == "openai-key"
    assert _resolve_api_key_for_model("anthropic:claude-3-5-sonnet", config) == "anthropic-key"
    assert _resolve_api_key_for_model("google-genai:gemini-1.5-pro", config) == "google-key"
    # Unprefixed should still work
    assert _resolve_api_key_for_model("gpt-4o", config) == "openai-key"
    assert _resolve_api_key_for_model("claude-3-5-sonnet", config) == "anthropic-key"


def test_resolve_api_key_includes_google_in_fallback():
    """Fallback should include google_api_key."""
    config = Config(google_api_key="gkey")
    from agents.llm.provider import _resolve_api_key_for_model

    assert _resolve_api_key_for_model("some-unknown-model", config) == "gkey"


def test_coerce_content_string():
    """coerce_content should pass through plain strings."""
    assert coerce_content("hello") == "hello"


def test_coerce_content_list_of_blocks():
    """coerce_content should flatten a list of content blocks."""
    blocks = [{"text": "Hello "}, {"text": "world"}]
    assert coerce_content(blocks) == "Hello world"


def test_coerce_content_non_string():
    """coerce_content should str() fallback for unknown types."""
    assert coerce_content(42) == "42"
