"""Tests for shared chat model builder."""

import pytest
from unittest.mock import patch, MagicMock


class TestBuildChatModel:
    """Tests for build_chat_model factory function."""

    @patch("src.agents.llm.chat_model_factory.init_chat_model")
    def test_standard_model_uses_init_chat_model(self, mock_init):
        """Standard (non-LiteLLM) models should use init_chat_model."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_init.return_value = MagicMock()
        config = Config(anthropic_api_key="test-key")

        result = build_chat_model(
            config=config,
            model_name="anthropic:claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        mock_init.assert_called_once()
        assert result is not None

    @patch("langchain_litellm.ChatLiteLLM")
    def test_litellm_flag_uses_chat_litellm(self, mock_litellm_cls):
        """use_litellm=True should use ChatLiteLLM."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(anthropic_api_key="test-key")

        result = build_chat_model(
            config=config,
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-key",
            use_litellm=True,
        )

        mock_litellm_cls.assert_called_once()
        assert result is not None

    @patch("langchain_litellm.ChatLiteLLM")
    def test_base_url_triggers_litellm(self, mock_litellm_cls):
        """Providing base_url should auto-trigger LiteLLM path."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(anthropic_api_key="test-key")

        build_chat_model(
            config=config,
            model_name="gpt-4o",
            base_url="https://gateway.example.com",
            api_key="test-key",
        )

        mock_litellm_cls.assert_called_once()

    @patch("langchain_litellm.ChatLiteLLM")
    def test_litellm_provider_config_triggers_litellm(self, mock_litellm_cls):
        """config.llm_provider == 'litellm' should trigger LiteLLM path."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(llm_provider="litellm", anthropic_api_key="test-key")

        build_chat_model(
            config=config,
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        mock_litellm_cls.assert_called_once()

    @patch("langchain_litellm.ChatLiteLLM")
    def test_litellm_kwargs_include_config_values(self, mock_litellm_cls):
        """LiteLLM kwargs should include config values for retries, timeout, max_tokens."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_litellm_cls.return_value = MagicMock()
        config = Config(
            anthropic_api_key="test-key",
            max_retries=5,
            timeout=120,
            max_output_tokens=8192,
        )

        build_chat_model(
            config=config,
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-key",
            use_litellm=True,
        )

        call_kwargs = mock_litellm_cls.call_args[1]
        assert call_kwargs["max_retries"] == 5
        assert call_kwargs["request_timeout"] == 120
        assert call_kwargs["max_tokens"] == 8192

    @patch("src.agents.llm.chat_model_factory.init_chat_model")
    def test_init_chat_model_404_raises_runtime_error(self, mock_init):
        """init_chat_model 404 error should raise RuntimeError with LiteLLM suggestion."""
        from src.agents.llm.chat_model_factory import build_chat_model
        from src.agents.config import Config

        mock_init.side_effect = Exception("404 Not Found")
        config = Config(anthropic_api_key="test-key")

        with pytest.raises(RuntimeError, match="litellm"):
            build_chat_model(
                config=config,
                model_name="anthropic:claude-3-5-sonnet-20241022",
                api_key="test-key",
            )


class TestBuildTokenMiddleware:
    """Tests for shared middleware builder."""

    def test_build_token_middleware_returns_middleware(self):
        """Should return a TokenBudgetMiddleware instance."""
        from src.agents.llm.chat_model_factory import build_token_middleware
        from src.agents.config import Config
        from src.agents.middleware.token_budget import TokenBudgetMiddleware

        config = Config()
        mw = build_token_middleware(config, "claude-3-5-sonnet-20241022")

        assert isinstance(mw, TokenBudgetMiddleware)

    def test_build_token_middleware_uses_config_tpm(self):
        """Should use config.max_tpm for throttle setup."""
        from src.agents.llm.chat_model_factory import build_token_middleware
        from src.agents.config import Config

        config = Config(max_tpm=50000, tpm_safety_factor=0.9)
        mw = build_token_middleware(config, "test-model")

        # Middleware should be created without error
        assert mw is not None

    def test_build_token_middleware_accepts_shared_throttle(self):
        """Should use provided throttle instead of creating a new one."""
        from src.agents.llm.chat_model_factory import build_token_middleware
        from src.agents.llm.rate_limiting import TPMThrottle
        from src.agents.config import Config

        shared_throttle = TPMThrottle(max_tpm=50000, safety_factor=0.9)
        config = Config()
        mw = build_token_middleware(config, "test-model", throttle=shared_throttle)

        assert mw.throttle is shared_throttle

    def test_build_token_middleware_strips_provider_prefix(self):
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
