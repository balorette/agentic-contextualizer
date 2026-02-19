"""Tests for observability / tracing configuration."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from agents.observability import (
    configure_langfuse_tracing,
    get_langfuse_callback_handler,
    is_langfuse_tracing_enabled,
)


# ---------------------------------------------------------------------------
# is_langfuse_tracing_enabled
# ---------------------------------------------------------------------------


def test_langfuse_enabled_when_both_keys_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert is_langfuse_tracing_enabled() is True


def test_langfuse_disabled_when_public_key_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert is_langfuse_tracing_enabled() is False


def test_langfuse_disabled_when_secret_key_missing(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert is_langfuse_tracing_enabled() is False


def test_langfuse_disabled_when_no_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert is_langfuse_tracing_enabled() is False


# ---------------------------------------------------------------------------
# configure_langfuse_tracing
# ---------------------------------------------------------------------------


def test_configure_langfuse_no_keys_logs_info(monkeypatch, caplog):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with caplog.at_level(logging.INFO):
        configure_langfuse_tracing()

    assert "Langfuse keys not found" in caplog.text


@patch("agents.observability.Langfuse", create=True)
def test_configure_langfuse_success(mock_langfuse_cls, monkeypatch, caplog):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    # Patch the import inside the function
    with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
        with caplog.at_level(logging.INFO):
            configure_langfuse_tracing(project_name="test-project")

    assert "Langfuse tracing enabled" in caplog.text


def test_configure_langfuse_exception_logs_warning(monkeypatch, caplog):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    with patch("agents.observability.Langfuse", side_effect=RuntimeError("connection failed"), create=True):
        import agents.observability as obs
        # Patch the import within the function body
        with patch.object(obs, "Langfuse", side_effect=RuntimeError("connection failed"), create=True):
            pass

    # The function does a local import, so we patch at the module level via sys.modules
    mock_langfuse_mod = MagicMock()
    mock_langfuse_mod.Langfuse.side_effect = RuntimeError("connection failed")

    with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
        with caplog.at_level(logging.WARNING):
            configure_langfuse_tracing()

    assert "Langfuse tracing setup failed" in caplog.text


# ---------------------------------------------------------------------------
# get_langfuse_callback_handler
# ---------------------------------------------------------------------------


def test_get_langfuse_handler_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert get_langfuse_callback_handler() is None


def test_get_langfuse_handler_returns_handler(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_handler = MagicMock()
    mock_cb_module = MagicMock()
    mock_cb_module.CallbackHandler.return_value = mock_handler

    mock_langfuse_pkg = MagicMock()
    mock_langfuse_pkg.langchain = mock_cb_module

    with patch.dict("sys.modules", {
        "langfuse": mock_langfuse_pkg,
        "langfuse.langchain": mock_cb_module,
    }):
        result = get_langfuse_callback_handler()

    assert result is mock_handler


def test_get_langfuse_handler_logs_warning_on_error(monkeypatch, caplog):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_cb_module = MagicMock()
    mock_cb_module.CallbackHandler.side_effect = RuntimeError("import boom")

    with patch.dict("sys.modules", {
        "langfuse": MagicMock(),
        "langfuse.langchain": mock_cb_module,
    }):
        with caplog.at_level(logging.WARNING):
            result = get_langfuse_callback_handler()

    assert result is None
    assert "Langfuse CallbackHandler setup failed" in caplog.text
