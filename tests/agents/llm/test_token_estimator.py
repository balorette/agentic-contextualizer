"""Tests for token estimation."""

from agents.llm.token_estimator import LiteLLMTokenEstimator


class TestLiteLLMTokenEstimator:
    """Tests for the LiteLLM-based token estimator."""

    def test_estimate_returns_positive_int(self, mocker):
        """Estimation should return a positive integer."""
        mocker.patch("litellm.token_counter", return_value=150)
        estimator = LiteLLMTokenEstimator()
        result = estimator.estimate(
            [{"role": "user", "content": "Hello world"}],
            model="claude-3-5-sonnet-20241022",
        )
        assert isinstance(result, int)
        assert result > 0

    def test_estimate_passes_model_and_messages(self, mocker):
        """Should forward model and messages to litellm.token_counter."""
        mock_counter = mocker.patch("litellm.token_counter", return_value=42)
        estimator = LiteLLMTokenEstimator()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        estimator.estimate(messages, model="gpt-4o")
        mock_counter.assert_called_once_with(model="gpt-4o", messages=messages)

    def test_estimate_handles_litellm_error_gracefully(self, mocker):
        """If litellm.token_counter raises, fall back to char-based estimate."""
        mocker.patch("litellm.token_counter", side_effect=Exception("tokenizer not found"))
        estimator = LiteLLMTokenEstimator()
        result = estimator.estimate(
            [{"role": "user", "content": "Hello world, this is a test."}],
            model="unknown-model",
        )
        # Fallback: ~4 chars per token
        assert result > 0
