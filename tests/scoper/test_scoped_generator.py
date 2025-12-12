"""Tests for scoped context generator."""

import pytest
from pathlib import Path
from agents.scoper.scoped_generator import ScopedGenerator
from agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return mock generated context."""
        return LLMResponse(
            content="""## Summary

The weather module provides forecast functionality.

## API Endpoints

- GET /weather/forecast

## Key Files

- src/weather/service.py
""",
            model="mock-model",
        )


class TestScopedGenerator:
    """Test ScopedGenerator."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory."""
        out = tmp_path / "contexts"
        out.mkdir()
        return out

    def test_generate_creates_file(self, output_dir):
        """Test that generate creates a scoped context file."""
        mock_llm = MockLLMProvider()
        generator = ScopedGenerator(mock_llm, output_dir)

        output_path = generator.generate(
            repo_name="test-repo",
            question="weather functionality",
            relevant_files={"src/weather.py": "def get_weather(): pass"},
            insights="Weather module provides forecasts",
            model_name="claude-3-5-sonnet",
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "weather" in content.lower()

    def test_generate_includes_frontmatter(self, output_dir):
        """Test that generated file includes YAML frontmatter."""
        mock_llm = MockLLMProvider()
        generator = ScopedGenerator(mock_llm, output_dir)

        output_path = generator.generate(
            repo_name="test-repo",
            question="weather functionality",
            relevant_files={"src/weather.py": "def get_weather(): pass"},
            insights="Weather module",
            model_name="claude-3-5-sonnet",
        )

        content = output_path.read_text()
        assert content.startswith("---")
        assert "scope_question:" in content
        assert "weather functionality" in content
        assert "files_analyzed:" in content

    def test_generate_uses_sanitized_filename(self, output_dir):
        """Test that output filename is sanitized from question."""
        mock_llm = MockLLMProvider()
        generator = ScopedGenerator(mock_llm, output_dir)

        output_path = generator.generate(
            repo_name="test-repo",
            question="How does the auth/login flow work?",
            relevant_files={"src/auth.py": "def login(): pass"},
            insights="Auth flow",
            model_name="claude-3-5-sonnet",
        )

        # Filename should be sanitized (no special chars)
        assert "scope-" in output_path.name
        assert "/" not in output_path.name
        assert "?" not in output_path.name
