"""Tests for path traversal protection in scoped analyzer."""

import pytest
from pathlib import Path
from src.agents.scoper.scoped_analyzer import ScopedAnalyzer
from src.agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        return LLMResponse(content="{}", model="mock")

    def generate_structured(self, prompt: str, system: str | None, schema):
        return schema(
            additional_files_needed=[],
            reasoning="Done",
            sufficient_context=True,
            preliminary_insights="Test",
        )


class TestPathTraversalProtection:
    """Test that path traversal attacks are prevented."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "safe.py").write_text("safe content")
        return repo

    @pytest.fixture
    def sensitive_file(self, tmp_path):
        """Create a sensitive file outside repo."""
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive data")
        return secret

    def test_read_file_blocks_parent_traversal(self, sample_repo, sensitive_file):
        """Test that ../path traversal is blocked."""
        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        # Attempt to read file outside repo via parent traversal
        result = analyzer._read_file(sample_repo, "../secret.txt")

        assert result is None, "Should block parent directory traversal"

    def test_read_file_blocks_absolute_path(self, sample_repo, sensitive_file):
        """Test that absolute paths outside repo are blocked."""
        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        # Attempt to read via absolute path
        result = analyzer._read_file(sample_repo, str(sensitive_file))

        assert result is None, "Should block absolute paths outside repo"

    def test_read_file_allows_valid_paths(self, sample_repo):
        """Test that valid paths within repo still work."""
        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        result = analyzer._read_file(sample_repo, "src/safe.py")

        assert result == "safe content", "Should allow valid repo paths"

    def test_read_file_blocks_symlink_escape(self, sample_repo, tmp_path):
        """Test that symlinks pointing outside repo are blocked."""
        # Create symlink pointing outside repo
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive via symlink")
        symlink = sample_repo / "src" / "sneaky_link"
        symlink.symlink_to(secret)

        mock_llm = MockLLMProvider()
        analyzer = ScopedAnalyzer(mock_llm)

        result = analyzer._read_file(sample_repo, "src/sneaky_link")

        assert result is None, "Should block symlinks escaping repo"
