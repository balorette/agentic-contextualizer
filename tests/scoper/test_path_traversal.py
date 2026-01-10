"""Tests for path traversal protection in file backends."""

import pytest
from src.agents.tools import LocalFileBackend


class TestLocalFileBackendPathTraversal:
    """Test that path traversal attacks are prevented in LocalFileBackend."""

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
        backend = LocalFileBackend(sample_repo)

        # Attempt to read file outside repo via parent traversal
        result = backend.read_file("../secret.txt")

        assert result is None, "Should block parent directory traversal"

    def test_read_file_blocks_absolute_path(self, sample_repo, sensitive_file):
        """Test that absolute paths outside repo are blocked."""
        backend = LocalFileBackend(sample_repo)

        # Attempt to read via absolute path
        result = backend.read_file(str(sensitive_file))

        assert result is None, "Should block absolute paths outside repo"

    def test_read_file_allows_valid_paths(self, sample_repo):
        """Test that valid paths within repo still work."""
        backend = LocalFileBackend(sample_repo)

        result = backend.read_file("src/safe.py")

        assert result == "safe content", "Should allow valid repo paths"

    def test_read_file_blocks_symlink_escape(self, sample_repo, tmp_path):
        """Test that symlinks pointing outside repo are blocked."""
        # Create symlink pointing outside repo
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive via symlink")
        symlink = sample_repo / "src" / "sneaky_link"
        symlink.symlink_to(secret)

        backend = LocalFileBackend(sample_repo)

        result = backend.read_file("src/sneaky_link")

        assert result is None, "Should block symlinks escaping repo"

    def test_file_exists_blocks_traversal(self, sample_repo, sensitive_file):
        """Test that file_exists also blocks traversal."""
        backend = LocalFileBackend(sample_repo)

        assert backend.file_exists("../secret.txt") is False
        assert backend.file_exists(str(sensitive_file)) is False
        assert backend.file_exists("src/safe.py") is True

    def test_read_file_respects_max_size(self, sample_repo):
        """Test that files exceeding max_size are rejected."""
        large_file = sample_repo / "large.txt"
        large_file.write_text("x" * 1000)

        backend = LocalFileBackend(sample_repo)

        # Should read with default max_size
        assert backend.read_file("large.txt") is not None

        # Should reject when max_size is smaller
        assert backend.read_file("large.txt", max_size=100) is None
