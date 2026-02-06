"""Tests for scoper file backends."""

import pytest
from src.agents.tools import (
    FileBackend,
    LocalFileBackend,
    InMemoryFileBackend,
)


class TestLocalFileBackend:
    """Tests for LocalFileBackend."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository structure."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Create directory structure
        (repo / "src").mkdir()
        (repo / "src" / "main.py").write_text("def main(): pass")
        (repo / "src" / "utils.py").write_text("def helper(): pass")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_main.py").write_text("def test_main(): assert True")
        (repo / "README.md").write_text("# Project\nAuthentication system")
        (repo / ".git").mkdir()
        (repo / ".git" / "config").write_text("[core]")

        return repo

    def test_repo_path_property(self, sample_repo):
        """Test repo_path returns the bound path."""
        backend = LocalFileBackend(sample_repo)
        assert backend.repo_path == str(sample_repo.resolve())

    def test_read_file_success(self, sample_repo):
        """Test reading a valid file."""
        backend = LocalFileBackend(sample_repo)
        content = backend.read_file("src/main.py")
        assert content == "def main(): pass"

    def test_read_file_nonexistent(self, sample_repo):
        """Test reading nonexistent file returns None."""
        backend = LocalFileBackend(sample_repo)
        assert backend.read_file("nonexistent.py") is None

    def test_file_exists(self, sample_repo):
        """Test file_exists method."""
        backend = LocalFileBackend(sample_repo)
        assert backend.file_exists("src/main.py") is True
        assert backend.file_exists("nonexistent.py") is False
        assert backend.file_exists("src") is False  # Directory, not file

    def test_walk_files_basic(self, sample_repo):
        """Test walking all files."""
        backend = LocalFileBackend(sample_repo)
        files = list(backend.walk_files())

        assert "src/main.py" in files
        assert "src/utils.py" in files
        assert "tests/test_main.py" in files
        assert "README.md" in files

    def test_walk_files_ignores_git(self, sample_repo):
        """Test that .git is ignored by default."""
        backend = LocalFileBackend(sample_repo)
        files = list(backend.walk_files())

        assert not any(".git" in f for f in files)

    def test_walk_files_custom_root(self, sample_repo):
        """Test walking from a subdirectory."""
        backend = LocalFileBackend(sample_repo)
        files = list(backend.walk_files(root="src"))

        assert "src/main.py" in files
        assert "src/utils.py" in files
        assert "README.md" not in files

    def test_search_content(self, sample_repo):
        """Test searching file contents."""
        backend = LocalFileBackend(sample_repo)
        results = backend.search_content("authentication")

        assert "README.md" in results

    def test_search_content_case_insensitive(self, sample_repo):
        """Test that search is case-insensitive."""
        backend = LocalFileBackend(sample_repo)
        results = backend.search_content("AUTHENTICATION")

        assert "README.md" in results

    def test_search_content_with_extension_filter(self, sample_repo):
        """Test searching with extension filter."""
        backend = LocalFileBackend(sample_repo)

        # Only search Python files
        results = backend.search_content("def", file_extensions={".py"})

        assert "src/main.py" in results
        assert "README.md" not in results

    def test_invalid_repo_path_raises(self, tmp_path):
        """Test that invalid repo path raises ValueError."""
        with pytest.raises(ValueError):
            LocalFileBackend(tmp_path / "nonexistent")

    def test_protocol_compliance(self, sample_repo):
        """Test that LocalFileBackend satisfies FileBackend protocol."""
        backend = LocalFileBackend(sample_repo)
        assert isinstance(backend, FileBackend)


class TestInMemoryFileBackend:
    """Tests for InMemoryFileBackend."""

    def test_basic_read_write(self):
        """Test basic file operations."""
        backend = InMemoryFileBackend(files={
            "src/main.py": "def main(): pass",
        })

        assert backend.read_file("src/main.py") == "def main(): pass"

    def test_add_file(self):
        """Test adding files after initialization."""
        backend = InMemoryFileBackend()
        backend.add_file("new.py", "content")

        assert backend.read_file("new.py") == "content"

    def test_remove_file(self):
        """Test removing files."""
        backend = InMemoryFileBackend(files={"test.py": "content"})
        backend.remove_file("test.py")

        assert backend.read_file("test.py") is None

    def test_file_exists(self):
        """Test file_exists method."""
        backend = InMemoryFileBackend(files={"exists.py": "content"})

        assert backend.file_exists("exists.py") is True
        assert backend.file_exists("missing.py") is False

    def test_walk_files(self):
        """Test iterating over all files."""
        backend = InMemoryFileBackend(files={
            "src/main.py": "main",
            "src/utils.py": "utils",
            "README.md": "readme",
        })

        files = list(backend.walk_files())

        assert "src/main.py" in files
        assert "src/utils.py" in files
        assert "README.md" in files

    def test_walk_files_with_root(self):
        """Test walking from subdirectory."""
        backend = InMemoryFileBackend(files={
            "src/main.py": "main",
            "tests/test.py": "test",
        })

        files = list(backend.walk_files(root="src"))

        assert "src/main.py" in files
        assert "tests/test.py" not in files

    def test_walk_files_ignores_dirs(self):
        """Test ignoring directories."""
        backend = InMemoryFileBackend(files={
            "src/main.py": "main",
            "node_modules/pkg/index.js": "pkg",
        })

        files = list(backend.walk_files(ignore_dirs={"node_modules"}))

        assert "src/main.py" in files
        assert "node_modules/pkg/index.js" not in files

    def test_search_content(self):
        """Test content search."""
        backend = InMemoryFileBackend(files={
            "auth.py": "def authenticate(): pass",
            "main.py": "def main(): pass",
        })

        results = backend.search_content("authenticate")

        assert "auth.py" in results
        assert "main.py" not in results

    def test_blocks_path_traversal(self):
        """Test that path traversal is blocked."""
        backend = InMemoryFileBackend(files={
            "src/safe.py": "safe",
        })

        assert backend.read_file("../escape.py") is None
        assert backend.file_exists("../../etc/passwd") is False

    def test_repo_path_property(self):
        """Test repo_path returns configured path."""
        backend = InMemoryFileBackend(repo_path="/custom/path")
        assert backend.repo_path == "/custom/path"

    def test_max_size_limit(self):
        """Test that max_size is respected."""
        backend = InMemoryFileBackend(files={
            "large.txt": "x" * 1000,
        })

        assert backend.read_file("large.txt", max_size=100) is None
        assert backend.read_file("large.txt", max_size=2000) is not None

    def test_protocol_compliance(self):
        """Test that InMemoryFileBackend satisfies FileBackend protocol."""
        backend = InMemoryFileBackend()
        assert isinstance(backend, FileBackend)
