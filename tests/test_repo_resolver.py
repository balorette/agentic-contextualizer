"""Tests for repo_resolver - GitHub URL detection, cloning, and cleanup."""

import shutil
import subprocess
import pytest
from unittest.mock import patch

from agents.repo_resolver import (
    is_github_url,
    extract_repo_name,
    clone_repo,
    resolve_repo,
)


class TestIsGithubUrl:
    """Test GitHub URL detection."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo.git",
            "https://github.com/owner/repo/",
            "http://github.com/owner/repo",
            "https://github.com/some-org/my-repo.js",
            "https://github.com/user123/repo_name",
            "https://github.com/pallets/flask",
        ],
    )
    def test_valid_github_urls(self, url):
        assert is_github_url(url) is True

    @pytest.mark.parametrize(
        "source",
        [
            "/local/path/to/repo",
            "./relative/path",
            "https://gitlab.com/owner/repo",
            "https://bitbucket.org/owner/repo",
            "https://github.com/owner",  # missing repo
            "https://github.com/",
            "github.com/owner/repo",  # missing scheme
            "not-a-url",
            "",
            "https://github.com/owner/repo/tree/main",  # subpath
        ],
    )
    def test_non_github_urls(self, source):
        assert is_github_url(source) is False


class TestExtractRepoName:
    """Test repo name extraction from URLs."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/owner/repo", "repo"),
            ("https://github.com/owner/repo.git", "repo"),
            ("https://github.com/owner/repo/", "repo"),
            ("https://github.com/owner/repo.git/", "repo"),
            ("https://github.com/pallets/flask", "flask"),
            ("https://github.com/some-org/my-project", "my-project"),
        ],
    )
    def test_extracts_name(self, url, expected):
        assert extract_repo_name(url) == expected


class TestCloneRepo:
    """Test git clone execution."""

    @patch("agents.repo_resolver.subprocess.run")
    def test_calls_git_clone_shallow(self, mock_run):
        """Verify git clone --depth 1 is called with correct args."""
        clone_repo("https://github.com/owner/repo", "/tmp/dest")

        mock_run.assert_called_once_with(
            ["git", "clone", "--depth", "1", "https://github.com/owner/repo", "/tmp/dest"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @patch("agents.repo_resolver.subprocess.run")
    def test_custom_timeout(self, mock_run):
        """Verify custom timeout is passed through."""
        clone_repo("https://github.com/owner/repo", "/tmp/dest", timeout=60)

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == 60

    @patch("agents.repo_resolver.subprocess.run")
    def test_clone_failure_raises(self, mock_run):
        """CalledProcessError propagates on clone failure."""
        mock_run.side_effect = subprocess.CalledProcessError(
            128, "git", stderr="fatal: repository not found"
        )

        with pytest.raises(subprocess.CalledProcessError):
            clone_repo("https://github.com/owner/nonexistent", "/tmp/dest")

    @patch("agents.repo_resolver.subprocess.run")
    def test_clone_timeout_raises(self, mock_run):
        """TimeoutExpired propagates when clone hangs."""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 120)

        with pytest.raises(subprocess.TimeoutExpired):
            clone_repo("https://github.com/owner/repo", "/tmp/dest")


class TestResolveRepo:
    """Test the resolve_repo context manager."""

    def test_local_path_passes_through(self, tmp_path):
        """Local paths are yielded directly with no cloning."""
        repo = tmp_path / "my_repo"
        repo.mkdir()

        with resolve_repo(str(repo)) as path:
            assert path == repo.resolve()
            assert path.exists()

    def test_local_path_not_found_raises(self):
        """Non-existent local path raises ValueError."""
        with pytest.raises(ValueError, match="Path does not exist"):
            with resolve_repo("/nonexistent/path"):
                pass

    @patch("agents.repo_resolver.clone_repo")
    @patch("agents.repo_resolver.shutil.rmtree", wraps=shutil.rmtree)
    def test_github_url_clones_and_cleans_up(self, mock_rmtree, mock_clone):
        """GitHub URL triggers clone, yields path, cleans up after."""
        with resolve_repo("https://github.com/owner/repo") as path:
            assert path.name == "repo"
            assert "ctx-" in str(path.parent)
            mock_clone.assert_called_once()

        # Cleanup called after context manager exits
        mock_rmtree.assert_called_once()
        assert mock_rmtree.call_args[1] == {"ignore_errors": True}

    @patch("agents.repo_resolver.clone_repo")
    @patch("agents.repo_resolver.shutil.rmtree", wraps=shutil.rmtree)
    def test_cleanup_on_pipeline_error(self, mock_rmtree, mock_clone):
        """Temp directory is cleaned up even if the pipeline raises."""
        with pytest.raises(RuntimeError, match="pipeline broke"):
            with resolve_repo("https://github.com/owner/repo") as path:
                raise RuntimeError("pipeline broke")

        # Cleanup still happens
        mock_rmtree.assert_called_once()

    @patch("agents.repo_resolver.clone_repo")
    @patch("agents.repo_resolver.shutil.rmtree", wraps=shutil.rmtree)
    def test_cleanup_on_clone_error(self, mock_rmtree, mock_clone):
        """Temp directory is cleaned up even if clone itself fails."""
        mock_clone.side_effect = subprocess.CalledProcessError(
            128, "git", stderr="fatal: repo not found"
        )

        with pytest.raises(subprocess.CalledProcessError):
            with resolve_repo("https://github.com/owner/repo"):
                pass

        # Cleanup still called
        mock_rmtree.assert_called_once()
