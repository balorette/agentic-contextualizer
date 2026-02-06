"""Resolve repository sources - local paths or GitHub URLs."""

import logging
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

GITHUB_URL_PATTERN = re.compile(
    r"^https?://github\.com/[\w.\-]+/[\w.\-]+(\.git)?/?$"
)


def is_github_url(source: str) -> bool:
    """Check if source string is a GitHub repository URL."""
    return bool(GITHUB_URL_PATTERN.match(source))


def extract_repo_name(url: str) -> str:
    """Extract repository name from a GitHub URL.

    Examples:
        https://github.com/owner/repo -> repo
        https://github.com/owner/repo.git -> repo
        https://github.com/owner/repo/ -> repo
    """
    # Strip trailing slash and .git suffix
    cleaned = url.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    # Last path segment is the repo name
    return cleaned.rsplit("/", 1)[-1]


def clone_repo(url: str, dest: str, timeout: int = 120) -> None:
    """Shallow clone a GitHub repository.

    Args:
        url: GitHub repository URL
        dest: Local directory to clone into
        timeout: Maximum seconds to wait for clone

    Raises:
        subprocess.CalledProcessError: If git clone fails
        subprocess.TimeoutExpired: If clone exceeds timeout
    """
    logger.info("Cloning %s into %s", url, dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    logger.info("Clone complete")


@contextmanager
def resolve_repo(source: str):
    """Resolve a source string to a local repository path.

    If source is a GitHub URL, clones it to a temp directory and yields that
    path. The temp directory is cleaned up when the context manager exits.

    If source is a local path, yields it directly with no cleanup.

    Args:
        source: GitHub URL or local filesystem path

    Yields:
        Path to the local repository

    Raises:
        ValueError: If source is not a valid local path or GitHub URL
        subprocess.CalledProcessError: If git clone fails
        subprocess.TimeoutExpired: If clone exceeds timeout
    """
    if is_github_url(source):
        tmp_dir = tempfile.mkdtemp(prefix="ctx-")
        try:
            clone_repo(source, tmp_dir)
            yield Path(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Cleaned up temp directory: %s", tmp_dir)
    else:
        path = Path(source)
        if not path.exists():
            raise ValueError(f"Path does not exist: {source}")
        yield path.resolve()
