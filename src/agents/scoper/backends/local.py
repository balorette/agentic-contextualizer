"""Local filesystem backend for scoped context generation."""

import os
from pathlib import Path
from typing import Iterator

# Directories to always ignore when walking
DEFAULT_IGNORED_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "egg-info", ".egg-info", ".tox", ".nox",
}

# File extensions that can be searched for content
DEFAULT_SEARCHABLE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".md", ".txt", ".yaml", ".yml", ".json",
    ".toml", ".ini", ".cfg", ".conf",
}


class LocalFileBackend:
    """Local filesystem backend with path traversal protection.

    All paths are relative to the repository root. The backend ensures
    that file access cannot escape the repository boundary.
    """

    def __init__(
        self,
        repo_path: str | Path,
        ignored_dirs: set[str] | None = None,
        searchable_extensions: set[str] | None = None,
    ):
        """Initialize backend bound to a repository root.

        Args:
            repo_path: Absolute path to repository root
            ignored_dirs: Directory names to skip (defaults to common ignores)
            searchable_extensions: Extensions for content search (defaults to code files)
        """
        self._repo_path = Path(repo_path).resolve()
        self._ignored_dirs = ignored_dirs or DEFAULT_IGNORED_DIRS
        self._searchable_extensions = searchable_extensions or DEFAULT_SEARCHABLE_EXTENSIONS

        if not self._repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {repo_path}")

    @property
    def repo_path(self) -> str:
        """Root path of the repository this backend is bound to."""
        return str(self._repo_path)

    def _resolve_safe_path(self, path: str) -> Path | None:
        """Resolve a relative path safely within repo boundary.

        Args:
            path: Relative path within repository

        Returns:
            Resolved absolute Path, or None if path escapes boundary or is invalid
        """
        try:
            # Resolve to absolute path
            full_path = (self._repo_path / path).resolve()

            # Verify the resolved path is within the repository
            try:
                full_path.relative_to(self._repo_path)
            except ValueError:
                # Path escapes repository boundary
                return None

            # Check symlink targets stay in repo
            if full_path.is_symlink():
                real_path = full_path.resolve()
                try:
                    real_path.relative_to(self._repo_path)
                except ValueError:
                    # Symlink target escapes boundary
                    return None

            return full_path

        except OSError:
            # Invalid path
            return None

    def read_file(self, path: str, max_size: int = 500_000) -> str | None:
        """Read file content with path traversal protection.

        Args:
            path: Relative path within the repository
            max_size: Maximum file size in bytes (default 500KB)

        Returns:
            File content as string, or None if file cannot be read safely
        """
        full_path = self._resolve_safe_path(path)
        if full_path is None:
            return None

        try:
            if not full_path.exists() or not full_path.is_file():
                return None

            if full_path.stat().st_size > max_size:
                return None

            return full_path.read_text(encoding="utf-8", errors="ignore")

        except OSError:
            # File access errors (permissions, broken symlinks, etc.)
            return None

    def file_exists(self, path: str) -> bool:
        """Check if a file exists and is readable within repo boundary.

        Args:
            path: Relative path within the repository

        Returns:
            True if file exists, is within repo boundary, and is a regular file
        """
        full_path = self._resolve_safe_path(path)
        if full_path is None:
            return False

        try:
            return full_path.exists() and full_path.is_file()
        except OSError:
            return False

    def walk_files(
        self,
        root: str = "",
        ignore_dirs: set[str] | None = None,
    ) -> Iterator[str]:
        """Iterate over all files in a directory tree.

        Args:
            root: Subdirectory to start from (relative to repo root)
            ignore_dirs: Additional directory names to skip

        Yields:
            Relative file paths within the repository
        """
        ignored = self._ignored_dirs | (ignore_dirs or set())

        start_path = self._resolve_safe_path(root) if root else self._repo_path
        if start_path is None or not start_path.is_dir():
            return

        for dirpath, dirnames, filenames in os.walk(start_path):
            # Prune ignored directories (modifying in-place)
            dirnames[:] = [
                d for d in dirnames
                if d not in ignored and not d.endswith(".egg-info")
            ]

            for filename in filenames:
                file_path = Path(dirpath) / filename
                try:
                    rel_path = file_path.relative_to(self._repo_path)
                    yield str(rel_path)
                except ValueError:
                    # Shouldn't happen, but skip if path escapes
                    continue

    def search_content(
        self,
        pattern: str,
        file_extensions: set[str] | None = None,
        max_results: int = 50,
    ) -> list[str]:
        """Search for files containing a pattern.

        Args:
            pattern: Text pattern to search for (case-insensitive)
            file_extensions: Limit search to these extensions
            max_results: Maximum number of results to return

        Returns:
            List of relative file paths containing the pattern
        """
        extensions = file_extensions or self._searchable_extensions
        pattern_lower = pattern.lower()
        results: list[str] = []

        for rel_path in self.walk_files():
            if len(results) >= max_results:
                break

            # Check extension
            suffix = Path(rel_path).suffix.lower()
            if suffix not in extensions:
                continue

            # Read and search content
            content = self.read_file(rel_path)
            if content and pattern_lower in content.lower():
                results.append(rel_path)

        return results
