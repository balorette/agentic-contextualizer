"""In-memory file backend for testing."""

from pathlib import PurePosixPath
from typing import Iterator


class InMemoryFileBackend:
    """In-memory file backend for testing.

    Provides a simple dict-based file store that implements the FileBackend
    protocol without hitting the real filesystem.

    Example:
        backend = InMemoryFileBackend("/fake/repo", {
            "src/main.py": "def main(): pass",
            "src/utils.py": "def helper(): pass",
            "README.md": "# My Project",
        })
        content = backend.read_file("src/main.py")
    """

    def __init__(
        self,
        repo_path: str = "/fake/repo",
        files: dict[str, str] | None = None,
    ):
        """Initialize with optional file contents.

        Args:
            repo_path: Fake repository path (for protocol compliance)
            files: Dictionary mapping relative paths to file contents
        """
        self._repo_path = repo_path
        self._files: dict[str, str] = files or {}

    @property
    def repo_path(self) -> str:
        """Root path of the repository this backend is bound to."""
        return self._repo_path

    def add_file(self, path: str, content: str) -> None:
        """Add or update a file in the fake filesystem.

        Args:
            path: Relative path within repository
            content: File content
        """
        # Normalize path
        normalized = str(PurePosixPath(path))
        self._files[normalized] = content

    def remove_file(self, path: str) -> None:
        """Remove a file from the fake filesystem.

        Args:
            path: Relative path within repository
        """
        normalized = str(PurePosixPath(path))
        self._files.pop(normalized, None)

    def _normalize_path(self, path: str) -> str:
        """Normalize path for lookup."""
        return str(PurePosixPath(path))

    def _is_safe_path(self, path: str) -> bool:
        """Check if path is safe (doesn't escape repo via ..)."""
        # Simple check: no .. that would escape
        parts = PurePosixPath(path).parts
        depth = 0
        for part in parts:
            if part == "..":
                depth -= 1
                if depth < 0:
                    return False
            elif part != ".":
                depth += 1
        return True

    def read_file(self, path: str, max_size: int = 500_000) -> str | None:
        """Read file content from in-memory store.

        Args:
            path: Relative path within the repository
            max_size: Maximum file size in bytes (for protocol compliance)

        Returns:
            File content as string, or None if not found
        """
        if not self._is_safe_path(path):
            return None

        normalized = self._normalize_path(path)
        content = self._files.get(normalized)

        if content is not None and len(content.encode()) > max_size:
            return None

        return content

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the in-memory store.

        Args:
            path: Relative path within the repository

        Returns:
            True if file exists in the store
        """
        if not self._is_safe_path(path):
            return False

        normalized = self._normalize_path(path)
        return normalized in self._files

    def walk_files(
        self,
        root: str = "",
        ignore_dirs: set[str] | None = None,
    ) -> Iterator[str]:
        """Iterate over all files in the in-memory store.

        Args:
            root: Subdirectory to start from
            ignore_dirs: Directory names to skip

        Yields:
            Relative file paths within the repository
        """
        ignored = ignore_dirs or set()
        root_prefix = self._normalize_path(root) if root else ""

        for path in sorted(self._files.keys()):
            # Check if path is under root
            if root_prefix and not path.startswith(root_prefix + "/") and path != root_prefix:
                continue

            # Check for ignored directories
            parts = PurePosixPath(path).parts
            if any(part in ignored for part in parts[:-1]):  # Don't check filename
                continue

            yield path

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
        pattern_lower = pattern.lower()
        results: list[str] = []

        for path in self.walk_files():
            if len(results) >= max_results:
                break

            # Check extension if specified
            if file_extensions:
                suffix = PurePosixPath(path).suffix.lower()
                if suffix not in file_extensions:
                    continue

            # Search content
            content = self._files.get(path, "")
            if pattern_lower in content.lower():
                results.append(path)

        return results
