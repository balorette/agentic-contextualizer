"""Protocol definition for file access backends."""

from typing import Protocol, Iterator, runtime_checkable


@runtime_checkable
class FileBackend(Protocol):
    """Protocol for file access backends.

    Backends are initialized with a repository root path. All file paths
    passed to methods are relative to that root.

    Implementations must handle:
    - Path traversal protection (prevent escaping repo boundary)
    - Encoding issues (UTF-8 with error handling)
    - File size limits
    - Symlink safety
    """

    @property
    def repo_path(self) -> str:
        """Root path of the repository this backend is bound to."""
        ...

    def read_file(self, path: str, max_size: int = 500_000) -> str | None:
        """Read file content.

        Args:
            path: Relative path within the repository
            max_size: Maximum file size in bytes (default 500KB)

        Returns:
            File content as string, or None if:
            - File doesn't exist
            - Path escapes repository boundary
            - File exceeds max_size
            - File cannot be read (permissions, encoding, etc.)
        """
        ...

    def file_exists(self, path: str) -> bool:
        """Check if a file exists and is readable.

        Args:
            path: Relative path within the repository

        Returns:
            True if file exists, is within repo boundary, and is a regular file
        """
        ...

    def walk_files(
        self,
        root: str = "",
        ignore_dirs: set[str] | None = None,
    ) -> Iterator[str]:
        """Iterate over all files in a directory tree.

        Args:
            root: Subdirectory to start from (relative to repo root)
            ignore_dirs: Directory names to skip (e.g., {'.git', 'node_modules'})

        Yields:
            Relative file paths within the repository
        """
        ...

    def search_content(
        self,
        pattern: str,
        file_extensions: set[str] | None = None,
        max_results: int = 50,
    ) -> list[str]:
        """Search for files containing a pattern.

        Args:
            pattern: Text pattern to search for (case-insensitive)
            file_extensions: Limit search to these extensions (e.g., {'.py', '.js'})
            max_results: Maximum number of results to return

        Returns:
            List of relative file paths containing the pattern
        """
        ...
