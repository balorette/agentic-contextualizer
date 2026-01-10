"""Backend abstractions for file access.

This module provides pluggable backends for reading files, enabling:
- Local filesystem access (default)
- In-memory backends for testing
- Future: GitHub API, MCP filesystem, remote repos
"""

from .protocol import FileBackend
from .local import LocalFileBackend, DEFAULT_IGNORED_DIRS, DEFAULT_SEARCHABLE_EXTENSIONS
from .memory import InMemoryFileBackend

__all__ = [
    "FileBackend",
    "LocalFileBackend",
    "InMemoryFileBackend",
    "DEFAULT_IGNORED_DIRS",
    "DEFAULT_SEARCHABLE_EXTENSIONS",
]
