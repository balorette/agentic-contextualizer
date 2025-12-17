"""Backend abstractions for file access in scoped context generation.

This module provides pluggable backends for reading files, enabling:
- Local filesystem access (default)
- In-memory backends for testing
- Future: GitHub API, MCP filesystem, remote repos
"""

from .protocol import FileBackend
from .local import LocalFileBackend
from .memory import InMemoryFileBackend

__all__ = ["FileBackend", "LocalFileBackend", "InMemoryFileBackend"]
