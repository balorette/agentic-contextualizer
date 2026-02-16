"""Protocol definition for semantic file analysis backends.

Backends analyze source code to extract symbols, outlines, and references.
Two implementations are planned:
- ASTFileAnalysisBackend (default): stdlib ast for Python, tree-sitter for JS/TS
- LSPFileAnalysisBackend (future): Language Server Protocol for rich semantic info
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import FileOutline, SymbolDetail, Reference


@runtime_checkable
class FileAnalysisBackend(Protocol):
    """Protocol for semantic code analysis backends.

    Methods receive source code as a string parameter rather than
    reading files directly — this keeps the backend testable and
    decoupled from I/O concerns. SmartFileAccess handles file reading.
    """

    def get_outline(self, file_path: str, source: str) -> FileOutline:
        """Extract file outline: imports and symbol signatures.

        Args:
            file_path: Used to detect language from extension.
            source: File content as string.

        Returns:
            FileOutline with imports and symbols (no bodies).
        """
        ...

    def read_symbol(self, file_path: str, symbol_name: str, source: str) -> SymbolDetail | None:
        """Extract a specific symbol's full body from source.

        Args:
            file_path: Used to detect language from extension.
            symbol_name: Name of function/class/method to extract.
            source: File content as string.

        Returns:
            SymbolDetail with body, or None if symbol not found.
        """
        ...

    def find_references(
        self, symbol_name: str, file_backend, scope: str | None = None
    ) -> list[Reference]:
        """Find references to a symbol across the codebase.

        Args:
            symbol_name: Symbol name to search for.
            file_backend: FileBackend for walking/reading files.
            scope: Optional directory scope to limit search.

        Returns:
            List of References with file path, line, and context.
        """
        ...
