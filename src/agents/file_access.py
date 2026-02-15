"""SmartFileAccess — unified layer for progressive file analysis.

Composes a FileBackend (raw I/O) with a FileAnalysisBackend (semantic ops).
Optionally accepts an LSP backend that takes priority with AST fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .backends import FileOutline, SymbolDetail, Reference

if TYPE_CHECKING:
    from .backends.protocol import FileAnalysisBackend
    from .tools.backends.protocol import FileBackend

logger = logging.getLogger(__name__)


class SmartFileAccess:
    """Unified file access with progressive disclosure.

    Usage:
        backend = LocalFileBackend(repo_path)
        analysis = ASTFileAnalysisBackend()
        smart = SmartFileAccess(backend, analysis)

        outline = smart.get_outline("src/main.py")   # ~500 bytes
        detail = smart.read_symbol("src/main.py", "authenticate")  # ~1-2 KB
        lines = smart.read_lines("src/main.py", 10, 20)  # surgical
        content = smart.read_file("src/main.py")  # last resort
    """

    def __init__(
        self,
        file_backend: FileBackend,
        analysis_backend: FileAnalysisBackend,
        lsp_backend: FileAnalysisBackend | None = None,
    ) -> None:
        self._files = file_backend
        self._analysis = analysis_backend
        self._lsp = lsp_backend

    @property
    def file_backend(self) -> FileBackend:
        return self._files

    def get_outline(self, file_path: str) -> FileOutline | None:
        """Get file outline: imports and symbol signatures."""
        source = self._files.read_file(file_path)
        if source is None:
            return None

        if self._lsp:
            try:
                return self._lsp.get_outline(file_path, source)
            except Exception:
                logger.debug("LSP get_outline failed for %s, falling back to AST", file_path)

        return self._analysis.get_outline(file_path, source)

    def read_symbol(self, file_path: str, symbol_name: str) -> SymbolDetail | None:
        """Extract a specific symbol's full body."""
        source = self._files.read_file(file_path)
        if source is None:
            return None

        if self._lsp:
            try:
                result = self._lsp.read_symbol(file_path, symbol_name, source)
                if result is not None:
                    return result
            except Exception:
                logger.debug("LSP read_symbol failed for %s:%s, falling back to AST", file_path, symbol_name)

        return self._analysis.read_symbol(file_path, symbol_name, source)

    def find_references(self, symbol_name: str, scope: str | None = None) -> list[Reference]:
        """Find references to a symbol across the codebase."""
        if self._lsp:
            try:
                return self._lsp.find_references(symbol_name, self._files, scope)
            except Exception:
                logger.debug("LSP find_references failed for %s, falling back to AST", symbol_name)

        return self._analysis.find_references(symbol_name, self._files, scope)

    def read_lines(self, file_path: str, start: int, end: int) -> str | None:
        """Read specific line range from a file (1-indexed, inclusive)."""
        source = self._files.read_file(file_path)
        if source is None:
            return None

        lines = source.splitlines()
        # Clamp to valid range
        start = max(1, start)
        end = min(len(lines), end)
        return "\n".join(lines[start - 1 : end])

    def read_file(self, file_path: str, max_chars: int = 500_000) -> str | None:
        """Read full file content. Last resort — prefer get_outline + read_symbol."""
        return self._files.read_file(file_path, max_size=max_chars)
