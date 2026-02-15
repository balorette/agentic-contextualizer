"""AST-based file analysis backend.

Uses stdlib ast for Python and tree-sitter for JS/TS.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from .models import FileOutline, SymbolDetail, Reference
from .parsers import PythonParser, TSParser

if TYPE_CHECKING:
    from src.agents.tools.backends import FileBackend

# Extension to language mapping
_PYTHON_EXTS = {".py", ".pyi"}
_JS_TS_EXTS = {".js", ".jsx", ".mjs", ".ts", ".tsx"}
_SEARCHABLE_EXTS = _PYTHON_EXTS | _JS_TS_EXTS


class ASTFileAnalysisBackend:
    """Semantic analysis using AST parsing.

    Python: stdlib ast. JS/TS: tree-sitter.
    Falls back gracefully for unsupported languages.
    """

    def __init__(self) -> None:
        self._python = PythonParser()
        self._ts = TSParser()

    def _detect_language(self, file_path: str) -> str:
        ext = PurePosixPath(file_path).suffix.lower()
        if ext in _PYTHON_EXTS:
            return "python"
        if ext in {".js", ".jsx", ".mjs"}:
            return "javascript"
        if ext in {".ts", ".tsx"}:
            return "typescript"
        return "unknown"

    def get_outline(self, file_path: str, source: str) -> FileOutline:
        language = self._detect_language(file_path)

        if language == "python":
            symbols = self._python.get_symbols(source, file_path)
            imports = self._python.get_imports(source)
        elif language in ("javascript", "typescript"):
            symbols = self._ts.get_symbols(source, file_path)
            imports = self._ts.get_imports(source)
        else:
            symbols = []
            imports = []

        return FileOutline(
            path=file_path,
            language=language,
            imports=imports,
            symbols=symbols,
            line_count=len(source.splitlines()),
        )

    def read_symbol(self, file_path: str, symbol_name: str, source: str) -> SymbolDetail | None:
        language = self._detect_language(file_path)

        if language == "python":
            return self._python.extract_symbol(source, symbol_name)
        elif language in ("javascript", "typescript"):
            return self._ts.extract_symbol(source, symbol_name, file_path)

        return None

    def find_references(
        self, symbol_name: str, file_backend: FileBackend, scope: str | None = None
    ) -> list[Reference]:
        """Grep-based reference finding. LSP backend will replace this."""
        refs: list[Reference] = []
        pattern = re.compile(re.escape(symbol_name))

        for path in file_backend.walk_files(root=scope or ""):
            ext = PurePosixPath(path).suffix.lower()
            if ext not in _SEARCHABLE_EXTS:
                continue

            content = file_backend.read_file(path)
            if not content:
                continue

            for i, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    refs.append(Reference(path=path, line=i, context=line.strip()))

            if len(refs) >= 50:
                break

        return refs
