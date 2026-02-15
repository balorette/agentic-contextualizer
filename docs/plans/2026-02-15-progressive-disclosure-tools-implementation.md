# Progressive Disclosure Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the scoped agent's "dump full files" tools with progressive disclosure (outline → symbol → lines → full file) to reduce context bloat by ~65%.

**Architecture:** SmartFileAccess unified layer composes LocalFileBackend (raw I/O) with a pluggable FileAnalysisBackend (semantic ops). AST backend ships first — Python via stdlib `ast`, JS/TS via `tree-sitter`. Agent tools and pipeline mode both consume SmartFileAccess.

**Tech Stack:** Python 3.11+, tree-sitter (tree-sitter-python, tree-sitter-javascript, tree-sitter-typescript), dataclasses, existing LangChain tool factories

---

### Task 1: Add tree-sitter Dependencies

**Files:**
- Modify: `pyproject.toml:7-23`

**Step 1: Add dependencies**

Add to the `[project.dependencies]` list in `pyproject.toml`:

```toml
tree-sitter>=0.24.0
tree-sitter-python>=0.23.0
tree-sitter-javascript>=0.23.0
tree-sitter-typescript>=0.23.0
```

**Step 2: Install**

Run: `cd /root/code/agentic-contextualizer && uv sync`
Expected: Clean install, no errors

**Step 3: Verify import**

Run: `cd /root/code/agentic-contextualizer && uv run python -c "import tree_sitter; import tree_sitter_python; import tree_sitter_javascript; import tree_sitter_typescript; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add tree-sitter dependencies for progressive disclosure parsing"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/agents/backends/__init__.py`
- Create: `src/agents/backends/models.py`
- Test: `tests/agents/backends/test_models.py`

Note: This creates a NEW `src/agents/backends/` package (separate from the existing `src/agents/tools/backends/`). The existing tools backends stay where they are.

**Step 1: Create directory structure**

Run: `mkdir -p /root/code/agentic-contextualizer/src/agents/backends /root/code/agentic-contextualizer/tests/agents/backends`

**Step 2: Create `__init__.py`**

```python
# src/agents/backends/__init__.py
"""Backend abstractions for semantic file analysis.

This package provides pluggable backends for code analysis:
- AST backend (default): Python stdlib ast + tree-sitter for JS/TS
- LSP backend (future): Language Server Protocol for rich semantic info
"""
```

**Step 3: Write the failing test**

```python
# tests/agents/backends/test_models.py
"""Tests for backend data models."""

from src.agents.backends.models import SymbolInfo, SymbolDetail, FileOutline, Reference


class TestSymbolInfo:
    def test_basic_function(self):
        sym = SymbolInfo(
            name="authenticate",
            kind="function",
            line=10,
            line_end=25,
            signature="def authenticate(token: str) -> bool",
        )
        assert sym.name == "authenticate"
        assert sym.kind == "function"
        assert sym.line == 10
        assert sym.line_end == 25
        assert sym.children == []
        assert sym.decorators == []
        assert sym.docstring is None

    def test_class_with_children(self):
        method = SymbolInfo(
            name="process",
            kind="method",
            line=15,
            line_end=25,
            signature="def process(self, data: str) -> None",
        )
        cls = SymbolInfo(
            name="Handler",
            kind="class",
            line=10,
            line_end=30,
            signature="class Handler(BaseHandler):",
            children=[method],
            decorators=["@dataclass"],
            docstring="Handles incoming requests.",
        )
        assert len(cls.children) == 1
        assert cls.children[0].name == "process"
        assert cls.decorators == ["@dataclass"]

    def test_to_dict(self):
        sym = SymbolInfo(
            name="foo",
            kind="function",
            line=1,
            line_end=5,
            signature="def foo()",
        )
        d = sym.to_dict()
        assert d["name"] == "foo"
        assert d["kind"] == "function"
        assert "children" in d


class TestSymbolDetail:
    def test_includes_body(self):
        detail = SymbolDetail(
            name="authenticate",
            kind="function",
            line=10,
            line_end=20,
            signature="def authenticate(token: str) -> bool",
            body='def authenticate(token: str) -> bool:\n    return validate(token)',
            parent=None,
            char_count=60,
        )
        assert detail.body.startswith("def authenticate")
        assert detail.char_count == 60
        assert detail.parent is None

    def test_method_with_parent(self):
        detail = SymbolDetail(
            name="process",
            kind="method",
            line=15,
            line_end=25,
            signature="def process(self) -> None",
            body="def process(self) -> None:\n    pass",
            parent="Handler",
            char_count=35,
        )
        assert detail.parent == "Handler"


class TestFileOutline:
    def test_empty_file(self):
        outline = FileOutline(
            path="src/empty.py",
            language="python",
            imports=[],
            symbols=[],
            line_count=0,
        )
        assert outline.path == "src/empty.py"
        assert outline.symbols == []

    def test_file_with_symbols(self):
        sym = SymbolInfo(
            name="main", kind="function", line=5, line_end=10,
            signature="def main()",
        )
        outline = FileOutline(
            path="src/main.py",
            language="python",
            imports=["os", "sys"],
            symbols=[sym],
            line_count=15,
        )
        assert len(outline.imports) == 2
        assert len(outline.symbols) == 1

    def test_to_dict(self):
        outline = FileOutline(
            path="src/app.py",
            language="python",
            imports=["flask"],
            symbols=[],
            line_count=20,
        )
        d = outline.to_dict()
        assert d["path"] == "src/app.py"
        assert d["language"] == "python"
        assert d["imports"] == ["flask"]


class TestReference:
    def test_basic_reference(self):
        ref = Reference(
            path="src/routes.py",
            line=34,
            context="user = handler.authenticate(token)",
        )
        assert ref.path == "src/routes.py"
        assert ref.line == 34

    def test_to_dict(self):
        ref = Reference(path="test.py", line=1, context="import foo")
        d = ref.to_dict()
        assert d["path"] == "test.py"
        assert d["line"] == 1
```

**Step 4: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/test_models.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 5: Write minimal implementation**

```python
# src/agents/backends/models.py
"""Data models for semantic file analysis."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SymbolInfo:
    """A symbol (function, class, method, variable) found in a file."""

    name: str
    kind: str  # "function", "class", "method", "variable"
    line: int
    line_end: int
    signature: str
    children: list[SymbolInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
            "line_end": self.line_end,
            "signature": self.signature,
            "children": [c.to_dict() for c in self.children],
            "decorators": self.decorators,
        }
        if self.docstring:
            result["docstring"] = self.docstring
        return result


@dataclass
class SymbolDetail(SymbolInfo):
    """A symbol with its full source code body."""

    body: str = ""
    parent: str | None = None
    char_count: int = 0


@dataclass
class FileOutline:
    """Outline of a file: imports and symbols without bodies."""

    path: str
    language: str
    imports: list[str] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)
    line_count: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "imports": self.imports,
            "symbols": [s.to_dict() for s in self.symbols],
            "line_count": self.line_count,
        }


@dataclass
class Reference:
    """A reference to a symbol in a file."""

    path: str
    line: int
    context: str

    def to_dict(self) -> dict:
        return {"path": self.path, "line": self.line, "context": self.context}
```

Also create `tests/agents/backends/__init__.py` (empty).

**Step 6: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/test_models.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/agents/backends/ tests/agents/backends/
git commit -m "feat: add data models for progressive disclosure (SymbolInfo, FileOutline, etc.)"
```

---

### Task 3: FileAnalysisBackend Protocol

**Files:**
- Create: `src/agents/backends/protocol.py`
- Test: `tests/agents/backends/test_protocol.py`

**Step 1: Write the failing test**

```python
# tests/agents/backends/test_protocol.py
"""Tests for FileAnalysisBackend protocol."""

from src.agents.backends.protocol import FileAnalysisBackend
from src.agents.backends.models import SymbolInfo, SymbolDetail, FileOutline, Reference


class FakeAnalysisBackend:
    """Minimal implementation to verify protocol conformance."""

    def get_outline(self, file_path: str, source: str) -> FileOutline:
        return FileOutline(path=file_path, language="python")

    def read_symbol(self, file_path: str, symbol_name: str, source: str) -> SymbolDetail | None:
        return None

    def find_references(self, symbol_name: str, file_backend, scope: str | None = None) -> list[Reference]:
        return []


class TestFileAnalysisBackendProtocol:
    def test_fake_backend_satisfies_protocol(self):
        backend = FakeAnalysisBackend()
        assert isinstance(backend, FileAnalysisBackend)

    def test_class_without_methods_fails_protocol(self):
        class Incomplete:
            pass

        assert not isinstance(Incomplete(), FileAnalysisBackend)
```

**Step 2: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/test_protocol.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# src/agents/backends/protocol.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/test_protocol.py -v`
Expected: All PASS

**Step 5: Update `src/agents/backends/__init__.py`**

```python
"""Backend abstractions for semantic file analysis."""

from .models import SymbolInfo, SymbolDetail, FileOutline, Reference
from .protocol import FileAnalysisBackend

__all__ = [
    "SymbolInfo",
    "SymbolDetail",
    "FileOutline",
    "Reference",
    "FileAnalysisBackend",
]
```

**Step 6: Commit**

```bash
git add src/agents/backends/ tests/agents/backends/
git commit -m "feat: add FileAnalysisBackend protocol for pluggable analysis"
```

---

### Task 4: Python Parser

**Files:**
- Create: `src/agents/backends/parsers/__init__.py`
- Create: `src/agents/backends/parsers/python_parser.py`
- Test: `tests/agents/backends/parsers/__init__.py`
- Test: `tests/agents/backends/parsers/test_python_parser.py`

**Step 1: Create directories**

Run: `mkdir -p /root/code/agentic-contextualizer/src/agents/backends/parsers /root/code/agentic-contextualizer/tests/agents/backends/parsers`

Create empty `__init__.py` files in both directories.

**Step 2: Write the failing test**

```python
# tests/agents/backends/parsers/test_python_parser.py
"""Tests for Python AST parser."""

import pytest
from src.agents.backends.parsers.python_parser import PythonParser


SAMPLE_PYTHON = '''"""Module docstring."""

import os
from pathlib import Path
from typing import Optional

CONSTANT = 42


def standalone_func(x: int, y: int = 0) -> int:
    """Add two numbers."""
    return x + y


@dataclass
class MyHandler:
    """Handles things."""

    name: str

    def process(self, data: str) -> None:
        """Process the data."""
        result = self._validate(data)
        return result

    def _validate(self, data: str) -> bool:
        return len(data) > 0


def another_func():
    pass
'''


class TestPythonParserGetSymbols:
    def setup_method(self):
        self.parser = PythonParser()

    def test_finds_functions(self):
        symbols = self.parser.get_symbols(SAMPLE_PYTHON, "test.py")
        names = [s.name for s in symbols]
        assert "standalone_func" in names
        assert "another_func" in names

    def test_finds_classes(self):
        symbols = self.parser.get_symbols(SAMPLE_PYTHON, "test.py")
        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "MyHandler"

    def test_class_has_method_children(self):
        symbols = self.parser.get_symbols(SAMPLE_PYTHON, "test.py")
        cls = next(s for s in symbols if s.name == "MyHandler")
        child_names = [c.name for c in cls.children]
        assert "process" in child_names
        assert "_validate" in child_names

    def test_captures_signatures(self):
        symbols = self.parser.get_symbols(SAMPLE_PYTHON, "test.py")
        func = next(s for s in symbols if s.name == "standalone_func")
        assert "x: int" in func.signature
        assert "-> int" in func.signature

    def test_captures_decorators(self):
        symbols = self.parser.get_symbols(SAMPLE_PYTHON, "test.py")
        cls = next(s for s in symbols if s.name == "MyHandler")
        assert "@dataclass" in cls.decorators

    def test_captures_docstrings(self):
        symbols = self.parser.get_symbols(SAMPLE_PYTHON, "test.py")
        func = next(s for s in symbols if s.name == "standalone_func")
        assert func.docstring == "Add two numbers."

    def test_captures_line_numbers(self):
        symbols = self.parser.get_symbols(SAMPLE_PYTHON, "test.py")
        func = next(s for s in symbols if s.name == "standalone_func")
        assert func.line > 0
        assert func.line_end > func.line

    def test_handles_syntax_error(self):
        symbols = self.parser.get_symbols("def broken(:", "bad.py")
        assert symbols == []


class TestPythonParserGetImports:
    def setup_method(self):
        self.parser = PythonParser()

    def test_finds_plain_imports(self):
        imports = self.parser.get_imports(SAMPLE_PYTHON)
        assert "os" in imports

    def test_finds_from_imports(self):
        imports = self.parser.get_imports(SAMPLE_PYTHON)
        assert "pathlib" in imports
        assert "typing" in imports


class TestPythonParserExtractSymbol:
    def setup_method(self):
        self.parser = PythonParser()

    def test_extracts_function_body(self):
        detail = self.parser.extract_symbol(SAMPLE_PYTHON, "standalone_func")
        assert detail is not None
        assert detail.name == "standalone_func"
        assert "return x + y" in detail.body
        assert detail.char_count > 0

    def test_extracts_method_with_parent(self):
        detail = self.parser.extract_symbol(SAMPLE_PYTHON, "process")
        assert detail is not None
        assert detail.parent == "MyHandler"
        assert detail.kind == "method"

    def test_returns_none_for_missing(self):
        detail = self.parser.extract_symbol(SAMPLE_PYTHON, "nonexistent")
        assert detail is None

    def test_extracts_class(self):
        detail = self.parser.extract_symbol(SAMPLE_PYTHON, "MyHandler")
        assert detail is not None
        assert detail.kind == "class"
        assert "def process" in detail.body
```

**Step 3: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/parsers/test_python_parser.py -v`
Expected: FAIL — ImportError

**Step 4: Write implementation**

```python
# src/agents/backends/parsers/python_parser.py
"""Python parser using stdlib ast module."""

from __future__ import annotations

import ast
from ..models import SymbolInfo, SymbolDetail


class PythonParser:
    """Parse Python source using stdlib ast."""

    def get_symbols(self, source: str, file_path: str) -> list[SymbolInfo]:
        """Extract top-level symbols from Python source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        lines = source.splitlines()
        symbols: list[SymbolInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                symbols.append(self._func_to_symbol(node, lines))
            elif isinstance(node, ast.ClassDef):
                symbols.append(self._class_to_symbol(node, lines))

        return symbols

    def get_imports(self, source: str) -> list[str]:
        """Extract import module names from Python source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def extract_symbol(self, source: str, symbol_name: str) -> SymbolDetail | None:
        """Extract a specific symbol's full body."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()

        # Search top-level and class methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == symbol_name:
                    parent = self._find_parent_class(tree, node)
                    return self._node_to_detail(node, lines, parent)
            elif isinstance(node, ast.ClassDef):
                if node.name == symbol_name:
                    return self._node_to_detail(node, lines, parent=None)

        return None

    def _func_to_symbol(self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> SymbolInfo:
        """Convert a function AST node to SymbolInfo."""
        return SymbolInfo(
            name=node.name,
            kind="function",
            line=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=self._extract_signature(node, lines),
            decorators=self._extract_decorators(node),
            docstring=self._extract_docstring(node),
        )

    def _class_to_symbol(self, node: ast.ClassDef, lines: list[str]) -> SymbolInfo:
        """Convert a class AST node to SymbolInfo with method children."""
        children: list[SymbolInfo] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                sym = SymbolInfo(
                    name=child.name,
                    kind="method",
                    line=child.lineno,
                    line_end=child.end_lineno or child.lineno,
                    signature=self._extract_signature(child, lines),
                    decorators=self._extract_decorators(child),
                    docstring=self._extract_docstring(child),
                )
                children.append(sym)

        return SymbolInfo(
            name=node.name,
            kind="class",
            line=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=self._extract_signature(node, lines),
            children=children,
            decorators=self._extract_decorators(node),
            docstring=self._extract_docstring(node),
        )

    def _node_to_detail(
        self,
        node: ast.AST,
        lines: list[str],
        parent: str | None,
    ) -> SymbolDetail:
        """Convert an AST node to SymbolDetail with body."""
        start = node.lineno - 1  # 0-indexed
        end = (node.end_lineno or node.lineno)
        body = "\n".join(lines[start:end])
        kind = "method" if parent else ("class" if isinstance(node, ast.ClassDef) else "function")

        return SymbolDetail(
            name=node.name,
            kind=kind,
            line=node.lineno,
            line_end=end,
            signature=self._extract_signature(node, lines),
            decorators=self._extract_decorators(node),
            docstring=self._extract_docstring(node),
            body=body,
            parent=parent,
            char_count=len(body),
        )

    def _extract_signature(self, node: ast.AST, lines: list[str]) -> str:
        """Extract the signature line from source."""
        line = lines[node.lineno - 1].strip()
        return line

    def _extract_decorators(self, node: ast.AST) -> list[str]:
        """Extract decorator names."""
        decorators: list[str] = []
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Name):
                decorators.append(f"@{dec.id}")
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"@{ast.unparse(dec)}")
            elif isinstance(dec, ast.Call):
                decorators.append(f"@{ast.unparse(dec.func)}")
            else:
                decorators.append(f"@{ast.unparse(dec)}")
        return decorators

    def _extract_docstring(self, node: ast.AST) -> str | None:
        """Extract first line of docstring if present."""
        doc = ast.get_docstring(node)
        if doc:
            return doc.split("\n")[0].strip()
        return None

    def _find_parent_class(self, tree: ast.Module, target: ast.AST) -> str | None:
        """Find the enclosing class name for a method."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if child is target:
                        return node.name
        return None
```

Also create `src/agents/backends/parsers/__init__.py`:

```python
"""Language-specific parsers for AST analysis."""

from .python_parser import PythonParser

__all__ = ["PythonParser"]
```

**Step 5: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/parsers/test_python_parser.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/agents/backends/parsers/ tests/agents/backends/parsers/
git commit -m "feat: add Python parser for symbol extraction using stdlib ast"
```

---

### Task 5: JS/TS Parser (tree-sitter)

**Files:**
- Create: `src/agents/backends/parsers/ts_parser.py`
- Test: `tests/agents/backends/parsers/test_ts_parser.py`

**Step 1: Write the failing test**

```python
# tests/agents/backends/parsers/test_ts_parser.py
"""Tests for JS/TS tree-sitter parser."""

import pytest
from src.agents.backends.parsers.ts_parser import TSParser


SAMPLE_JS = '''import { Router } from 'express';
import jwt from 'jsonwebtoken';

const SECRET = 'abc123';

/**
 * Authenticate a user token.
 */
function authenticate(token) {
    const payload = jwt.verify(token, SECRET);
    return payload;
}

class AuthHandler {
    constructor(secret) {
        this.secret = secret;
    }

    validate(token) {
        return jwt.verify(token, this.secret);
    }

    refresh(user) {
        return jwt.sign({ id: user.id }, this.secret);
    }
}

const helper = (x) => x + 1;

export default AuthHandler;
'''

SAMPLE_TS = '''import { Request, Response } from 'express';

interface UserPayload {
    id: string;
    email: string;
}

export function verifyToken(token: string): UserPayload {
    return JSON.parse(atob(token));
}

export class TokenService {
    private secret: string;

    constructor(secret: string) {
        this.secret = secret;
    }

    sign(payload: UserPayload): string {
        return btoa(JSON.stringify(payload));
    }
}
'''


class TestTSParserJS:
    def setup_method(self):
        self.parser = TSParser()

    def test_finds_functions(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        names = [s.name for s in symbols]
        assert "authenticate" in names

    def test_finds_classes(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "AuthHandler"

    def test_class_has_method_children(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        cls = next(s for s in symbols if s.name == "AuthHandler")
        child_names = [c.name for c in cls.children]
        assert "validate" in child_names
        assert "refresh" in child_names

    def test_finds_arrow_functions(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        names = [s.name for s in symbols]
        assert "helper" in names

    def test_finds_constants(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        names = [s.name for s in symbols]
        assert "SECRET" in names

    def test_captures_line_numbers(self):
        symbols = self.parser.get_symbols(SAMPLE_JS, "app.js")
        func = next(s for s in symbols if s.name == "authenticate")
        assert func.line > 0
        assert func.line_end >= func.line


class TestTSParserTS:
    def setup_method(self):
        self.parser = TSParser()

    def test_finds_ts_functions(self):
        symbols = self.parser.get_symbols(SAMPLE_TS, "app.ts")
        names = [s.name for s in symbols]
        assert "verifyToken" in names

    def test_finds_ts_classes(self):
        symbols = self.parser.get_symbols(SAMPLE_TS, "app.ts")
        classes = [s for s in symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "TokenService"

    def test_finds_ts_interfaces(self):
        symbols = self.parser.get_symbols(SAMPLE_TS, "app.ts")
        interfaces = [s for s in symbols if s.kind == "interface"]
        assert len(interfaces) == 1
        assert interfaces[0].name == "UserPayload"


class TestTSParserImports:
    def setup_method(self):
        self.parser = TSParser()

    def test_js_imports(self):
        imports = self.parser.get_imports(SAMPLE_JS)
        assert "express" in imports
        assert "jsonwebtoken" in imports

    def test_ts_imports(self):
        imports = self.parser.get_imports(SAMPLE_TS)
        assert "express" in imports


class TestTSParserExtractSymbol:
    def setup_method(self):
        self.parser = TSParser()

    def test_extracts_function(self):
        detail = self.parser.extract_symbol(SAMPLE_JS, "authenticate", "app.js")
        assert detail is not None
        assert "jwt.verify" in detail.body
        assert detail.char_count > 0

    def test_extracts_method(self):
        detail = self.parser.extract_symbol(SAMPLE_JS, "validate", "app.js")
        assert detail is not None
        assert detail.parent == "AuthHandler"

    def test_returns_none_for_missing(self):
        detail = self.parser.extract_symbol(SAMPLE_JS, "nonexistent", "app.js")
        assert detail is None
```

**Step 2: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/parsers/test_ts_parser.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# src/agents/backends/parsers/ts_parser.py
"""JavaScript/TypeScript parser using tree-sitter."""

from __future__ import annotations

import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

from ..models import SymbolInfo, SymbolDetail

JS_LANGUAGE = Language(tsjs.language())
TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())

# Map file extensions to tree-sitter languages
_LANG_MAP: dict[str, Language] = {
    ".js": JS_LANGUAGE,
    ".jsx": JS_LANGUAGE,
    ".mjs": JS_LANGUAGE,
    ".ts": TS_LANGUAGE,
    ".tsx": TSX_LANGUAGE,
}


class TSParser:
    """Parse JS/TS source using tree-sitter."""

    def _get_language(self, file_path: str) -> Language:
        """Pick the right tree-sitter language from file extension."""
        for ext, lang in _LANG_MAP.items():
            if file_path.endswith(ext):
                return lang
        return JS_LANGUAGE  # default

    def _parse(self, source: str, file_path: str):
        """Parse source and return tree."""
        lang = self._get_language(file_path)
        parser = Parser(lang)
        return parser.parse(source.encode())

    def get_symbols(self, source: str, file_path: str) -> list[SymbolInfo]:
        """Extract top-level symbols from JS/TS source."""
        tree = self._parse(source, file_path)
        lines = source.splitlines()
        symbols: list[SymbolInfo] = []

        for node in tree.root_node.children:
            sym = self._node_to_symbol(node, lines, source)
            if sym:
                symbols.append(sym)

        return symbols

    def get_imports(self, source: str) -> list[str]:
        """Extract import module names from JS/TS source."""
        # Use JS parser — import syntax is the same
        parser = Parser(JS_LANGUAGE)
        tree = parser.parse(source.encode())
        imports: list[str] = []

        for node in tree.root_node.children:
            if node.type == "import_statement":
                source_node = node.child_by_field_name("source")
                if source_node:
                    module = source_node.text.decode().strip("'\"")
                    imports.append(module)

        return imports

    def extract_symbol(self, source: str, symbol_name: str, file_path: str = "file.js") -> SymbolDetail | None:
        """Extract a specific symbol's full body."""
        tree = self._parse(source, file_path)
        lines = source.splitlines()

        return self._find_symbol_in_tree(tree.root_node, symbol_name, lines, source, parent=None)

    def _find_symbol_in_tree(self, root, symbol_name: str, lines: list[str], source: str, parent: str | None) -> SymbolDetail | None:
        """Recursively search for a named symbol."""
        for node in root.children:
            name = self._get_node_name(node)

            # Check class bodies for methods
            if node.type in ("class_declaration", "class"):
                if name == symbol_name:
                    return self._node_to_detail(node, lines, source, parent=None, kind="class")
                body = node.child_by_field_name("body")
                if body:
                    result = self._find_symbol_in_tree(body, symbol_name, lines, source, parent=name)
                    if result:
                        return result
            elif node.type == "export_statement":
                # Check exported declarations
                declaration = node.child_by_field_name("declaration")
                if declaration:
                    dname = self._get_node_name(declaration)
                    if dname == symbol_name:
                        return self._node_to_detail(declaration, lines, source, parent=parent, kind=self._get_kind(declaration))
                    # Check class bodies inside exports
                    if declaration.type in ("class_declaration", "class"):
                        body = declaration.child_by_field_name("body")
                        if body:
                            result = self._find_symbol_in_tree(body, symbol_name, lines, source, parent=dname)
                            if result:
                                return result
            elif name == symbol_name:
                return self._node_to_detail(node, lines, source, parent=parent, kind=self._get_kind(node))

        return None

    def _node_to_symbol(self, node, lines: list[str], source: str) -> SymbolInfo | None:
        """Convert a tree-sitter node to SymbolInfo."""
        # Handle export wrappers
        actual = node
        if node.type == "export_statement":
            declaration = node.child_by_field_name("declaration")
            if declaration:
                actual = declaration
            else:
                return None

        name = self._get_node_name(actual)
        if not name:
            return None

        kind = self._get_kind(actual)
        if not kind:
            return None

        start_line = actual.start_point[0] + 1  # 1-indexed
        end_line = actual.end_point[0] + 1

        signature = lines[start_line - 1].strip() if start_line <= len(lines) else name

        children: list[SymbolInfo] = []
        if kind in ("class", "interface"):
            body = actual.child_by_field_name("body")
            if body:
                for child in body.children:
                    child_sym = self._method_to_symbol(child, lines)
                    if child_sym:
                        children.append(child_sym)

        return SymbolInfo(
            name=name,
            kind=kind,
            line=start_line,
            line_end=end_line,
            signature=signature,
            children=children,
        )

    def _method_to_symbol(self, node, lines: list[str]) -> SymbolInfo | None:
        """Convert a class body member to SymbolInfo."""
        if node.type not in ("method_definition", "public_field_definition", "property_definition"):
            return None

        name = self._get_node_name(node)
        if not name:
            return None

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        signature = lines[start_line - 1].strip() if start_line <= len(lines) else name

        return SymbolInfo(
            name=name,
            kind="method",
            line=start_line,
            line_end=end_line,
            signature=signature,
        )

    def _node_to_detail(self, node, lines: list[str], source: str, parent: str | None, kind: str) -> SymbolDetail:
        """Convert a tree-sitter node to SymbolDetail."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        body = "\n".join(lines[start_line - 1 : end_line])
        signature = lines[start_line - 1].strip() if start_line <= len(lines) else ""

        return SymbolDetail(
            name=self._get_node_name(node) or "",
            kind="method" if parent else kind,
            line=start_line,
            line_end=end_line,
            signature=signature,
            body=body,
            parent=parent,
            char_count=len(body),
        )

    def _get_node_name(self, node) -> str | None:
        """Extract name from various node types."""
        # Direct name field
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()

        # Lexical declaration: const FOO = ...
        if node.type in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type == "variable_declarator":
                    name_n = child.child_by_field_name("name")
                    if name_n:
                        return name_n.text.decode()

        return None

    def _get_kind(self, node) -> str | None:
        """Determine the symbol kind from node type."""
        kind_map = {
            "function_declaration": "function",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "method_definition": "method",
            "lexical_declaration": "variable",
            "variable_declaration": "variable",
        }
        return kind_map.get(node.type)
```

Update `src/agents/backends/parsers/__init__.py`:

```python
"""Language-specific parsers for AST analysis."""

from .python_parser import PythonParser
from .ts_parser import TSParser

__all__ = ["PythonParser", "TSParser"]
```

**Step 4: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/parsers/test_ts_parser.py -v`
Expected: All PASS (some tests may need minor adjustments based on tree-sitter node types — iterate until green)

**Step 5: Commit**

```bash
git add src/agents/backends/parsers/ tests/agents/backends/parsers/
git commit -m "feat: add JS/TS parser using tree-sitter for symbol extraction"
```

---

### Task 6: ASTFileAnalysisBackend

**Files:**
- Create: `src/agents/backends/ast_backend.py`
- Test: `tests/agents/backends/test_ast_backend.py`

**Step 1: Write the failing test**

```python
# tests/agents/backends/test_ast_backend.py
"""Tests for ASTFileAnalysisBackend."""

import pytest
from src.agents.backends.ast_backend import ASTFileAnalysisBackend
from src.agents.backends.protocol import FileAnalysisBackend
from src.agents.tools.backends import InMemoryFileBackend


PYTHON_SOURCE = '''import os
from pathlib import Path

def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"

class Greeter:
    def say_hi(self, name: str) -> str:
        return f"Hi, {name}"
'''

JS_SOURCE = '''import express from 'express';

function handleRequest(req, res) {
    res.send('ok');
}

class Router {
    get(path, handler) {
        return this;
    }
}
'''


class TestASTBackendSatisfiesProtocol:
    def test_isinstance_check(self):
        backend = ASTFileAnalysisBackend()
        assert isinstance(backend, FileAnalysisBackend)


class TestASTBackendGetOutline:
    def setup_method(self):
        self.backend = ASTFileAnalysisBackend()

    def test_python_outline(self):
        outline = self.backend.get_outline("app.py", PYTHON_SOURCE)
        assert outline.language == "python"
        assert "os" in outline.imports
        assert "pathlib" in outline.imports
        symbol_names = [s.name for s in outline.symbols]
        assert "greet" in symbol_names
        assert "Greeter" in symbol_names

    def test_js_outline(self):
        outline = self.backend.get_outline("app.js", JS_SOURCE)
        assert outline.language == "javascript"
        assert "express" in outline.imports
        symbol_names = [s.name for s in outline.symbols]
        assert "handleRequest" in symbol_names
        assert "Router" in symbol_names

    def test_unknown_language_returns_empty_outline(self):
        outline = self.backend.get_outline("data.csv", "a,b,c\n1,2,3")
        assert outline.language == "unknown"
        assert outline.symbols == []

    def test_line_count(self):
        outline = self.backend.get_outline("app.py", PYTHON_SOURCE)
        assert outline.line_count == len(PYTHON_SOURCE.splitlines())


class TestASTBackendReadSymbol:
    def setup_method(self):
        self.backend = ASTFileAnalysisBackend()

    def test_python_symbol(self):
        detail = self.backend.read_symbol("app.py", "greet", PYTHON_SOURCE)
        assert detail is not None
        assert "Hello" in detail.body

    def test_js_symbol(self):
        detail = self.backend.read_symbol("app.js", "handleRequest", JS_SOURCE)
        assert detail is not None
        assert "res.send" in detail.body

    def test_missing_symbol(self):
        detail = self.backend.read_symbol("app.py", "missing", PYTHON_SOURCE)
        assert detail is None


class TestASTBackendFindReferences:
    def setup_method(self):
        self.backend = ASTFileAnalysisBackend()
        self.file_backend = InMemoryFileBackend("/repo", {
            "src/main.py": "from src.utils import greet\nresult = greet('world')\n",
            "src/utils.py": "def greet(name): return f'Hello, {name}'\n",
            "tests/test_main.py": "from src.utils import greet\nassert greet('test') == 'Hello, test'\n",
        })

    def test_finds_references(self):
        refs = self.backend.find_references("greet", self.file_backend)
        assert len(refs) >= 2
        paths = [r.path for r in refs]
        assert "src/main.py" in paths

    def test_references_have_context(self):
        refs = self.backend.find_references("greet", self.file_backend)
        for ref in refs:
            assert "greet" in ref.context

    def test_scoped_search(self):
        refs = self.backend.find_references("greet", self.file_backend, scope="tests")
        paths = [r.path for r in refs]
        assert all(p.startswith("tests/") for p in paths)
```

**Step 2: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/test_ast_backend.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# src/agents/backends/ast_backend.py
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
```

Update `src/agents/backends/__init__.py`:

```python
"""Backend abstractions for semantic file analysis."""

from .models import SymbolInfo, SymbolDetail, FileOutline, Reference
from .protocol import FileAnalysisBackend
from .ast_backend import ASTFileAnalysisBackend

__all__ = [
    "SymbolInfo",
    "SymbolDetail",
    "FileOutline",
    "Reference",
    "FileAnalysisBackend",
    "ASTFileAnalysisBackend",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/backends/test_ast_backend.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agents/backends/ tests/agents/backends/
git commit -m "feat: add ASTFileAnalysisBackend wiring Python and TS parsers"
```

---

### Task 7: SmartFileAccess Unified Layer

**Files:**
- Create: `src/agents/file_access.py`
- Test: `tests/agents/test_file_access.py`

**Step 1: Write the failing test**

```python
# tests/agents/test_file_access.py
"""Tests for SmartFileAccess unified layer."""

import pytest
from src.agents.file_access import SmartFileAccess
from src.agents.tools.backends import InMemoryFileBackend
from src.agents.backends import ASTFileAnalysisBackend


PYTHON_FILE = '''import os

def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"

class Greeter:
    def greet(self, name: str) -> str:
        return f"Hi, {name}"
'''

JS_FILE = '''import express from 'express';

function handleRequest(req, res) {
    res.send('ok');
}
'''


@pytest.fixture
def smart_access():
    file_backend = InMemoryFileBackend("/repo", {
        "src/main.py": PYTHON_FILE,
        "src/app.js": JS_FILE,
        "src/caller.py": "from src.main import hello\nresult = hello('world')\n",
        "config.yaml": "key: value\n",
    })
    analysis = ASTFileAnalysisBackend()
    return SmartFileAccess(file_backend, analysis)


class TestSmartFileAccessGetOutline:
    def test_python(self, smart_access):
        outline = smart_access.get_outline("src/main.py")
        assert outline.language == "python"
        assert "os" in outline.imports
        names = [s.name for s in outline.symbols]
        assert "hello" in names

    def test_js(self, smart_access):
        outline = smart_access.get_outline("src/app.js")
        assert outline.language == "javascript"

    def test_nonexistent_file(self, smart_access):
        outline = smart_access.get_outline("nope.py")
        assert outline is None

    def test_non_code_file(self, smart_access):
        outline = smart_access.get_outline("config.yaml")
        assert outline.language == "unknown"
        assert outline.symbols == []


class TestSmartFileAccessReadSymbol:
    def test_extracts_function(self, smart_access):
        detail = smart_access.read_symbol("src/main.py", "hello")
        assert detail is not None
        assert "Hello" in detail.body

    def test_nonexistent_file(self, smart_access):
        detail = smart_access.read_symbol("nope.py", "hello")
        assert detail is None

    def test_nonexistent_symbol(self, smart_access):
        detail = smart_access.read_symbol("src/main.py", "missing")
        assert detail is None


class TestSmartFileAccessReadLines:
    def test_reads_range(self, smart_access):
        content = smart_access.read_lines("src/main.py", 3, 5)
        assert "def hello" in content
        assert "Hello" in content

    def test_nonexistent_file(self, smart_access):
        content = smart_access.read_lines("nope.py", 1, 5)
        assert content is None


class TestSmartFileAccessReadFile:
    def test_reads_full(self, smart_access):
        content = smart_access.read_file("src/main.py")
        assert "def hello" in content

    def test_nonexistent(self, smart_access):
        content = smart_access.read_file("nope.py")
        assert content is None


class TestSmartFileAccessFindReferences:
    def test_finds_refs(self, smart_access):
        refs = smart_access.find_references("hello")
        paths = [r.path for r in refs]
        assert "src/caller.py" in paths


class TestSmartFileAccessLSPFallback:
    """Test that LSP backend is tried first, falls back to AST."""

    def test_lsp_used_when_available(self):
        from src.agents.backends.models import FileOutline

        class FakeLSP:
            def get_outline(self, path, source):
                return FileOutline(path=path, language="python-lsp", imports=[], symbols=[], line_count=0)

            def read_symbol(self, path, name, source):
                return None

            def find_references(self, name, fb, scope=None):
                return []

        file_backend = InMemoryFileBackend("/repo", {"a.py": "x = 1"})
        analysis = ASTFileAnalysisBackend()
        smart = SmartFileAccess(file_backend, analysis, lsp_backend=FakeLSP())
        outline = smart.get_outline("a.py")
        assert outline.language == "python-lsp"  # came from LSP, not AST

    def test_falls_back_on_lsp_error(self):
        class BrokenLSP:
            def get_outline(self, path, source):
                raise RuntimeError("LSP crashed")

            def read_symbol(self, path, name, source):
                raise RuntimeError("LSP crashed")

            def find_references(self, name, fb, scope=None):
                raise RuntimeError("LSP crashed")

        file_backend = InMemoryFileBackend("/repo", {"a.py": "def foo(): pass\n"})
        analysis = ASTFileAnalysisBackend()
        smart = SmartFileAccess(file_backend, analysis, lsp_backend=BrokenLSP())
        outline = smart.get_outline("a.py")
        assert outline.language == "python"  # fell back to AST
```

**Step 2: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/test_file_access.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# src/agents/file_access.py
"""SmartFileAccess — unified layer for progressive file analysis.

Composes a FileBackend (raw I/O) with a FileAnalysisBackend (semantic ops).
Optionally accepts an LSP backend that takes priority with AST fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .backends import ASTFileAnalysisBackend, FileOutline, SymbolDetail, Reference

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
```

**Step 4: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/test_file_access.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agents/file_access.py tests/agents/test_file_access.py
git commit -m "feat: add SmartFileAccess unified layer with LSP fallback"
```

---

### Task 8: New Agent Tools

**Files:**
- Modify: `src/agents/tools/analysis.py` — replace `extract_file_imports` with `get_file_outline`
- Modify: `src/agents/tools/file.py` — add `read_lines` tool, keep `read_file`
- Modify: `src/agents/tools/search.py` — add `find_references` tool, remove `find_code_definitions`
- Test: `tests/agents/tools/test_progressive_tools.py` (new)

**Step 1: Write the failing test**

```python
# tests/agents/tools/test_progressive_tools.py
"""Tests for progressive disclosure agent tools."""

import pytest
from src.agents.tools.backends import InMemoryFileBackend
from src.agents.backends import ASTFileAnalysisBackend
from src.agents.file_access import SmartFileAccess


PYTHON_SRC = '''import os
from pathlib import Path

def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"

class Handler:
    def process(self, data: str) -> None:
        return data.upper()
'''


@pytest.fixture
def smart_access():
    fb = InMemoryFileBackend("/repo", {
        "src/main.py": PYTHON_SRC,
        "src/caller.py": "from src.main import hello\nhello('world')\n",
    })
    return SmartFileAccess(fb, ASTFileAnalysisBackend())


class TestCreateProgressiveTools:
    def test_creates_all_tools(self, smart_access):
        from src.agents.tools.progressive import create_progressive_tools

        tools = create_progressive_tools(smart_access, max_read_chars=8000)
        names = [t.name for t in tools]
        assert "get_file_outline" in names
        assert "read_symbol" in names
        assert "read_lines" in names
        assert "find_references" in names
        assert "read_file" in names


class TestGetFileOutlineTool:
    def test_returns_outline(self, smart_access):
        from src.agents.tools.progressive import create_progressive_tools

        tools = create_progressive_tools(smart_access)
        outline_tool = next(t for t in tools if t.name == "get_file_outline")
        result = outline_tool.invoke({"file_path": "src/main.py"})
        assert result["language"] == "python"
        assert "os" in result["imports"]
        assert any(s["name"] == "hello" for s in result["symbols"])

    def test_nonexistent_file(self, smart_access):
        from src.agents.tools.progressive import create_progressive_tools

        tools = create_progressive_tools(smart_access)
        outline_tool = next(t for t in tools if t.name == "get_file_outline")
        result = outline_tool.invoke({"file_path": "nope.py"})
        assert result["error"] is not None


class TestReadSymbolTool:
    def test_returns_body(self, smart_access):
        from src.agents.tools.progressive import create_progressive_tools

        tools = create_progressive_tools(smart_access)
        sym_tool = next(t for t in tools if t.name == "read_symbol")
        result = sym_tool.invoke({"file_path": "src/main.py", "symbol_name": "hello"})
        assert "Hello" in result["body"]
        assert result["char_count"] > 0


class TestReadLinesTool:
    def test_reads_range(self, smart_access):
        from src.agents.tools.progressive import create_progressive_tools

        tools = create_progressive_tools(smart_access)
        lines_tool = next(t for t in tools if t.name == "read_lines")
        result = lines_tool.invoke({"file_path": "src/main.py", "start_line": 4, "end_line": 6})
        assert "def hello" in result["content"]


class TestFindReferencesTool:
    def test_finds_refs(self, smart_access):
        from src.agents.tools.progressive import create_progressive_tools

        tools = create_progressive_tools(smart_access)
        ref_tool = next(t for t in tools if t.name == "find_references")
        result = ref_tool.invoke({"symbol_name": "hello"})
        assert len(result["references"]) >= 1
```

**Step 2: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/tools/test_progressive_tools.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

Create a new file `src/agents/tools/progressive.py`:

```python
# src/agents/tools/progressive.py
"""Progressive disclosure tools for scoped agent.

These tools implement the outline → symbol → lines → file hierarchy
to minimize context bloat in agent conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from langchain_core.tools import tool, BaseTool

if TYPE_CHECKING:
    from src.agents.file_access import SmartFileAccess

DEFAULT_MAX_READ_CHARS = 8_000
DEFAULT_MAX_REFS = 30


def create_progressive_tools(
    smart_access: SmartFileAccess,
    max_read_chars: int = DEFAULT_MAX_READ_CHARS,
    max_refs: int = DEFAULT_MAX_REFS,
) -> list[BaseTool]:
    """Create the progressive disclosure tool set.

    Returns tools in order of preference (cheapest first):
    - get_file_outline (~500 bytes)
    - read_symbol (~1-2 KB)
    - read_lines (surgical)
    - find_references (~2-3 KB)
    - read_file (~8 KB, last resort)
    """

    @tool
    def get_file_outline(file_path: str) -> dict:
        """Get a file's structure: imports, function/class names, signatures, and line numbers. No source code bodies.

        This is the CHEAPEST way to understand a file (~500 bytes vs ~8KB for read_file).
        Always call this BEFORE read_file on code files.

        Args:
            file_path: Relative path within the repository.

        Returns:
            Dict with path, language, imports, symbols (with names, kinds, signatures, line numbers, children).
        """
        outline = smart_access.get_outline(file_path)
        if outline is None:
            return {"error": f"File not found: {file_path}", "path": file_path}
        return outline.to_dict()

    @tool
    def read_symbol(file_path: str, symbol_name: str) -> dict:
        """Read a specific function, method, or class body by name.

        Much cheaper than read_file (~1-2 KB vs ~8 KB). Use get_file_outline first
        to find the symbol name, then call this to read its code.

        Args:
            file_path: Relative path within the repository.
            symbol_name: Name of the function, method, or class to extract.

        Returns:
            Dict with name, kind, signature, body (source code), parent class, line numbers, char_count.
        """
        detail = smart_access.read_symbol(file_path, symbol_name)
        if detail is None:
            return {"error": f"Symbol '{symbol_name}' not found in {file_path}", "path": file_path}

        return {
            "name": detail.name,
            "kind": detail.kind,
            "parent": detail.parent,
            "line": detail.line,
            "line_end": detail.line_end,
            "signature": detail.signature,
            "body": detail.body,
            "char_count": detail.char_count,
        }

    @tool
    def read_lines(file_path: str, start_line: int, end_line: int) -> dict:
        """Read a specific line range from a file (1-indexed, inclusive).

        Use when you know the exact lines from grep results or outline line numbers.

        Args:
            file_path: Relative path within the repository.
            start_line: First line to read (1-indexed).
            end_line: Last line to read (inclusive).

        Returns:
            Dict with path, content, line_start, line_end, char_count.
        """
        content = smart_access.read_lines(file_path, start_line, end_line)
        if content is None:
            return {"error": f"File not found: {file_path}", "path": file_path}

        return {
            "path": file_path,
            "content": content,
            "line_start": start_line,
            "line_end": end_line,
            "char_count": len(content),
        }

    @tool
    def find_references(symbol_name: str, scope: str | None = None) -> dict:
        """Find all references to a symbol across the codebase.

        Shows where a function/class is used without reading full files.

        Args:
            symbol_name: Name of the function, class, or variable to find.
            scope: Optional directory to limit search (e.g., "src/" or "tests/").

        Returns:
            Dict with symbol, references (path, line, context), total_found.
        """
        refs = smart_access.find_references(symbol_name, scope)
        limited = refs[:max_refs]
        return {
            "symbol": symbol_name,
            "references": [r.to_dict() for r in limited],
            "total_found": len(refs),
        }

    @tool
    def read_file(file_path: str) -> dict:
        """Read full file content. LAST RESORT — prefer get_file_outline + read_symbol.

        Only use for non-code files (config, README, etc.) or when you truly need the whole file.

        Args:
            file_path: Relative path within the repository.

        Returns:
            Dict with path, content, char_count, truncated.
        """
        content = smart_access.read_file(file_path, max_chars=max_read_chars)
        if content is None:
            return {"error": f"File not found: {file_path}", "path": file_path}

        truncated = len(content) >= max_read_chars
        return {
            "path": file_path,
            "content": content[:max_read_chars],
            "char_count": len(content[:max_read_chars]),
            "truncated": truncated,
        }

    return [get_file_outline, read_symbol, read_lines, find_references, read_file]
```

**Step 4: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/tools/test_progressive_tools.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agents/tools/progressive.py tests/agents/tools/test_progressive_tools.py
git commit -m "feat: add progressive disclosure tools (outline, symbol, lines, references)"
```

---

### Task 9: Wire Tools into Scoped Agent

**Files:**
- Modify: `src/agents/scoper/agent.py` — use progressive tools and updated system prompt
- Modify: `tests/scoper/test_scoped_agent.py` — update tool name assertions

**Step 1: Update the system prompt in `src/agents/scoper/agent.py`**

Replace `SCOPED_AGENT_SYSTEM_PROMPT` (lines 24-107) with the new progressive disclosure prompt. Keep the same variable name.

New prompt content:

```python
SCOPED_AGENT_SYSTEM_PROMPT = """You are a scoped context generator agent. Your goal is to analyze a specific aspect of a codebase and produce focused documentation.

## Available Tools

### File Discovery
1. **search_for_files** - Search for files by keywords (filename or content match)
2. **grep_in_files** - Search for regex patterns with line numbers and context

### Progressive File Analysis (cheapest → most expensive)
3. **get_file_outline** (~500 bytes) - Get file structure: imports, symbols, signatures. ALWAYS use this before read_file.
4. **read_symbol** (~1-2 KB) - Extract a specific function/method/class body by name.
5. **read_lines** (variable) - Read an exact line range from a file.
6. **find_references** (~2-3 KB) - Find all usages of a symbol across the codebase.
7. **read_file** (~8 KB) - Read full file. LAST RESORT — only for config/non-code files.

### Output Generation
8. **generate_scoped_context** - Generate the final scoped context markdown file (pass file paths, not contents).

## Workflow Strategy

### Step 1: SEARCH
- Use `search_for_files` with keywords from the scope question to find candidate files.
- Use `grep_in_files` to search for specific patterns and get line numbers.

### Step 2: OUTLINE
- Use `get_file_outline` on the top candidates to see their structure.
- This shows imports, function names, signatures, and line numbers — WITHOUT reading file bodies.
- Decide which specific symbols are relevant from the outlines.

### Step 3: DRILL
- Use `read_symbol` to read specific functions or classes you identified in Step 2.
- Use `read_lines` when you need a specific range (from grep results or outline line numbers).
- Only use `read_file` for non-code files (config, README, etc.).

### Step 4: CONNECT
- Use `find_references` to see where key functions/classes are used across the codebase.
- Use `get_file_outline` on referenced files to understand how they fit.

### Step 5: GENERATE
When you have sufficient context (typically 5-15 relevant files), use `generate_scoped_context` with:
- The list of **file paths** (the tool reads contents automatically)
- Your analysis and insights
- Code references with specific line numbers

## Cost Hierarchy

RULE: Never call read_file on a code file without first calling get_file_outline.

| Tool | Cost | Use When |
|------|------|----------|
| get_file_outline | ~500 bytes | Understanding any file's structure |
| read_symbol | ~1-2 KB | Need a specific function/method body |
| read_lines | variable | Need an exact line range |
| find_references | ~2-3 KB | Tracing cross-file relationships |
| read_file | ~8 KB | Config/non-code files only |

## Guidelines

- **Budget**: Aim for 5-10 file outlines + 5-10 symbol reads (much cheaper than 10-20 full file reads)
- **Focus**: Stay on topic — don't explore tangential code
- **Tests**: Outline test files to see what's tested, then read specific test functions
- **Imports**: Outlines show imports — follow them to understand architecture
- **Line Numbers**: Track and report specific line numbers for key code
- **Confidence**: Generate output when you can answer the question, not when you've read everything
- **File Paths**: When calling generate_scoped_context, pass file paths — NOT file contents

## Token Economy

Every tool result is added to the conversation and sent with each subsequent API call. Progressive disclosure keeps this small:

- get_file_outline returns ~500 bytes per file (vs ~8 KB for read_file)
- read_symbol returns only the code you need (~1-2 KB)
- A typical session should accumulate ~12 KB of tool results (vs ~35 KB with full file reads)

Start with outlines. Only drill into symbols you actually need.
"""
```

**Step 2: Update `create_scoped_agent` function**

In `create_scoped_agent()` (line 110), replace the tool creation section. Change the imports at top of file and the tool assembly:

Replace imports (lines 11-18):
```python
from ..tools import (
    FileBackend,
    LocalFileBackend,
    CodeReference,
    create_file_tools,
    create_search_tools,
)
from ..tools.progressive import create_progressive_tools
from ..backends import ASTFileAnalysisBackend
from ..file_access import SmartFileAccess
```

Replace tool creation inside `create_scoped_agent()` — where it currently does:
```python
file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
analysis_tools = create_analysis_tools(backend)
code_search_tools = create_search_tools(backend, ...)
```

Replace with:
```python
analysis_backend = ASTFileAnalysisBackend()
smart_access = SmartFileAccess(backend, analysis_backend)

# Progressive disclosure tools (outline → symbol → lines → references → read_file)
progressive_tools = create_progressive_tools(smart_access, max_read_chars=8000)

# Keep search_for_files and grep_in_files for file discovery
file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
search_tools = create_search_tools(backend, max_grep_results=15, max_def_results=15, context_lines=1)

# Combine: discovery tools + progressive tools + generate
# Remove read_file from file_tools (progressive has its own), keep search_for_files
search_for_files_tool = next(t for t in file_tools if t.name == "search_for_files")
grep_tool = next(t for t in search_tools if t.name == "grep_in_files")
tools = [search_for_files_tool, grep_tool] + progressive_tools + [generate_scoped_context]
```

Apply the same changes to `create_scoped_agent_with_budget()`.

**Step 3: Update test assertions**

In `tests/scoper/test_scoped_agent.py`, update `test_creates_agent_with_correct_tools` (line 53-80):

Replace the tool name assertions:
```python
# Old assertions to remove:
# assert "extract_file_imports" in tool_names
# assert "find_code_definitions" in tool_names

# New assertions:
assert "search_for_files" in tool_names
assert "grep_in_files" in tool_names
assert "get_file_outline" in tool_names
assert "read_symbol" in tool_names
assert "read_lines" in tool_names
assert "find_references" in tool_names
assert "read_file" in tool_names
assert "generate_scoped_context" in tool_names
```

**Step 4: Run tests**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/scoper/test_scoped_agent.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agents/scoper/agent.py tests/scoper/test_scoped_agent.py
git commit -m "feat: wire progressive disclosure tools into scoped agent"
```

---

### Task 10: Context Priority Tagging

**Files:**
- Modify: `src/agents/middleware/token_budget.py` — add priority-based tool result tracking
- Test: `tests/agents/test_context_priority.py` (new)

**Step 1: Write the failing test**

```python
# tests/agents/test_context_priority.py
"""Tests for context priority tagging and smart trimming."""

import pytest
from src.agents.middleware.context_priority import (
    ToolResultPriority,
    get_tool_priority,
    summarize_tool_result,
)


class TestGetToolPriority:
    def test_outline_is_high(self):
        assert get_tool_priority("get_file_outline") == ToolResultPriority.HIGH

    def test_search_is_high(self):
        assert get_tool_priority("search_for_files") == ToolResultPriority.HIGH

    def test_grep_is_high(self):
        assert get_tool_priority("grep_in_files") == ToolResultPriority.HIGH

    def test_read_symbol_is_medium(self):
        assert get_tool_priority("read_symbol") == ToolResultPriority.MEDIUM

    def test_find_references_is_medium(self):
        assert get_tool_priority("find_references") == ToolResultPriority.MEDIUM

    def test_read_lines_is_medium(self):
        assert get_tool_priority("read_lines") == ToolResultPriority.MEDIUM

    def test_read_file_is_low(self):
        assert get_tool_priority("read_file") == ToolResultPriority.LOW

    def test_unknown_tool_is_medium(self):
        assert get_tool_priority("some_other_tool") == ToolResultPriority.MEDIUM


class TestSummarizeToolResult:
    def test_summarizes_read_file(self):
        result = {
            "path": "src/main.py",
            "content": "def foo():\n    return 1\n" * 100,
            "char_count": 2400,
            "truncated": False,
        }
        summary = summarize_tool_result("read_file", result)
        assert len(summary) < len(str(result))
        assert "src/main.py" in summary

    def test_passes_through_outline(self):
        result = {"path": "src/main.py", "language": "python", "symbols": [], "imports": []}
        summary = summarize_tool_result("get_file_outline", result)
        # Outlines are already compact, should pass through
        assert "src/main.py" in summary

    def test_passes_through_read_symbol(self):
        result = {"name": "foo", "body": "def foo(): pass", "char_count": 15}
        summary = summarize_tool_result("read_symbol", result)
        assert "foo" in summary
```

**Step 2: Run test to verify it fails**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/test_context_priority.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# src/agents/middleware/context_priority.py
"""Context priority tagging for tool results.

Assigns priorities to tool results so smart trimming can evict
verbose results (full file reads) before compact navigational
results (outlines, search results).
"""

from __future__ import annotations

import json
from enum import IntEnum


class ToolResultPriority(IntEnum):
    """Priority levels for tool results in conversation history."""

    HIGH = 3    # Navigational: outlines, search results
    MEDIUM = 2  # Targeted: symbol reads, references
    LOW = 1     # Verbose: full file reads


# Tool name → priority mapping
_PRIORITY_MAP: dict[str, ToolResultPriority] = {
    "get_file_outline": ToolResultPriority.HIGH,
    "search_for_files": ToolResultPriority.HIGH,
    "grep_in_files": ToolResultPriority.HIGH,
    "read_symbol": ToolResultPriority.MEDIUM,
    "read_lines": ToolResultPriority.MEDIUM,
    "find_references": ToolResultPriority.MEDIUM,
    "read_file": ToolResultPriority.LOW,
    "generate_scoped_context": ToolResultPriority.HIGH,
}


def get_tool_priority(tool_name: str) -> ToolResultPriority:
    """Get the priority level for a tool's results."""
    return _PRIORITY_MAP.get(tool_name, ToolResultPriority.MEDIUM)


def summarize_tool_result(tool_name: str, result: dict) -> str:
    """Produce a compact summary of a tool result for older conversation turns.

    High/medium priority results pass through unchanged (already compact).
    Low priority results (read_file) get summarized to path + char_count.
    """
    if get_tool_priority(tool_name) != ToolResultPriority.LOW:
        return json.dumps(result, default=str)

    # Summarize verbose results
    path = result.get("path", "unknown")
    char_count = result.get("char_count", 0)
    truncated = result.get("truncated", False)
    error = result.get("error")

    if error:
        return json.dumps({"path": path, "error": error})

    summary = {
        "path": path,
        "_summarized": True,
        "char_count": char_count,
        "truncated": truncated,
        "note": f"Full content was read ({char_count} chars). Use read_symbol or read_lines to re-read specific parts.",
    }
    return json.dumps(summary)
```

**Step 4: Run test to verify it passes**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/test_context_priority.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/agents/middleware/context_priority.py tests/agents/test_context_priority.py
git commit -m "feat: add context priority tagging for smart tool result trimming"
```

---

### Task 11: Integrate Priority into Token Budget Middleware

**Files:**
- Modify: `src/agents/middleware/token_budget.py` — use priority in `wrap_tool_call` and `before_model`
- Modify: `tests/agents/test_token_budget_middleware.py` — add priority trimming tests

**Step 1: Write the failing test**

Add to `tests/agents/test_token_budget_middleware.py`:

```python
class TestPriorityBasedTrimming:
    def test_wrap_tool_call_tags_priority(self):
        """Tool results should include priority metadata."""
        from src.agents.middleware.context_priority import ToolResultPriority
        # This test verifies the middleware passes priority through.
        # Exact assertion depends on how LangChain ToolMessage metadata works.
        # The key contract: wrap_tool_call for "read_file" should summarize
        # older results more aggressively than "get_file_outline".
        pass  # Placeholder — implement after reading current wrap_tool_call

    def test_summarizes_old_read_file_results(self):
        """Older read_file results should be summarized before model call."""
        pass  # Placeholder
```

Note: The exact integration depends on how LangChain serializes ToolMessage metadata. This task requires reading the current `wrap_tool_call` and `before_model` implementations and adapting. The key changes:

1. In `wrap_tool_call`: After getting the tool result, call `summarize_tool_result()` for results older than 3 turns.
2. In `before_model`: When trimming, sort messages by priority before evicting. Evict LOW first, then MEDIUM.

**Step 2: Implement the integration**

In `token_budget.py`, import the priority module and modify `wrap_tool_call` (line 130-151) to tag results. Modify `before_model` (line 48-82) to respect priority when trimming.

This is an incremental change — keep existing truncation behavior, add priority awareness on top.

**Step 3: Run all middleware tests**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/agents/test_token_budget_middleware.py -v`
Expected: All PASS (existing + new)

**Step 4: Commit**

```bash
git add src/agents/middleware/token_budget.py tests/agents/test_token_budget_middleware.py
git commit -m "feat: integrate context priority into token budget middleware"
```

---

### Task 12: Pipeline Mode Integration

**Files:**
- Modify: `src/agents/scoper/scoped_analyzer.py` — use SmartFileAccess for exploration
- Modify: `src/agents/scoper/scoped_generator.py` — use SmartFileAccess for generation
- Test: Update existing tests in `tests/scoper/test_scoped_analyzer.py` and `tests/scoper/test_scoped_generator.py`

**Step 1: Update ScopedAnalyzer**

In `scoped_analyzer.py`, update `__init__` (line 64-79) to accept an optional `SmartFileAccess`:

```python
def __init__(
    self,
    llm_provider: LLMProvider,
    file_backend: FileBackend | None = None,
    smart_access: SmartFileAccess | None = None,
    max_rounds: int = MAX_EXPLORATION_ROUNDS,
):
    self._llm = llm_provider
    self._backend = file_backend
    self._smart = smart_access
    self._max_rounds = max_rounds
```

In `analyze()` (line 94-162), when the `SmartFileAccess` is available, use `get_outline()` to build exploration context instead of `read_file()`. This changes `_format_contents_with_limits()` to format outlines instead of full file contents.

This is backward-compatible — if `smart_access` is None, the old `read_file` path is used.

**Step 2: Update ScopedGenerator**

In `scoped_generator.py`, update `__init__` (line 25-33) to accept optional `SmartFileAccess`. In `_format_files()` (line 102-137), when smart access is available, use it for reading files with the same limits.

**Step 3: Run existing tests**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/scoper/test_scoped_analyzer.py tests/scoper/test_scoped_generator.py -v`
Expected: All PASS (backward-compatible)

**Step 4: Commit**

```bash
git add src/agents/scoper/scoped_analyzer.py src/agents/scoper/scoped_generator.py tests/scoper/
git commit -m "feat: integrate SmartFileAccess into pipeline mode (backward-compatible)"
```

---

### Task 13: Update Package Exports

**Files:**
- Modify: `src/agents/tools/__init__.py` — export new tools
- Modify: `src/agents/backends/__init__.py` — ensure clean exports

**Step 1: Update `src/agents/tools/__init__.py`**

Add imports for the progressive tools module:

```python
# Progressive disclosure tools
from .progressive import create_progressive_tools
```

Add to `__all__`:

```python
"create_progressive_tools",
```

**Step 2: Run full test suite**

Run: `cd /root/code/agentic-contextualizer && uv run pytest -v --tb=short`
Expected: All PASS

**Step 3: Commit**

```bash
git add src/agents/tools/__init__.py src/agents/backends/__init__.py
git commit -m "chore: update package exports for progressive disclosure tools"
```

---

### Task 14: Integration Test

**Files:**
- Create: `tests/integration/test_progressive_disclosure.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_progressive_disclosure.py
"""Integration test: progressive disclosure tool flow."""

import pytest
from src.agents.tools.backends import InMemoryFileBackend
from src.agents.backends import ASTFileAnalysisBackend
from src.agents.file_access import SmartFileAccess
from src.agents.tools.progressive import create_progressive_tools


PYTHON_AUTH = '''"""Authentication module."""

import jwt
from datetime import datetime, timedelta
from .models import User, Session

SECRET_KEY = "changeme"

def create_token(user: User) -> str:
    """Create a JWT token for the user."""
    payload = {
        "user_id": user.id,
        "exp": datetime.utcnow() + timedelta(hours=24),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str) -> dict | None:
    """Verify and decode a JWT token."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None

class AuthMiddleware:
    """Middleware for request authentication."""

    def __init__(self, secret: str = SECRET_KEY):
        self.secret = secret

    def authenticate(self, request) -> User | None:
        """Extract and verify token from request."""
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        payload = verify_token(token)
        if payload:
            return User(id=payload["user_id"])
        return None

    def require_auth(self, handler):
        """Decorator for protected routes."""
        def wrapper(request, *args, **kwargs):
            user = self.authenticate(request)
            if not user:
                return {"error": "Unauthorized"}, 401
            return handler(request, *args, user=user, **kwargs)
        return wrapper
'''

PYTHON_ROUTES = '''"""Route handlers."""

from .auth import AuthMiddleware, create_token

auth = AuthMiddleware()

@auth.require_auth
def get_profile(request, user=None):
    return {"user_id": user.id}

def login(request):
    token = create_token(request.user)
    return {"token": token}
'''

JS_APP = '''import express from 'express';
import { authenticate } from './auth.js';

const app = express();

app.get('/profile', authenticate, (req, res) => {
    res.json({ user: req.user });
});

export default app;
'''


@pytest.fixture
def tools():
    fb = InMemoryFileBackend("/repo", {
        "src/auth.py": PYTHON_AUTH,
        "src/routes.py": PYTHON_ROUTES,
        "src/app.js": JS_APP,
        "config.yaml": "secret: changeme\nport: 8080\n",
    })
    smart = SmartFileAccess(fb, ASTFileAnalysisBackend())
    return {t.name: t for t in create_progressive_tools(smart)}


class TestProgressiveDisclosureFlow:
    """Simulate the agent's progressive exploration pattern."""

    def test_outline_then_symbol_flow(self, tools):
        """Step 1: Outline a file. Step 2: Read specific symbol."""
        # Outline first — cheap
        outline = tools["get_file_outline"].invoke({"file_path": "src/auth.py"})
        assert outline["language"] == "python"
        assert any(s["name"] == "create_token" for s in outline["symbols"])
        assert any(s["name"] == "AuthMiddleware" for s in outline["symbols"])

        # Drill into specific symbol — targeted
        detail = tools["read_symbol"].invoke({"file_path": "src/auth.py", "symbol_name": "create_token"})
        assert "jwt.encode" in detail["body"]
        assert detail["char_count"] < 500  # much less than 8KB full file

    def test_find_references_flow(self, tools):
        """Step 3: Find who uses a symbol across files."""
        refs = tools["find_references"].invoke({"symbol_name": "create_token"})
        paths = [r["path"] for r in refs["references"]]
        assert "src/routes.py" in paths

    def test_read_file_for_config(self, tools):
        """read_file is appropriate for non-code files."""
        result = tools["read_file"].invoke({"file_path": "config.yaml"})
        assert "secret: changeme" in result["content"]

    def test_total_context_size_is_small(self, tools):
        """Verify the progressive approach produces much less data than full reads."""
        # Simulate a realistic exploration session
        total_bytes = 0

        # Outline 3 files
        for path in ["src/auth.py", "src/routes.py", "src/app.js"]:
            result = tools["get_file_outline"].invoke({"file_path": path})
            total_bytes += len(str(result))

        # Read 2 specific symbols
        for sym in [("src/auth.py", "create_token"), ("src/auth.py", "verify_token")]:
            result = tools["read_symbol"].invoke({"file_path": sym[0], "symbol_name": sym[1]})
            total_bytes += len(str(result))

        # Find references for 1 symbol
        result = tools["find_references"].invoke({"symbol_name": "authenticate"})
        total_bytes += len(str(result))

        # Total should be well under 8KB (a single full file read)
        assert total_bytes < 8000, f"Progressive flow produced {total_bytes} bytes — should be under 8KB"
```

**Step 2: Run the integration test**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/integration/test_progressive_disclosure.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `cd /root/code/agentic-contextualizer && uv run pytest -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/integration/test_progressive_disclosure.py
git commit -m "test: add integration test for progressive disclosure tool flow"
```

---

### Task 15: Final Verification

**Step 1: Run full test suite with coverage**

Run: `cd /root/code/agentic-contextualizer && uv run pytest --cov=src --cov-report=term-missing -v`
Expected: All PASS, new code has coverage

**Step 2: Run linter**

Run: `cd /root/code/agentic-contextualizer && uv run ruff check src/agents/backends/ src/agents/file_access.py src/agents/tools/progressive.py src/agents/middleware/context_priority.py`
Expected: Clean

**Step 3: Verify no regressions in existing tools**

Run: `cd /root/code/agentic-contextualizer && uv run pytest tests/scoper/ tests/agents/ -v --tb=short`
Expected: All PASS
