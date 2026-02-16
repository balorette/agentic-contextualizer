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
