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
