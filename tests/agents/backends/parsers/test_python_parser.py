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
