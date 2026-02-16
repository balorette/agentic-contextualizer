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
