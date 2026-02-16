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
