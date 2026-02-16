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


# Tool name -> priority mapping
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
