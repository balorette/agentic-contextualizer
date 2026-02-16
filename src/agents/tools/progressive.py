"""Progressive disclosure tools for scoped agent.

These tools implement the outline -> symbol -> lines -> file hierarchy
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

        truncated = len(content) > max_read_chars
        visible_content = content[:max_read_chars] if truncated else content
        return {
            "path": file_path,
            "content": visible_content,
            "char_count": len(visible_content),
            "truncated": truncated,
        }

    return [get_file_outline, read_symbol, read_lines, find_references, read_file]
