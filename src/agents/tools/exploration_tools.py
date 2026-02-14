"""Utility tools for quick exploration of repository contents."""

from pathlib import Path
from typing import Any
from langchain_core.tools import tool

from .file import KEY_FILE_PATTERNS
from .repository_tools import _allowed_repo_root


@tool
def list_key_files(file_list: list[str]) -> dict[str, list[str]]:
    """Categorize key files (configs, entry_points, docs) from a file list.

    Args:
        file_list: List of repo-relative file paths from scan_structure tool

    Returns:
        Dictionary with configs, entry_points, docs, and all_key_files lists.
    """
    found_files: dict[str, list[str]] = {
        "configs": [],
        "entry_points": [],
        "docs": [],
    }

    for file_path in file_list:
        file_name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path

        for category, patterns in KEY_FILE_PATTERNS.items():
            for pattern in patterns:
                if file_name == pattern or file_path.endswith(pattern):
                    found_files[category].append(file_path)
                    break

    all_key_files: list[str] = []
    for category_files in found_files.values():
        all_key_files.extend(category_files)

    return {
        **found_files,
        "all_key_files": sorted(set(all_key_files)),
    }


@tool
def read_file_snippet(file_path: str, start_line: int = 0, num_lines: int = 50) -> dict[str, Any]:
    """Read lines from a file. Returns content, line range, and total_lines.

    Args:
        file_path: Absolute path to the file to read
        start_line: Line number to start from (0-indexed, default: 0)
        num_lines: Number of lines to read (default: 50, max: 500)

    Returns:
        Dictionary with content, start_line, end_line, total_lines, file_path, or error.
    """
    try:
        path = Path(file_path).resolve()

        # Path traversal protection: ensure file is within allowed repo root
        allowed_root = _allowed_repo_root.get(None)
        if allowed_root is not None:
            try:
                path.relative_to(allowed_root)
            except ValueError:
                return {"error": f"Access denied: path is outside the allowed repository"}

        # Validate file exists
        if not path.exists():
            return {"error": f"File does not exist: {file_path}"}

        if not path.is_file():
            return {"error": f"Path is not a file: {file_path}"}

        # Validate parameters
        if start_line < 0:
            return {"error": "start_line must be non-negative"}

        if num_lines <= 0:
            return {"error": "num_lines must be positive"}

        if num_lines > 500:
            return {"error": "num_lines cannot exceed 500 (too large for context)"}

        # Read file
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}

        total_lines = len(all_lines)

        # Handle out-of-bounds start_line
        if start_line >= total_lines:
            return {
                "error": f"start_line ({start_line}) exceeds file length ({total_lines} lines)"
            }

        # Extract snippet
        end_line = min(start_line + num_lines, total_lines)
        snippet_lines = all_lines[start_line:end_line]
        content = "".join(snippet_lines)

        return {
            "content": content,
            "start_line": start_line,
            "end_line": end_line - 1,  # 0-indexed, inclusive
            "total_lines": total_lines,
            "file_path": str(path),
        }

    except Exception as e:
        return {"error": f"Unexpected error reading file: {str(e)}"}
