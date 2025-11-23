"""Utility tools for quick exploration of repository contents."""

from pathlib import Path
from typing import Any
from langchain_core.tools import tool


# Key file patterns to identify important files
KEY_FILE_PATTERNS = {
    "configs": [
        "package.json",
        "pyproject.toml",
        "setup.py",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "composer.json",
        ".env.example",
        "tsconfig.json",
        "webpack.config.js",
        "vite.config.js",
    ],
    "entry_points": [
        "main.py",
        "__main__.py",
        "index.js",
        "index.ts",
        "main.go",
        "main.rs",
        "app.py",
        "server.py",
        "cli.py",
    ],
    "docs": [
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "LICENSE",
        "CLAUDE.md",
        "docs/index.md",
    ],
}


@tool
def list_key_files(file_tree: dict[str, Any]) -> dict[str, list[str]]:
    """List important files from a file tree structure.

    Identifies and categorizes key files like configuration files, entry points,
    and documentation from a repository file tree. Useful for quick navigation
    and understanding project structure.

    Args:
        file_tree: File tree dictionary from scan_structure tool

    Returns:
        Dictionary with categorized lists:
        - configs: Configuration files
        - entry_points: Main entry point files
        - docs: Documentation files
        - all_key_files: Combined list of all key files found
    """
    found_files = {
        "configs": [],
        "entry_points": [],
        "docs": [],
    }

    def scan_tree(node: dict[str, Any], current_path: str = ""):
        """Recursively scan tree for key files."""
        if node.get("type") == "file":
            file_name = node.get("name", "")
            file_path = node.get("path", current_path)

            # Check each category
            for category, patterns in KEY_FILE_PATTERNS.items():
                for pattern in patterns:
                    if file_name == pattern or file_path.endswith(pattern):
                        found_files[category].append(file_path)
                        break

        elif node.get("type") == "directory":
            for child in node.get("children", []):
                child_path = f"{current_path}/{child.get('name', '')}" if current_path else child.get("name", "")
                scan_tree(child, child_path)

    # Scan the tree
    scan_tree(file_tree)

    # Create combined list
    all_key_files = []
    for category_files in found_files.values():
        all_key_files.extend(category_files)

    return {
        **found_files,
        "all_key_files": sorted(set(all_key_files)),
    }


@tool
def read_file_snippet(file_path: str, start_line: int = 0, num_lines: int = 50) -> dict[str, Any]:
    """Read a snippet from a specific file.

    Reads a portion of a file's content, useful for examining specific sections
    without loading entire large files. Supports line-based reading with
    start position and length control.

    Args:
        file_path: Absolute path to the file to read
        start_line: Line number to start reading from (0-indexed, default: 0)
        num_lines: Number of lines to read (default: 50, max: 500)

    Returns:
        Dictionary containing:
        - content: The file content snippet
        - start_line: Starting line number (0-indexed)
        - end_line: Ending line number (0-indexed)
        - total_lines: Total number of lines in the file
        - file_path: Path to the file that was read
        - error: Error message if read failed
    """
    try:
        path = Path(file_path)

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
            with path.open("r", encoding="utf-8", errors="ignore") as f:
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
