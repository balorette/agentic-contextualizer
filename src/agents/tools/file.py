"""File access tools using the FileBackend abstraction.

This module provides LangChain tools for reading files and searching
repositories using a pluggable backend system.
"""

from pathlib import Path
from typing import Any
from langchain_core.tools import tool, BaseTool

from .backends import FileBackend, DEFAULT_IGNORED_DIRS
from .schemas import ReadFileOutput, SearchFilesOutput, FileMatch

# Default maximum characters to return from a file
DEFAULT_MAX_CHARS = 13_500

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


# =============================================================================
# Core Functions (Backend-aware)
# =============================================================================


def read_file_content(
    backend: FileBackend,
    file_path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ReadFileOutput:
    """Read a file from the repository using a backend.

    Args:
        backend: File backend to use
        file_path: Relative path within the repository
        max_chars: Maximum characters to return (truncates if exceeded)

    Returns:
        ReadFileOutput with content and metadata
    """
    content = backend.read_file(file_path)

    if content is None:
        return ReadFileOutput(
            content=None,
            path=file_path,
            char_count=0,
            truncated=False,
            error=f"Could not read file: {file_path}",
        )

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    return ReadFileOutput(
        content=content,
        path=file_path,
        char_count=len(content),
        truncated=truncated,
        error=None,
    )


def search_files(
    backend: FileBackend,
    keywords: list[str],
    max_results: int = 30,
) -> SearchFilesOutput:
    """Search for files matching keywords using a backend.

    Args:
        backend: File backend to use
        keywords: Keywords to search for in filenames and content
        max_results: Maximum number of results to return

    Returns:
        SearchFilesOutput with matching files
    """
    try:
        results: list[dict] = []
        keyword_set = set(kw.lower() for kw in keywords)

        for rel_path in backend.walk_files(ignore_dirs=DEFAULT_IGNORED_DIRS):
            if len(results) >= max_results:
                break

            # Check filename/path match
            path_lower = rel_path.lower()
            filename_lower = Path(rel_path).name.lower()

            name_matches = sum(1 for kw in keyword_set if kw in filename_lower)
            path_matches = sum(1 for kw in keyword_set if kw in path_lower)

            if name_matches > 0 or path_matches > 0:
                results.append({
                    "path": rel_path,
                    "match_type": "filename",
                    "score": name_matches * 2 + path_matches,
                })
                continue

            # Check content match
            content = backend.read_file(rel_path)
            if content:
                content_lower = content.lower()
                content_matches = sum(1 for kw in keyword_set if kw in content_lower)
                if content_matches > 0:
                    results.append({
                        "path": rel_path,
                        "match_type": "content",
                        "score": content_matches,
                    })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        matches = [
            FileMatch(
                path=r["path"],
                match_type=r["match_type"],
                score=r["score"],
            )
            for r in results[:max_results]
        ]

        return SearchFilesOutput(
            matches=matches,
            total_found=len(matches),
            keywords_used=keywords,
            error=None,
        )
    except Exception as e:
        return SearchFilesOutput(
            matches=[],
            total_found=0,
            keywords_used=keywords,
            error=str(e),
        )


def list_key_files(file_tree: dict[str, Any]) -> dict[str, list[str]]:
    """List important files from a file tree structure.

    Identifies and categorizes key files like configuration files, entry points,
    and documentation from a repository file tree.

    Args:
        file_tree: File tree dictionary from scan_structure tool

    Returns:
        Dictionary with categorized lists:
        - configs: Configuration files
        - entry_points: Main entry point files
        - docs: Documentation files
        - all_key_files: Combined list of all key files found
    """
    found_files: dict[str, list[str]] = {
        "configs": [],
        "entry_points": [],
        "docs": [],
    }

    def scan_tree(node: dict[str, Any], current_path: str = "") -> None:
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
                child_path = (
                    f"{current_path}/{child.get('name', '')}"
                    if current_path
                    else child.get("name", "")
                )
                scan_tree(child, child_path)

    # Scan the tree
    scan_tree(file_tree)

    # Create combined list
    all_key_files: list[str] = []
    for category_files in found_files.values():
        all_key_files.extend(category_files)

    return {
        **found_files,
        "all_key_files": sorted(set(all_key_files)),
    }


# =============================================================================
# LangChain Tool Factory
# =============================================================================


def create_file_tools(
    backend: FileBackend,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_search_results: int = 30,
) -> list[BaseTool]:
    """Create file tools bound to a specific backend.

    Args:
        backend: File backend to bind tools to
        max_chars: Maximum characters for read_file (default: 13500)
        max_search_results: Maximum results for search_for_files (default: 30)

    Returns:
        List of LangChain tools ready for agent use
    """

    _default_max_chars = max_chars
    _default_max_search = max_search_results

    @tool
    def read_file(file_path: str, max_chars: int = _default_max_chars) -> dict:
        """Read a file from the repository.

        Use this to examine files you've identified as potentially relevant.
        The file must be within the repository.

        Args:
            file_path: Relative path to file within the repository
            max_chars: Maximum characters to return (default: 13500)

        Returns:
            Dictionary with:
            - content: File content (or None if unreadable)
            - path: The requested path
            - char_count: Number of characters returned
            - truncated: Whether content was truncated
            - error: Error message if file couldn't be read
        """
        result = read_file_content(backend, file_path, max_chars)
        return result.model_dump()

    @tool
    def search_for_files(keywords: list[str], max_results: int = _default_max_search) -> dict:
        """Search for files matching keywords in the repository.

        Use this as your first step to find candidate files.
        Searches both filenames/paths and file contents.

        Args:
            keywords: Keywords to search for (case-insensitive)
            max_results: Maximum number of results (default: 30)

        Returns:
            Dictionary with:
            - matches: List of matching files with path, match_type, score
            - total_found: Number of matches found
            - keywords_used: Keywords that were searched
            - error: Error message if search failed
        """
        result = search_files(backend, keywords, max_results)
        return result.model_dump()

    return [read_file, search_for_files]
