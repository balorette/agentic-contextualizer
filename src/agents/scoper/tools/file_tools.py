"""File access tools for scoped context generation."""

from typing import Callable
from langchain_core.tools import tool, BaseTool

from ..backends import FileBackend
from ..discovery import extract_keywords, search_relevant_files, SEARCHABLE_EXTENSIONS
from .schemas import ReadFileOutput, SearchFilesOutput, FileMatch

# Default maximum characters to return from a file
DEFAULT_MAX_CHARS = 13_500


def read_scoped_file(
    backend: FileBackend,
    file_path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ReadFileOutput:
    """Read a file from the repository.

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
    """Search for files matching keywords.

    Args:
        backend: File backend to use
        keywords: Keywords to search for in filenames and content
        max_results: Maximum number of results to return

    Returns:
        SearchFilesOutput with matching files
    """
    from pathlib import Path

    try:
        repo_path = Path(backend.repo_path)
        results = search_relevant_files(repo_path, keywords, max_results)

        matches = [
            FileMatch(
                path=r["path"],
                match_type=r["match_type"],
                score=r["score"],
            )
            for r in results
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


def create_file_tools(backend: FileBackend) -> list[BaseTool]:
    """Create file tools bound to a specific backend.

    Args:
        backend: File backend to bind tools to

    Returns:
        List of LangChain tools ready for agent use
    """

    @tool
    def read_file(file_path: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
        """Read a file from the repository for scoped context analysis.

        Use this to examine files you've identified as potentially relevant
        to the scope question. The file must be within the repository.

        Args:
            file_path: Relative path to file within the repository
            max_chars: Maximum characters to return (default: 15000)

        Returns:
            Dictionary with:
            - content: File content (or None if unreadable)
            - path: The requested path
            - char_count: Number of characters returned
            - truncated: Whether content was truncated
            - error: Error message if file couldn't be read
        """
        result = read_scoped_file(backend, file_path, max_chars)
        return result.model_dump()

    @tool
    def search_for_files(keywords: list[str], max_results: int = 30) -> dict:
        """Search for files matching keywords in the repository.

        Use this as your first step to find candidate files for a scope question.
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
