"""LangChain tools for scoped context generation.

These tools wrap the FileBackend abstraction for use by LangChain agents,
enabling flexible, LLM-guided exploration of repositories.
"""

from .file_tools import create_file_tools, read_scoped_file, search_files
from .analysis_tools import extract_imports
from .code_search_tools import (
    create_code_search_tools,
    grep_pattern,
    find_definitions,
)
from .schemas import (
    ReadFileOutput,
    SearchFilesOutput,
    ExtractImportsOutput,
    ImportInfo,
    LineMatch,
    CodeReference,
    GrepMatch,
    GrepOutput,
    DefinitionMatch,
    FindDefinitionsOutput,
    FileMatch,
)

__all__ = [
    # Tool creators
    "create_file_tools",
    "create_code_search_tools",
    # Core functions
    "read_scoped_file",
    "search_files",
    "extract_imports",
    "grep_pattern",
    "find_definitions",
    # Schemas
    "ReadFileOutput",
    "SearchFilesOutput",
    "ExtractImportsOutput",
    "ImportInfo",
    "LineMatch",
    "CodeReference",
    "GrepMatch",
    "GrepOutput",
    "DefinitionMatch",
    "FindDefinitionsOutput",
    "FileMatch",
]
