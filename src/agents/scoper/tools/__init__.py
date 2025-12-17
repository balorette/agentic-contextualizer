"""LangChain tools for scoped context generation.

These tools wrap the FileBackend abstraction for use by LangChain agents,
enabling flexible, LLM-guided exploration of repositories.
"""

from .file_tools import create_file_tools, read_scoped_file, search_files
from .analysis_tools import extract_imports
from .schemas import (
    ReadFileOutput,
    SearchFilesOutput,
    ExtractImportsOutput,
    ImportInfo,
)

__all__ = [
    "create_file_tools",
    "read_scoped_file",
    "search_files",
    "extract_imports",
    "ReadFileOutput",
    "SearchFilesOutput",
    "ExtractImportsOutput",
    "ImportInfo",
]
