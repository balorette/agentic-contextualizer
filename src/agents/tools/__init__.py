"""LangChain tools for repository analysis.

This module provides a unified set of tools for both full context generation
and scoped context generation pipelines, using a pluggable backend system
for file access.

Structure:
- backends/: File access abstraction (local, in-memory, etc.)
- file.py: File reading and searching tools
- search.py: Code search tools (grep, find definitions)
- analysis.py: Code analysis tools (import extraction)
- repository_tools.py: High-level pipeline tools (scan, analyze, generate)
- schemas.py: Pydantic schemas for all tool inputs/outputs
"""

# Backend abstractions
from .backends import (
    FileBackend,
    LocalFileBackend,
    InMemoryFileBackend,
    DEFAULT_IGNORED_DIRS,
    DEFAULT_SEARCHABLE_EXTENSIONS,
)

# Schemas - Common
from .schemas import (
    LineMatch,
    CodeReference,
    # File schemas
    ReadFileOutput,
    FileMatch,
    SearchFilesOutput,
    # Search schemas
    GrepMatch,
    GrepOutput,
    DefinitionMatch,
    FindDefinitionsOutput,
    # Analysis schemas
    ImportInfo,
    ExtractImportsOutput,
    # Repository pipeline schemas
    ScanStructureInput,
    ScanStructureOutput,
    ExtractMetadataInput,
    ExtractMetadataOutput,
    AnalyzeCodeInput,
    AnalyzeCodeOutput,
    GenerateContextInput,
    GenerateContextOutput,
    RefineContextInput,
    RefineContextOutput,
    ListKeyFilesInput,
    ListKeyFilesOutput,
    ReadFileSnippetInput,
    ReadFileSnippetOutput,
)

# File tools
from .file import (
    read_file_content,
    search_files,
    list_key_files,
    create_file_tools,
    KEY_FILE_PATTERNS,
    DEFAULT_MAX_CHARS,
)

# Alias for backward compatibility
read_file = read_file_content

# Search tools
from .search import (
    grep_pattern,
    find_definitions,
    create_search_tools,
)

# Analysis tools
from .analysis import (
    extract_imports,
    create_analysis_tools,
)

# Repository pipeline tools (high-level, for full context generation)
from .repository_tools import (
    scan_structure,
    extract_metadata,
    analyze_code,
    generate_context,
    refine_context,
)

# Exploration tools (kept for backward compatibility)
from .exploration_tools import (
    read_file_snippet,
)

__all__ = [
    # Backends
    "FileBackend",
    "LocalFileBackend",
    "InMemoryFileBackend",
    "DEFAULT_IGNORED_DIRS",
    "DEFAULT_SEARCHABLE_EXTENSIONS",
    # Common Schemas
    "LineMatch",
    "CodeReference",
    # File Schemas
    "ReadFileOutput",
    "FileMatch",
    "SearchFilesOutput",
    # Search Schemas
    "GrepMatch",
    "GrepOutput",
    "DefinitionMatch",
    "FindDefinitionsOutput",
    # Analysis Schemas
    "ImportInfo",
    "ExtractImportsOutput",
    # Repository Pipeline Schemas
    "ScanStructureInput",
    "ScanStructureOutput",
    "ExtractMetadataInput",
    "ExtractMetadataOutput",
    "AnalyzeCodeInput",
    "AnalyzeCodeOutput",
    "GenerateContextInput",
    "GenerateContextOutput",
    "RefineContextInput",
    "RefineContextOutput",
    "ListKeyFilesInput",
    "ListKeyFilesOutput",
    "ReadFileSnippetInput",
    "ReadFileSnippetOutput",
    # File tools
    "read_file",
    "search_files",
    "list_key_files",
    "create_file_tools",
    "KEY_FILE_PATTERNS",
    "DEFAULT_MAX_CHARS",
    # Search tools
    "grep_pattern",
    "find_definitions",
    "create_search_tools",
    # Analysis tools
    "extract_imports",
    "create_analysis_tools",
    # Repository tools
    "scan_structure",
    "extract_metadata",
    "analyze_code",
    "generate_context",
    "refine_context",
    # Exploration tools (backward compatibility)
    "read_file_snippet",
]
