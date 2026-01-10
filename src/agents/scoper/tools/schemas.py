"""Pydantic schemas for scoped context tools."""

from pydantic import BaseModel, Field


class LineMatch(BaseModel):
    """A single line match from grep/search."""

    line_num: int = Field(description="1-indexed line number")
    content: str = Field(description="The line content (stripped)")
    context_before: list[str] = Field(
        default_factory=list, description="Lines before match"
    )
    context_after: list[str] = Field(
        default_factory=list, description="Lines after match"
    )


class CodeReference(BaseModel):
    """A code reference for output documentation."""

    path: str = Field(description="File path")
    line_start: int = Field(description="Starting line number")
    line_end: int | None = Field(
        default=None, description="Ending line (None for single line)"
    )
    description: str = Field(description="Brief description of what this code does")


class GrepMatch(BaseModel):
    """A single grep match result."""

    path: str = Field(description="File path where match was found")
    line_num: int = Field(description="1-indexed line number")
    content: str = Field(description="The matching line content")
    context_before: list[str] = Field(
        default_factory=list, description="Lines before match"
    )
    context_after: list[str] = Field(
        default_factory=list, description="Lines after match"
    )


class GrepOutput(BaseModel):
    """Output schema for grep_pattern tool."""

    matches: list[GrepMatch] = Field(description="List of matches with location info")
    total_matches: int = Field(description="Total matches found")
    pattern: str = Field(description="Pattern that was searched")
    files_searched: int = Field(default=0, description="Number of files searched")
    error: str | None = Field(default=None, description="Error message if search failed")


class DefinitionMatch(BaseModel):
    """A code definition match."""

    name: str = Field(description="Name of the definition")
    def_type: str = Field(description="Type: function, class, method, variable")
    path: str = Field(description="File path")
    line_num: int = Field(description="Line number where definition starts")
    line_end: int | None = Field(
        default=None, description="Line where definition ends"
    )
    signature: str | None = Field(
        default=None, description="Function/method signature if applicable"
    )


class FindDefinitionsOutput(BaseModel):
    """Output schema for find_definitions tool."""

    definitions: list[DefinitionMatch] = Field(description="Matching definitions")
    name_searched: str = Field(description="Name that was searched")
    files_searched: int = Field(default=0, description="Number of files searched")
    error: str | None = Field(default=None, description="Error message if search failed")


class ReadFileOutput(BaseModel):
    """Output schema for read_scoped_file tool."""

    content: str | None = Field(
        description="File content, or None if file could not be read"
    )
    path: str = Field(description="Relative path that was requested")
    char_count: int = Field(description="Number of characters in content (0 if None)")
    truncated: bool = Field(
        description="Whether content was truncated due to max_chars limit"
    )
    error: str | None = Field(
        default=None,
        description="Error message if file could not be read",
    )


class FileMatch(BaseModel):
    """A single file match from search."""

    path: str = Field(description="Relative path to the file")
    match_type: str = Field(description="How the file matched: 'filename' or 'content'")
    score: int = Field(description="Match score (higher = more relevant)")
    line_matches: list[LineMatch] = Field(
        default_factory=list,
        description="Specific line matches if available (for content matches)",
    )


class SearchFilesOutput(BaseModel):
    """Output schema for search_files tool."""

    matches: list[FileMatch] = Field(description="List of matching files")
    total_found: int = Field(description="Total number of matches found")
    keywords_used: list[str] = Field(description="Keywords that were searched")
    error: str | None = Field(
        default=None,
        description="Error message if search failed",
    )


class ImportInfo(BaseModel):
    """Information about a single import statement."""

    module: str = Field(description="The imported module or package name")
    names: list[str] = Field(
        default_factory=list,
        description="Specific names imported (for 'from X import Y')",
    )
    alias: str | None = Field(
        default=None,
        description="Alias if imported with 'as' (e.g., 'import numpy as np')",
    )
    is_relative: bool = Field(
        default=False,
        description="Whether this is a relative import (starts with .)",
    )
    resolved_path: str | None = Field(
        default=None,
        description="Resolved file path within repo, if determinable",
    )


class ExtractImportsOutput(BaseModel):
    """Output schema for extract_imports tool."""

    imports: list[ImportInfo] = Field(description="List of imports found in file")
    path: str = Field(description="Path of the analyzed file")
    language: str = Field(description="Detected language: 'python', 'javascript', etc.")
    error: str | None = Field(
        default=None,
        description="Error message if extraction failed",
    )
