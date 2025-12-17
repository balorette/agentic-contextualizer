"""Pydantic schemas for scoped context tools."""

from pydantic import BaseModel, Field


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
