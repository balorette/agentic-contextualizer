"""Pydantic schemas for LangChain tool inputs and outputs."""

from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class ScanStructureInput(BaseModel):
    """Input schema for scan_structure tool."""

    repo_path: str = Field(description="Path to the repository to scan")
    ignore_patterns: list[str] = Field(
        default_factory=list, description="Additional patterns to ignore beyond defaults"
    )

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        """Validate that repo_path exists and is a directory."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Repository path is not a directory: {v}")
        return str(path.resolve())


class ScanStructureOutput(BaseModel):
    """Output schema for scan_structure tool."""

    tree: dict[str, Any] = Field(description="Nested directory tree structure")
    all_files: list[str] = Field(description="Flat list of all file paths")
    total_files: int = Field(description="Total number of files")
    total_dirs: int = Field(description="Total number of directories")
    error: Optional[str] = Field(default=None, description="Error message if scan failed")


class ExtractMetadataInput(BaseModel):
    """Input schema for extract_metadata tool."""

    repo_path: str = Field(description="Path to the repository")

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        """Validate that repo_path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        return str(path.resolve())


class ExtractMetadataOutput(BaseModel):
    """Output schema for extract_metadata tool."""

    project_type: Optional[str] = Field(
        default=None, description="Detected project type (python, node, rust, etc.)"
    )
    dependencies: dict[str, str] = Field(
        default_factory=dict, description="Project dependencies (name -> version)"
    )
    entry_points: list[str] = Field(
        default_factory=list, description="Identified entry point files"
    )
    key_files: list[str] = Field(
        default_factory=list, description="Important configuration/metadata files"
    )
    error: Optional[str] = Field(default=None, description="Error message if extraction failed")


class AnalyzeCodeInput(BaseModel):
    """Input schema for analyze_code tool."""

    repo_path: str = Field(description="Path to the repository")
    user_summary: str = Field(description="User's description of the project")
    metadata_dict: dict[str, Any] = Field(
        default_factory=dict, description="Metadata from extract_metadata"
    )
    file_tree: dict[str, Any] = Field(
        default_factory=dict, description="File tree from scan_structure"
    )

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        """Validate that repo_path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        return str(path.resolve())


class AnalyzeCodeOutput(BaseModel):
    """Output schema for analyze_code tool."""

    architecture_patterns: list[str] = Field(
        default_factory=list, description="Identified architectural patterns"
    )
    coding_conventions: dict[str, str] = Field(
        default_factory=dict, description="Coding conventions and standards"
    )
    tech_stack: list[str] = Field(default_factory=list, description="Technologies used")
    insights: str = Field(default="", description="Additional insights about the codebase")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")


class GenerateContextInput(BaseModel):
    """Input schema for generate_context tool."""

    repo_path: str = Field(description="Path to the repository")
    user_summary: str = Field(description="User's description of the project")
    metadata_dict: dict[str, Any] = Field(description="Metadata from extract_metadata")
    analysis_dict: dict[str, Any] = Field(description="Analysis from analyze_code")
    output_path: Optional[str] = Field(
        default=None, description="Optional custom output path for context file"
    )

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        """Validate that repo_path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        return str(path.resolve())


class GenerateContextOutput(BaseModel):
    """Output schema for generate_context tool."""

    context_md: str = Field(description="Generated markdown context content")
    output_path: str = Field(description="Path where context file was written")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")


class RefineContextInput(BaseModel):
    """Input schema for refine_context tool."""

    context_file_path: str = Field(description="Path to existing context file")
    refinement_request: str = Field(description="Description of what to change/add")

    @field_validator("context_file_path")
    @classmethod
    def validate_context_file(cls, v: str) -> str:
        """Validate that context file exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Context file does not exist: {v}")
        if not path.is_file():
            raise ValueError(f"Context path is not a file: {v}")
        return str(path.resolve())


class RefineContextOutput(BaseModel):
    """Output schema for refine_context tool."""

    updated_context: str = Field(description="Updated markdown context content")
    output_path: str = Field(description="Path where updated context was written")
    error: Optional[str] = Field(default=None, description="Error message if refinement failed")


class ListKeyFilesInput(BaseModel):
    """Input schema for list_key_files utility tool."""

    file_tree: dict[str, Any] = Field(description="File tree structure from scan_structure")


class ListKeyFilesOutput(BaseModel):
    """Output schema for list_key_files utility tool."""

    key_files: list[str] = Field(
        description="List of important files (configs, entry points, docs)"
    )


class ReadFileSnippetInput(BaseModel):
    """Input schema for read_file_snippet utility tool."""

    file_path: str = Field(description="Path to the file to read")
    start_line: int = Field(default=0, description="Line number to start reading from (0-indexed)")
    num_lines: int = Field(default=50, description="Number of lines to read")

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate that file exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return str(path.resolve())

    @field_validator("start_line")
    @classmethod
    def validate_start_line(cls, v: int) -> int:
        """Validate start_line is non-negative."""
        if v < 0:
            raise ValueError("start_line must be non-negative")
        return v

    @field_validator("num_lines")
    @classmethod
    def validate_num_lines(cls, v: int) -> int:
        """Validate num_lines is positive and reasonable."""
        if v <= 0:
            raise ValueError("num_lines must be positive")
        if v > 500:
            raise ValueError("num_lines cannot exceed 500 (too large for context)")
        return v


class ReadFileSnippetOutput(BaseModel):
    """Output schema for read_file_snippet utility tool."""

    content: str = Field(description="File content snippet")
    start_line: int = Field(description="Starting line number (0-indexed)")
    end_line: int = Field(description="Ending line number (0-indexed)")
    total_lines: int = Field(description="Total lines in the file")
    error: Optional[str] = Field(default=None, description="Error message if read failed")
