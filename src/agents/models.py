"""Core data models for the Agentic Contextualizer."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class ProjectMetadata(BaseModel):
    """Metadata extracted from a repository."""

    name: str
    path: Path
    project_type: Optional[str] = None  # "python", "node", "rust", etc.
    dependencies: Dict[str, str] = Field(default_factory=dict)
    entry_points: List[str] = Field(default_factory=list)
    key_files: List[str] = Field(default_factory=list)
    readme_content: Optional[str] = None


class CodeAnalysis(BaseModel):
    """Results from LLM-based code analysis."""

    architecture_patterns: List[str] = Field(default_factory=list)
    coding_conventions: Dict[str, str] = Field(default_factory=dict)
    tech_stack: List[str] = Field(default_factory=list)
    insights: str = ""


class ContextMetadata(BaseModel):
    """Frontmatter metadata for generated context files."""

    source_repo: str
    scan_date: datetime = Field(default_factory=datetime.utcnow)
    user_summary: str
    model_used: str


class GeneratedContext(BaseModel):
    """Complete generated context file."""

    metadata: ContextMetadata
    architecture_overview: str
    key_commands: Dict[str, str] = Field(default_factory=dict)
    code_patterns: str
    entry_points: List[str] = Field(default_factory=list)
