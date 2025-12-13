"""Agentic Contextualizer - Generate AI-friendly codebase context."""

__version__ = "0.1.0"

from .config import Config
from .models import (
    ProjectMetadata,
    CodeAnalysis,
    ContextMetadata,
    GeneratedContext,
    ScopedContextMetadata,
)
from .scoper import (
    extract_keywords,
    search_relevant_files,
    ScopedAnalyzer,
    ScopedGenerator,
)

__all__ = [
    "Config",
    "ProjectMetadata",
    "CodeAnalysis",
    "ContextMetadata",
    "GeneratedContext",
    "ScopedContextMetadata",
    "extract_keywords",
    "search_relevant_files",
    "ScopedAnalyzer",
    "ScopedGenerator",
]
