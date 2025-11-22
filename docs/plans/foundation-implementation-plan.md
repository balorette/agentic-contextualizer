# Implementation Plan: Agentic Contextualizer Foundation

**Goal**: Establish the base project structure, configuration, and core abstractions needed to build the Agentic Contextualizer.

**Scope**: Foundations, boilerplate, and project scaffolding only. No implementation of the actual pipeline logic.

---

## Task 1: Project Configuration & Dependency Setup

**File**: `pyproject.toml`

Create the project configuration file with:

```toml
[project]
name = "agentic-contextualizer"
version = "0.1.0"
description = "Generate effective context files for codebases that AI coding agents can use"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "langchain>=0.1.0",
    "langchain-anthropic>=0.1.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
target-version = "py310"
```

**Why**: Defines the project metadata, dependencies, and tooling configuration using modern Python packaging standards.

---

## Task 2: Directory Structure Creation

Create the following directory structure:

```
agentic-contextualizer/
├── src/
│   └── agentic_contextualizer/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── models.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── provider.py
│       │   └── prompts.py
│       ├── scanner/
│       │   ├── __init__.py
│       │   ├── structure.py
│       │   └── metadata.py
│       ├── analyzer/
│       │   ├── __init__.py
│       │   └── code_analyzer.py
│       └── generator/
│           ├── __init__.py
│           └── context_generator.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_scanner.py
│   ├── test_analyzer.py
│   └── test_generator.py
├── contexts/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

**Commands to create**:
```bash
mkdir -p src/agentic_contextualizer/{llm,scanner,analyzer,generator}
mkdir -p tests
mkdir -p contexts
touch src/agentic_contextualizer/__init__.py
touch src/agentic_contextualizer/{main.py,config.py,models.py}
touch src/agentic_contextualizer/llm/{__init__.py,provider.py,prompts.py}
touch src/agentic_contextualizer/scanner/{__init__.py,structure.py,metadata.py}
touch src/agentic_contextualizer/analyzer/{__init__.py,code_analyzer.py}
touch src/agentic_contextualizer/generator/{__init__.py,context_generator.py}
touch tests/{__init__.py,conftest.py,test_scanner.py,test_analyzer.py,test_generator.py}
touch contexts/.gitkeep
```

**Why**: Establishes clear separation of concerns with dedicated modules for each pipeline stage.

---

## Task 3: Core Data Models

**File**: `src/agentic_contextualizer/models.py`

Define Pydantic models for data structures:

```python
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
```

**Why**: Type-safe data models ensure consistency across the pipeline and enable validation.

---

## Task 4: Configuration Management

**File**: `src/agentic_contextualizer/config.py`

```python
"""Configuration management for the Agentic Contextualizer."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Config(BaseModel):
    """Application configuration."""

    # LLM Settings
    llm_provider: str = Field(default="anthropic")
    model_name: str = Field(default="claude-3-5-sonnet-20241022")
    api_key: Optional[str] = Field(default=None)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=60)

    # Scanner Settings
    max_file_size: int = Field(default=1_000_000)  # 1MB
    ignored_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git", "node_modules", "__pycache__", ".venv",
            "venv", "dist", "build", ".pytest_cache"
        ]
    )

    # Output Settings
    output_dir: Path = Field(default=Path("contexts"))

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model_name=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
```

**Why**: Centralizes configuration with environment variable support and sensible defaults.

---

## Task 5: LLM Provider Abstraction

**File**: `src/agentic_contextualizer/llm/provider.py`

```python
"""Abstract LLM provider interface."""
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    model: str
    tokens_used: Optional[int] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system: Optional system prompt

        Returns:
            LLMResponse containing the generated text
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider using LangChain."""

    def __init__(self, model_name: str, api_key: str):
        """Initialize the Anthropic provider.

        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key
        """
        self.model_name = model_name
        self.api_key = api_key
        # TODO: Initialize LangChain Anthropic client

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Generate a response using Claude."""
        # TODO: Implement using LangChain
        raise NotImplementedError("LLM integration will be implemented in next phase")
```

**Why**: Provides abstraction layer that allows swapping LLM providers without changing business logic.

---

## Task 6: Prompt Templates

**File**: `src/agentic_contextualizer/llm/prompts.py`

```python
"""Prompt templates for LLM calls."""


CODE_ANALYSIS_PROMPT = """You are analyzing a codebase to extract architectural insights.

PROJECT STRUCTURE:
{file_tree}

KEY FILES CONTENT:
{key_files_content}

USER SUMMARY:
{user_summary}

Analyze this codebase and provide:
1. Architectural patterns (e.g., MVC, microservices, monolith)
2. Technology stack
3. Coding conventions (error handling, testing patterns, state management)
4. Notable design decisions

Format your response as structured JSON with keys: architecture_patterns, tech_stack, coding_conventions, insights.
"""


CONTEXT_GENERATION_PROMPT = """Generate a comprehensive context file for this codebase.

PROJECT METADATA:
{project_metadata}

CODE ANALYSIS:
{code_analysis}

USER SUMMARY:
{user_summary}

Create a markdown document following this structure:
1. Architecture Overview
2. Key Commands (build, test, run, deploy)
3. Code Patterns
4. Entry Points

Be concise but thorough. Focus on what an AI agent needs to understand the codebase quickly.
"""


REFINEMENT_PROMPT = """Update the existing context file based on user feedback.

CURRENT CONTEXT:
{current_context}

USER REQUEST:
{user_request}

Generate the updated context file incorporating the requested changes.
"""
```

**Why**: Separates prompt engineering from code logic, making prompts easy to iterate on.

---

## Task 7: CLI Entry Point

**File**: `src/agentic_contextualizer/main.py`

```python
"""Main CLI entry point for Agentic Contextualizer."""
import click
from pathlib import Path
from .config import Config


@click.group()
def cli():
    """Agentic Contextualizer - Generate AI-friendly codebase context."""
    pass


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--summary', '-s', required=True, help='Brief description of the project')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def generate(repo_path: str, summary: str, output: str | None):
    """Generate context for a repository.

    Example:
        agentic-contextualizer generate /path/to/repo -s "FastAPI REST API"
    """
    click.echo(f"Generating context for: {repo_path}")
    click.echo(f"Summary: {summary}")
    # TODO: Implement pipeline
    raise NotImplementedError("Pipeline implementation coming in next phase")


@cli.command()
@click.argument('context_file', type=click.Path(exists=True))
@click.option('--request', '-r', required=True, help='What to change')
def refine(context_file: str, request: str):
    """Refine an existing context file.

    Example:
        agentic-contextualizer refine contexts/myapp/context.md -r "Add auth details"
    """
    click.echo(f"Refining: {context_file}")
    click.echo(f"Request: {request}")
    # TODO: Implement refinement
    raise NotImplementedError("Refinement implementation coming in next phase")


if __name__ == '__main__':
    cli()
```

**Why**: Provides user-facing CLI interface using Click for clean argument handling.

---

## Task 8: Package Initialization

**File**: `src/agentic_contextualizer/__init__.py`

```python
"""Agentic Contextualizer - Generate AI-friendly codebase context."""

__version__ = "0.1.0"

from .config import Config
from .models import (
    ProjectMetadata,
    CodeAnalysis,
    ContextMetadata,
    GeneratedContext,
)

__all__ = [
    "Config",
    "ProjectMetadata",
    "CodeAnalysis",
    "ContextMetadata",
    "GeneratedContext",
]
```

**Why**: Exposes public API and version information.

---

## Task 9: Test Infrastructure

**File**: `tests/conftest.py`

```python
"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path
from agentic_contextualizer.config import Config


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository structure for testing."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Create sample files
    (repo / "README.md").write_text("# Test Project")
    (repo / "main.py").write_text("print('hello')")

    return repo


@pytest.fixture
def config() -> Config:
    """Provide a test configuration."""
    return Config(
        llm_provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        api_key="test-key",
    )
```

**Why**: Provides reusable test fixtures for consistent testing.

---

## Task 10: Basic Structure Tests

**File**: `tests/test_scanner.py`

```python
"""Tests for the structure scanner (placeholder)."""
import pytest
from pathlib import Path


def test_scanner_placeholder():
    """Placeholder test for scanner module."""
    # TODO: Implement when scanner is built
    assert True
```

**File**: `tests/test_analyzer.py`

```python
"""Tests for the code analyzer (placeholder)."""
import pytest


def test_analyzer_placeholder():
    """Placeholder test for analyzer module."""
    # TODO: Implement when analyzer is built
    assert True
```

**File**: `tests/test_generator.py`

```python
"""Tests for the context generator (placeholder)."""
import pytest


def test_generator_placeholder():
    """Placeholder test for generator module."""
    # TODO: Implement when generator is built
    assert True
```

**Why**: Establishes test file structure; actual tests will be added during implementation.

---

## Task 11: Environment Configuration

**File**: `.env.example`

```bash
# LLM Provider Configuration
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_api_key_here

# Optional: OpenRouter fallback
# OPENROUTER_API_KEY=your_key_here

# Scanner Settings
MAX_FILE_SIZE=1000000
```

**Why**: Documents required environment variables for users.

---

## Task 12: Git Configuration

**File**: `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Environment
.env

# Generated contexts (keep the directory)
contexts/*
!contexts/.gitkeep
```

**Why**: Prevents committing generated files, secrets, and build artifacts.

---

## Task 13: README Setup

**File**: `README.md`

```markdown
# Agentic Contextualizer

Generate effective context files for codebases that AI coding agents can use to understand projects quickly.

## Status

🚧 **Under Active Development** - Foundation phase complete

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repo-url>
cd agentic-contextualizer

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

*Coming soon - pipeline implementation in progress*

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture and design decisions.

## License

MIT
```

**Why**: Provides quick start guide and sets expectations about project status.

---

## Task 14: Verification

Run the following commands to verify the foundation is set up correctly:

```bash
# 1. Install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Verify package imports
python -c "from agentic_contextualizer import Config, ProjectMetadata; print('✓ Imports work')"

# 3. Run tests
pytest -v

# 4. Check CLI
python -m agentic_contextualizer.main --help

# 5. Verify linting passes
ruff check src/
black --check src/
```

**Expected output**: All imports succeed, tests pass (even if placeholder), CLI shows help text.

---

## Summary

This plan establishes:

✅ **Project structure** - Clean separation of concerns
✅ **Configuration** - Environment-based with type safety
✅ **Data models** - Pydantic models for the pipeline
✅ **LLM abstraction** - Provider pattern for flexibility
✅ **CLI interface** - User-facing commands
✅ **Testing infrastructure** - Pytest setup with fixtures
✅ **Documentation** - README and development guide

**Next Phase**: Implement the actual pipeline logic (scanner → analyzer → generator).

**Estimated Complexity**: Foundation setup is straightforward; no complex logic yet. All files contain clear TODOs for implementation phase.
