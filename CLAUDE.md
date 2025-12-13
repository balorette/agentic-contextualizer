# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agentic Contextualizer** is a tool designed to generate effective context files for codebases that AI coding agents can use to understand projects quickly.

**Goal**: Build a maintainable, cost-effective solution that scans a repository and produces actionable markdown documentation ("Context") for downstream agents.

**Tech Stack:**
- **Language**: Python 3.x
- **LLM Abstraction**: LangChain (designed to support LangGraph in the future)
- **Package Manager**: uv
- **Testing**: pytest

**Design Principles:**
- **Simplicity**: Avoid needless complexity. Use a linear pipeline over complex autonomous loops initially.
- **Cost-Efficiency**: Minimize LLM calls. Use static analysis where possible.
- **Maintainability**: Clean, readable code that is easy to build upon.
- **Smart**: Use targeted LLM calls for code analysis to extract insights static tools miss.

## Core Architecture

The system provides two main pipelines: **Full Context Generation** and **Scoped Context Generation**.

### Full Context Pipeline

Follows a **Linear Pipeline** approach to ensure predictability and control costs.

#### 1. Structure Scan (No LLM)
- **Goal**: Gather raw data about the project structure.
- **Actions**:
  - Walk the file tree.
  - Read key metadata files (`package.json`, `pyproject.toml`, `README.md`, etc.).
  - Identify project type and key directories.

#### 2. Code Analysis (1 LLM Call)
- **Goal**: Extract "smart" insights that static analysis misses.
- **Actions**:
  - Analyze entry points identified in step 1.
  - Identify architectural patterns (e.g., "MVC", "Microservices").
  - Extract coding conventions (error handling, testing patterns).
- **Input**: File tree + content of key files + User Summary.

#### 3. Context Generation (1 LLM Call)
- **Goal**: Synthesize all information into the final Context File.
- **Actions**:
  - Combine structure data, code analysis insights, and user summary.
  - Generate a Markdown file following the defined Output Format.

#### 4. Refinement (Human-in-the-loop)
- **Goal**: Allow user to improve the context.
- **Actions**:
  - User reviews the generated file.
  - User requests specific changes (e.g., "Add more detail on the auth flow").
  - System updates the context (1 LLM Call).

### Scoped Context Pipeline

Generates **focused context** for a specific question or topic. Uses a three-phase approach.

#### 1. Discovery (No LLM)
- **Goal**: Find candidate files relevant to the scope question.
- **Actions**:
  - Extract keywords from the question (filters stopwords, normalizes case).
  - Search files by name/path match and content match.
  - Score and rank candidates.
- **Module**: `src/agents/scoper/discovery.py`

#### 2. Exploration (1-3 LLM Calls)
- **Goal**: Intelligently expand context by following imports/dependencies.
- **Actions**:
  - LLM reviews candidate files.
  - Identifies additional files to examine (imports, tests, configs).
  - Iterates until sufficient context or max rounds reached.
- **Module**: `src/agents/scoper/scoped_analyzer.py`

#### 3. Synthesis (1 LLM Call)
- **Goal**: Generate focused context answering the specific question.
- **Actions**:
  - Combine relevant files and insights.
  - Generate targeted markdown documentation.
- **Module**: `src/agents/scoper/scoped_generator.py`

## Output Format

### Full Context Format

Generated context files are **Markdown with YAML frontmatter**:

```markdown
---
source_repo: /path/to/repo
scan_date: 2025-01-22T10:30:00Z
user_summary: "User's original description"
model_used: anthropic/claude-3-5-sonnet
---

# Repository Context: {repo-name}

## Architecture Overview
{High-level patterns, tech stack, design decisions}

## Key Commands
{Build, test, lint, run, deploy commands}

## Code Patterns
{How errors are handled, state managed, tests written}

## Entry Points
{Where execution starts, main files}
```

### Scoped Context Format

Scoped context files use a similar format with scope-specific metadata:

```markdown
---
source_repo: /path/to/repo
scope_question: "authentication flow"
scan_date: 2025-01-22T10:30:00Z
model_used: anthropic/claude-3-5-sonnet
files_analyzed: 8
source_context: contexts/repo/context.md  # Optional, if scoped from existing context
---

## Summary
{Direct answer to the scope question}

## {Relevant Sections}
{Tailored to the specific topic - could be API endpoints, data models, etc.}

## Key Files
{List of files the reader should examine}

## Usage Examples
{How this functionality is used, related tests}
```

**Output locations:**
- Full context: `contexts/{repo-name}/context.md`
- Scoped context: `contexts/{repo-name}/scope-{topic}.md`

## Development Commands

### Setup
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_generator.py
```

### Running
```bash
# Generate full context
python -m agents.main generate /path/to/repo --summary "Brief description"

# Refine existing context
python -m agents.main refine contexts/repo/context.md --request "add auth details"

# Generate scoped context from repository
python -m agents.main scope /path/to/repo --question "authentication flow"

# Generate scoped context from existing context file
python -m agents.main scope contexts/repo/context.md --question "API endpoints"

# Scoped context with custom output
python -m agents.main scope /path/to/repo -q "weather functionality" -o my-scope.md

# Agent mode (for any command)
python -m agents.main generate /path/to/repo -s "Description" --mode agent --stream
python -m agents.main scope /path/to/repo -q "auth flow" --mode agent --stream
```

## Code Patterns & Standards

- **LLM Provider**: Use a `LLMProvider` abstract base class to allow easy switching (e.g., OpenRouter, Anthropic, OpenAI).
- **Configuration**: Store model choices and API keys in environment variables or a `.env` file.
- **Logging**: Use standard Python `logging` with JSON formatting for machine-readability.
- **Error Handling**: Fail fast on configuration errors; use retries for transient LLM errors.

## Module Structure

```
src/agents/
├── __init__.py          # Public API exports
├── main.py              # CLI entry point (generate, refine, scope commands)
├── config.py            # Configuration management
├── models.py            # Pydantic data models
├── scanner/             # Structure scanning (no LLM)
│   ├── structure.py     # File tree walking
│   └── metadata.py      # Project metadata extraction
├── analyzer/            # Code analysis (LLM-based)
│   └── code_analyzer.py # Architecture/pattern detection
├── generator/           # Context generation (LLM-based)
│   └── context_generator.py
├── scoper/              # Scoped context generation
│   ├── __init__.py      # Module exports
│   ├── discovery.py     # Keyword extraction, file search
│   ├── scoped_analyzer.py   # LLM-guided exploration
│   └── scoped_generator.py  # Scoped context synthesis
└── llm/                 # LLM abstraction layer
    ├── provider.py      # LLMProvider base class, AnthropicProvider
    └── prompts.py       # Prompt templates
```

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `Config` | `config.py` | Environment configuration |
| `ContextMetadata` | `models.py` | Full context frontmatter |
| `ScopedContextMetadata` | `models.py` | Scoped context frontmatter |
| `LLMProvider` | `llm/provider.py` | Abstract LLM interface |
| `AnthropicProvider` | `llm/provider.py` | Claude implementation |
| `ScopedAnalyzer` | `scoper/scoped_analyzer.py` | LLM-guided file exploration |
| `ScopedGenerator` | `scoper/scoped_generator.py` | Scoped context file generation |
