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

The system follows a **Linear Pipeline** approach to ensure predictability and control costs.

### 1. Structure Scan (No LLM)
- **Goal**: Gather raw data about the project structure.
- **Actions**:
  - Walk the file tree.
  - Read key metadata files (`package.json`, `pyproject.toml`, `README.md`, etc.).
  - Identify project type and key directories.

### 2. Code Analysis (1 LLM Call)
- **Goal**: Extract "smart" insights that static analysis misses.
- **Actions**:
  - Analyze entry points identified in step 1.
  - Identify architectural patterns (e.g., "MVC", "Microservices").
  - Extract coding conventions (error handling, testing patterns).
- **Input**: File tree + content of key files + User Summary.

### 3. Context Generation (1 LLM Call)
- **Goal**: Synthesize all information into the final Context File.
- **Actions**:
  - Combine structure data, code analysis insights, and user summary.
  - Generate a Markdown file following the defined Output Format.

### 4. Refinement (Human-in-the-loop)
- **Goal**: Allow user to improve the context.
- **Actions**:
  - User reviews the generated file.
  - User requests specific changes (e.g., "Add more detail on the auth flow").
  - System updates the context (1 LLM Call).

## Output Format

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
# Generate context
python -m src.main /path/to/repo --summary "Brief description"

# Refine context
python -m src.main --refine "add auth details" contexts/repo/context.md
```

## Code Patterns & Standards

- **LLM Provider**: Use a `LLMProvider` abstract base class to allow easy switching (e.g., OpenRouter, Anthropic, OpenAI).
- **Configuration**: Store model choices and API keys in environment variables or a `.env` file.
- **Logging**: Use standard Python `logging` with JSON formatting for machine-readability.
- **Error Handling**: Fail fast on configuration errors; use retries for transient LLM errors.
