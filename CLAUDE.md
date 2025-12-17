## Project Overview

**Agentic Contextualizer** is a tool designed to generate effective context files for codebases that AI coding agents can use to understand projects quickly.

**Goal**: Build a maintainable, cost-effective solution that scans a repository and produces actionable markdown documentation ("Context") for downstream agents.

**Tech Stack:**
- **Language**: Python 3.x
- **LLM Abstraction**: LangChain / LangGraph 
- **Package Manager**: uv
- **Testing**: pytest

**Design Principles:**
- **Simplicity**: Avoid needless complexity.
- **Cost-Efficiency**: Minimize LLM calls. Use static analysis where possible.
- **Maintainability**: Clean, readable code that is easy to build upon.

## Core Architecture

The system provides two main pipelines: **Full Context Generation** and **Scoped Context Generation**.

### Full Context Pipeline

Follows an approach to ensure predictability and control costs.

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
## Code Patterns & Standards

- **LLM Provider**: Use a `LLMProvider` abstract base class to allow easy switching (e.g., OpenRouter, Anthropic, OpenAI).
- **Configuration**: Store model choices and API keys in environment variables or a `.env` file.
- **Logging**: Use standard Python `logging` with JSON formatting for machine-readability.
- **Error Handling**: Fail fast on configuration errors; use retries for transient LLM errors.
