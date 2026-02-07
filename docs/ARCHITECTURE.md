# Architecture Documentation

> Agentic Contextualizer v0.1.0

## System Overview

Agentic Contextualizer is a Python CLI tool that scans codebases and generates structured markdown context files for AI coding agents. It provides two main pipelines — **Full Context Generation** and **Scoped Context Generation** — each available in both **pipeline** (deterministic) and **agent** (LangChain-driven) execution modes.

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (Click)                          │
│           generate  |  refine  |  scope                     │
├────────────┬────────┴──────────┬────────────────────────────┤
│  Pipeline  │                   │  Agent Mode                │
│  Mode      │                   │  (LangChain/LangGraph)     │
├────────────┴───────────────────┴────────────────────────────┤
│                    Core Modules                             │
│  Scanner | Analyzer | Generator | Scoper | LLM Provider     │
├─────────────────────────────────────────────────────────────┤
│                    Tools Layer                              │
│  File Tools | Search Tools | Analysis Tools | Repo Tools    │
├─────────────────────────────────────────────────────────────┤
│                  Backend Abstraction                        │
│         LocalFileBackend  |  InMemoryFileBackend            │
└─────────────────────────────────────────────────────────────┘
```

## Module Map

| Module | Path | LLM Calls | Purpose |
|--------|------|-----------|---------|
| CLI | `src/agents/main.py` | 0 | Command routing and orchestration |
| Config | `src/agents/config.py` | 0 | Environment-based configuration |
| Models | `src/agents/models.py` | 0 | Pydantic data models |
| Scanner | `src/agents/scanner/` | 0 | File tree walking, metadata extraction |
| Analyzer | `src/agents/analyzer/` | 1 | LLM-based code analysis |
| Generator | `src/agents/generator/` | 1 | Context markdown generation |
| Scoper | `src/agents/scoper/` | 2-4 | Scoped context pipeline |
| LLM Provider | `src/agents/llm/` | - | Anthropic API abstraction |
| Factory | `src/agents/factory.py` | 0 | Agent creation with tools |
| Tools | `src/agents/tools/` | 0 | LangChain tool definitions |
| Middleware | `src/agents/middleware/` | 0 | Budget tracking, HITL |
| Memory | `src/agents/memory.py` | 0 | Checkpointing and state |
| Streaming | `src/agents/streaming.py` | 0 | Real-time output |
| Observability | `src/agents/observability.py` | 0 | LangSmith tracing |
| Repo Resolver | `src/agents/repo_resolver.py` | 0 | GitHub URL / local path resolution |

## Pipeline Architecture

### Full Context Generation

The full pipeline makes exactly **2 LLM calls**:

```
                     ┌──────────────┐
                     │  CLI Input   │
                     │  repo + desc │
                     └──────┬───────┘
                            │
              ┌─────────────▼──────────────┐
              │  1. Structure Scan         │  No LLM
              │  StructureScanner.scan()   │
              │  - Walk file tree          │
              │  - Respect .gitignore      │
              │  - Depth limit: 6          │
              │  - Max children: 200/dir   │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  2. Metadata Extraction    │  No LLM
              │  MetadataExtractor.extract │
              │  - Detect project type     │
              │  - Parse dependencies      │
              │  - Find entry points       │
              │  - Read README             │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  3. Code Analysis          │  LLM Call #1
              │  CodeAnalyzer.analyze()    │
              │  - Architecture patterns   │
              │  - Tech stack              │
              │  - Coding conventions      │
              │  - Max 20 files read       │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  4. Context Generation     │  LLM Call #2
              │  ContextGenerator.generate │
              │  - YAML frontmatter        │
              │  - Structured markdown     │
              │  - Writes to contexts/     │
              └─────────────┬──────────────┘
                            │
                     ┌──────▼───────┐
                     │  Output:     │
                     │  context.md  │
                     └──────────────┘
```

### Scoped Context Generation

The scoped pipeline uses **2-4 LLM calls** depending on exploration depth:

```
                     ┌──────────────────┐
                     │  CLI Input       │
                     │  repo + question │
                     └──────┬───────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Phase 1: Discovery        │  No LLM
              │  - Extract keywords        │
              │  - Search filenames        │
              │  - Search file content     │
              │  - Score and rank          │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Phase 2: Exploration      │  1-3 LLM Calls
              │  ScopedAnalyzer.analyze()  │
              │  - Review candidates       │
              │  - Follow imports/deps     │
              │  - Max 2 rounds            │
              │  - Max 12 files/prompt     │
              │  - 80k char content limit  │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Phase 3: Synthesis        │  1 LLM Call
              │  ScopedGenerator.generate  │
              │  - Focused markdown        │
              │  - Code references         │
              │  - 120k char limit         │
              └─────────────┬──────────────┘
                            │
                     ┌──────▼───────────┐
                     │  Output:         │
                     │  scope-{topic}.md│
                     └──────────────────┘
```

### Refinement Pipeline

Refinement makes **1 LLM call**:

```
  Existing context.md + User request
         │
         ▼
  ContextGenerator.refine()  ──→  Updated context.md
         (1 LLM call)
```

## Agent Mode Architecture

Agent mode uses LangChain/LangGraph to create autonomous agents that decide which tools to call.

### Full Context Agent

Created via `factory.create_contextualizer_agent()`:

```
┌──────────────────────────────────────────┐
│  LangChain Agent (StateGraph)            │
│                                          │
│  Model: Claude (via init_chat_model)     │
│  System Prompt: AGENT_SYSTEM_PROMPT      │
│                                          │
│  Tools:                                  │
│  ├── scan_structure                      │
│  ├── extract_metadata                    │
│  ├── analyze_code      (expensive)       │
│  ├── generate_context  (expensive)       │
│  ├── refine_context    (expensive)       │
│  ├── list_key_files                      │
│  └── read_file_snippet                   │
│                                          │
│  Optional:                               │
│  ├── Checkpointer (state persistence)    │
│  ├── BudgetTracker (cost control)        │
│  └── Human-in-the-loop (approval gates)  │
└──────────────────────────────────────────┘
```

### Scoped Context Agent

Created via `scoper.agent.create_scoped_agent()`:

```
┌──────────────────────────────────────────┐
│  Scoped Agent (StateGraph)               │
│                                          │
│  Model: Claude (bound to LocalBackend)   │
│                                          │
│  Tools:                                  │
│  ├── search_for_files                    │
│  ├── read_file                           │
│  ├── grep_in_files                       │
│  ├── find_code_definitions               │
│  ├── extract_file_imports                │
│  └── generate_scoped_context             │
│                                          │
│  Backend: LocalFileBackend(repo_path)    │
└──────────────────────────────────────────┘
```

## Tools System

### Backend Abstraction

All file operations go through the `FileBackend` protocol:

```python
class FileBackend(Protocol):
    @property
    def repo_path(self) -> Path: ...
    def read_file(self, path: str, max_chars: int) -> str | None: ...
    def file_exists(self, path: str) -> bool: ...
    def walk_files(self) -> Iterator[str]: ...
    def search_content(self, pattern: str) -> list[tuple[str, int, str]]: ...
```

**Implementations:**
- `LocalFileBackend` — Real filesystem with path traversal protection
- `InMemoryFileBackend` — Dict-based fake filesystem for testing

### Tool Categories

| Category | Tools | Used By |
|----------|-------|---------|
| File | `read_file`, `search_for_files` | Scoped agent |
| Search | `grep_in_files`, `find_code_definitions` | Scoped agent |
| Analysis | `extract_file_imports` | Scoped agent |
| Repository | `scan_structure`, `extract_metadata`, `analyze_code`, `generate_context`, `refine_context` | Full agent |
| Exploration | `list_key_files`, `read_file_snippet` | Full agent |

## Data Models

### Core Models (models.py)

```
ProjectMetadata
├── name: str
├── path: Path
├── project_type: str | None
├── dependencies: dict[str, str]
├── entry_points: list[str]
├── key_files: list[str]
└── readme_content: str | None

CodeAnalysis
├── architecture_patterns: list[str]
├── coding_conventions: dict[str, str]
├── tech_stack: list[str]
└── insights: str

ContextMetadata (YAML frontmatter)
├── source_repo: str
├── scan_date: datetime
├── user_summary: str
└── model_used: str

ScopedContextMetadata
├── source_repo: str
├── scope_question: str
├── scan_date: datetime
├── model_used: str
├── files_analyzed: int
└── source_context: str | None
```

## LLM Integration

### Provider Layer

```
LLMProvider (ABC)
    │
    └── AnthropicProvider
            ├── generate(prompt, system_prompt) → LLMResponse
            └── generate_structured(prompt, schema) → BaseModel
```

- **Rate limiting**: 0.7 req/sec (InMemoryRateLimiter)
- **Structured output**: Uses Pydantic schemas via `with_structured_output()`
- **Content coercion**: Handles LangChain multi-block responses

### Prompt Templates (llm/prompts.py)

| Template | Used By | Purpose |
|----------|---------|---------|
| `CODE_ANALYSIS_PROMPT` | CodeAnalyzer | Extract architecture, stack, conventions |
| `CONTEXT_GENERATION_PROMPT` | ContextGenerator | Generate final markdown |
| `REFINEMENT_PROMPT` | ContextGenerator.refine | Update existing context |
| `SCOPE_EXPLORATION_PROMPT` | ScopedAnalyzer | Find relevant files |
| `SCOPE_GENERATION_PROMPT` | ScopedGenerator | Generate scoped markdown |

## Middleware

### Budget Tracking

```
BudgetTracker
├── max_tokens: int (default 50,000)
├── max_cost_usd: float (default $5.00)
├── add_usage(prompt_tokens, completion_tokens, operation)
├── is_over_budget() → bool
├── check_budget() → raises BudgetExceededError
└── print_summary()

Cost model:
├── Input: $3.00 / 1M tokens
└── Output: $15.00 / 1M tokens
```

### Human-in-the-Loop

Uses LangGraph `interrupt()` to pause agent execution before expensive operations. Requires a checkpointer for state persistence.

## Security

### Path Traversal Protection

`LocalFileBackend._resolve_safe_path()` prevents directory traversal:
- Resolves all paths to absolute
- Verifies resolved path starts with repo_path
- Rejects symlinks that escape the repository
- Blocks access to files outside repo boundary

### Input Validation

All tool input schemas use Pydantic field validators:
- `repo_path` — Must exist and be a directory
- `context_file_path` — Must exist and be a file
- `start_line` — Must be non-negative
- `num_lines` — Must be positive, capped at 500

## Configuration

Configuration flows from environment variables through `Config.from_env()`:

```
.env file
    │
    ▼
os.environ (via python-dotenv)
    │
    ▼
Config.from_env()
    │
    ▼
Config(BaseModel)
├── llm_provider: str
├── model_name: str
├── api_key: str | None
├── max_retries: int
├── timeout: int
├── max_file_size: int
├── ignored_dirs: list[str]
└── output_dir: Path
```

## Output Structure

```
contexts/
├── {repo-name}/
│   ├── context.md              # Full context
│   ├── scope-{topic}-{ts}.md   # Scoped contexts
│   └── scope-{topic}-{ts}.md
└── .gitkeep
```

## Key Design Decisions

1. **Two execution modes** — Pipeline mode is predictable and cost-controlled (exactly 2 LLM calls). Agent mode is flexible but potentially more expensive.

2. **Backend abstraction** — The `FileBackend` protocol decouples all file I/O from the filesystem. This enables `InMemoryFileBackend` for fast, deterministic tests.

3. **Cost control at multiple levels** — Rate limiting (0.7 req/sec), content size limits (13.5k chars per file, 80k total in exploration, 120k in generation), tree depth limits (6), and optional budget tracking.

4. **Stateful agents** — LangGraph checkpointers enable multi-turn conversations. Refinement reuses the same thread_id as generation, preserving context.

5. **Separation of discovery and analysis** — Scoped context first discovers candidates using keyword search (no LLM), then uses the LLM only to analyze relevant files. This minimizes cost.

6. **GitHub URL support** — `repo_resolver.py` handles both local paths and GitHub URLs via shallow clone to temp directories, with automatic cleanup.
