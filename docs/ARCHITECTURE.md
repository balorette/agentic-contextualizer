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
│  File | Search | Analysis | Repo | Progressive Disclosure   │
├─────────────────────────────────────────────────────────────┤
│               SmartFileAccess (unified layer)               │
│          Composes FileBackend + FileAnalysisBackend          │
├──────────────────────────┬──────────────────────────────────┤
│   File I/O Backend       │   Semantic Analysis Backend      │
│   LocalFileBackend       │   ASTFileAnalysisBackend         │
│   InMemoryFileBackend    │   (Python ast + tree-sitter)     │
├──────────────────────────┴──────────────────────────────────┤
│                  Middleware                                  │
│  TokenBudgetMiddleware | Context Priority | Budget Tracker  │
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
| Progressive Tools | `src/agents/tools/progressive.py` | 0 | Outline/symbol/lines/references tools |
| Backends | `src/agents/backends/` | 0 | Semantic code analysis (AST/tree-sitter) |
| SmartFileAccess | `src/agents/file_access.py` | 0 | Unified file I/O + semantic analysis |
| Middleware | `src/agents/middleware/` | 0 | Budget, HITL, context priority |
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
│  Model: Claude (via SmartFileAccess)     │
│                                          │
│  Discovery Tools:                        │
│  ├── search_for_files                    │
│  ├── grep_in_files                       │
│                                          │
│  Progressive Disclosure Tools:           │
│  ├── get_file_outline    (~500 bytes)    │
│  ├── read_symbol         (~1-2 KB)      │
│  ├── read_lines          (surgical)      │
│  ├── find_references     (~2-3 KB)      │
│  └── read_file           (~8 KB, last)  │
│                                          │
│  Output:                                 │
│  └── generate_scoped_context             │
│                                          │
│  Backend: SmartFileAccess                │
│    ├── LocalFileBackend (I/O)            │
│    └── ASTFileAnalysisBackend (semantic) │
│                                          │
│  Middleware: TokenBudgetMiddleware        │
│    ├── Per-request trimming              │
│    ├── TPM rate limiting                 │
│    ├── Context priority (LOW/MED/HIGH)   │
│    └── Budget tracking (optional)        │
└──────────────────────────────────────────┘
```

**Agent Workflow** (5-step progressive disclosure):

1. **SEARCH** — `search_for_files` + `grep_in_files` to find candidates
2. **OUTLINE** — `get_file_outline` to understand structure (~500 bytes/file vs ~8 KB)
3. **DRILL** — `read_symbol` / `read_lines` to extract specific code
4. **CONNECT** — `find_references` to trace cross-file relationships
5. **GENERATE** — `generate_scoped_context` with paths and insights

**Result**: ~12 KB cumulative context per session (vs ~35 KB with full file reads, ~65% reduction).

## Tools System

### Two-Layer Backend Architecture

File operations use a two-layer abstraction:

**Layer 1: File I/O** (`FileBackend` protocol)

```python
class FileBackend(Protocol):
    @property
    def repo_path(self) -> Path: ...
    def read_file(self, path: str, max_chars: int) -> str | None: ...
    def file_exists(self, path: str) -> bool: ...
    def walk_files(self) -> Iterator[str]: ...
    def search_content(self, pattern: str) -> list[tuple[str, int, str]]: ...
```

- `LocalFileBackend` — Real filesystem with path traversal protection
- `InMemoryFileBackend` — Dict-based fake filesystem for testing

**Layer 2: Semantic Analysis** (`FileAnalysisBackend` protocol)

```python
class FileAnalysisBackend(Protocol):
    def get_outline(self, file_path: str, source: str) -> FileOutline: ...
    def read_symbol(self, file_path: str, symbol_name: str, source: str) -> SymbolDetail | None: ...
    def find_references(self, symbol_name: str, file_backend, scope: str | None) -> list[Reference]: ...
```

- `ASTFileAnalysisBackend` — Python stdlib `ast` + tree-sitter for JS/TS
- LSP backend (future) — Language Server Protocol for richer semantics

**Unified Access** (`SmartFileAccess`)

```python
class SmartFileAccess:
    """Composes FileBackend (I/O) + FileAnalysisBackend (semantic)."""
    def get_outline(self, file_path: str) -> FileOutline | None: ...
    def read_symbol(self, file_path: str, symbol_name: str) -> SymbolDetail | None: ...
    def read_lines(self, file_path: str, start: int, end: int) -> str | None: ...
    def find_references(self, symbol_name: str, scope: str | None) -> list[Reference]: ...
    def read_file(self, file_path: str, max_chars: int) -> str | None: ...
```

### Tool Categories

| Category | Tools | Used By |
|----------|-------|---------|
| Discovery | `search_for_files`, `grep_in_files` | Scoped agent |
| Progressive | `get_file_outline`, `read_symbol`, `read_lines`, `find_references`, `read_file` | Scoped agent |
| Repository | `scan_structure`, `extract_metadata`, `analyze_code`, `generate_context`, `refine_context` | Full agent |
| Exploration | `list_key_files`, `read_file_snippet` | Full agent |
| Output | `generate_scoped_context` | Scoped agent |

### Progressive Disclosure Tool Hierarchy

Tools are ordered from cheapest to most expensive. The agent is instructed to use the cheapest tool that satisfies its needs:

| Tool | Output Size | Purpose |
|------|------------|---------|
| `get_file_outline` | ~500 bytes | Imports, symbol names, signatures, line numbers (no bodies) |
| `read_symbol` | ~1-2 KB | Extract a single function/class/method body |
| `read_lines` | variable | Surgical line range extraction |
| `find_references` | ~2-3 KB | Where a symbol is used across the codebase |
| `read_file` | ~8 KB max | Full file content (last resort) |

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

### Semantic Analysis Models (backends/models.py)

```
SymbolInfo (outline element)
├── name: str
├── kind: str               # "function", "class", "method", "variable"
├── line: int
├── line_end: int
├── signature: str
├── children: list[SymbolInfo]
├── decorators: list[str]
└── docstring: str | None

SymbolDetail (extends SymbolInfo)
├── body: str               # Full source code
├── parent: str | None      # Containing class name
└── char_count: int

FileOutline
├── path: str
├── language: str
├── imports: list[str]
├── symbols: list[SymbolInfo]
└── line_count: int

Reference
├── path: str
├── line: int
└── context: str            # The source line containing the reference
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

### TokenBudgetMiddleware

Three-layer cost control applied during agent execution:

```
TokenBudgetMiddleware
├── before_model()
│   ├── Trim old messages to max_input_tokens (~20K)
│   └── TPM rate limiting via TPMThrottle.wait_if_needed()
├── after_model()
│   ├── Record actual token usage to throttle
│   └── Update BudgetTracker (if configured)
└── wrap_tool_call()
    ├── Priority-aware truncation of tool output
    ├── LOW priority tools: stricter limit (max_chars / 2)
    └── HIGH/MEDIUM priority tools: standard limit
```

### Context Priority Tagging

Tool results are tagged with priority levels for smart trimming. When the conversation grows too large, LOW priority results (verbose full file reads) are evicted first.

| Priority | Tools | Rationale |
|----------|-------|-----------|
| HIGH (3) | `get_file_outline`, `search_for_files`, `grep_in_files`, `generate_scoped_context` | Navigational — agent needs these for orientation |
| MEDIUM (2) | `read_symbol`, `read_lines`, `find_references` | Targeted — useful but replaceable |
| LOW (1) | `read_file` | Verbose — evict first during trimming |

### Budget Tracking

```
BudgetTracker
├── max_tokens: int (default 30,000 for scoped, 50,000 for full)
├── max_cost_usd: float (default $2.00 for scoped, $5.00 for full)
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

2. **Two-layer backend abstraction** — `FileBackend` (I/O) and `FileAnalysisBackend` (semantic) are separate protocols, composed by `SmartFileAccess`. This enables `InMemoryFileBackend` for fast tests while keeping language-specific parsing pluggable.

3. **Progressive disclosure** — Agent tools are ordered cheapest-first (outline → symbol → lines → references → full file). The agent system prompt enforces "never read a code file without outlining first". This reduces context bloat from ~35 KB to ~12 KB per session (~65% reduction).

4. **Priority-aware context trimming** — Tool results are tagged HIGH/MEDIUM/LOW. When trimming is needed, LOW priority (full file reads) gets evicted first, preserving navigational context the agent needs for orientation.

5. **Cost control at multiple levels** — TPM rate limiting, per-request message trimming, priority-aware tool output truncation, tree depth limits (6), and optional budget tracking.

6. **Stateful agents** — LangGraph checkpointers enable multi-turn conversations. Refinement reuses the same thread_id as generation, preserving context.

7. **Separation of discovery and analysis** — Scoped context first discovers candidates using keyword search (no LLM), then uses the LLM only to analyze relevant files. This minimizes cost.

8. **GitHub URL support** — `repo_resolver.py` handles both local paths and GitHub URLs via shallow clone to temp directories, with automatic cleanup.
