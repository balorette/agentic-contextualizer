# Progressive Disclosure Tools — Design

**Date**: 2026-02-15
**Status**: Approved
**Branch**: TBD

## Problem

The scoped agent's tool results accumulate in the LangChain conversation history and are re-sent with every subsequent LLM call. With 10-20 file reads at 8KB each, context balloons to 35-40KB per session. This drives up token costs through litellm and will degrade quality and cause context loss as the tool scales to larger repos.

## Decision

Approach C: Hybrid — AST-first with optional LSP enhancement.

Design the tool interface around progressive disclosure (tree → outline → symbol → full file). Implement the base layer with AST/static analysis. Make LSP an optional backend that enhances the same tools without changing the agent-facing interface.

## Architecture

### SmartFileAccess Layer

Unified layer serving both agent mode and pipeline mode. Replaces the current scattered file-reading logic in `scoped_analyzer.py`, `scoped_generator.py`, and agent tools.

```
┌──────────────────────────────────────────────┐
│              SmartFileAccess                  │
│                                              │
│  get_tree(path) → FileTree                   │
│  get_outline(file) → FileOutline             │
│  read_symbol(file, name) → SymbolDetail      │
│  find_references(name, scope?) → [Reference] │
│  read_lines(file, start, end) → str          │
│  read_file(file) → str                       │
│                                              │
│  ┌─────────────┐  ┌───────────────────────┐  │
│  │ AST Backend │  │ LSP Backend (future)  │  │
│  │  (default)  │  │  (pluggable)          │  │
│  └─────────────┘  └───────────────────────┘  │
└──────────────────────────────────────────────┘
```

Composes two concerns:
- `LocalFileBackend` — raw file I/O (read_file, read_lines, get_tree). Unchanged.
- `FileAnalysisBackend` — semantic operations (get_outline, read_symbol, find_references). Pluggable.

### Backend Protocol

```python
class FileAnalysisBackend(Protocol):
    def get_outline(self, file_path: str) -> list[SymbolInfo]: ...
    def read_symbol(self, file_path: str, symbol_name: str) -> SymbolDetail | None: ...
    def find_references(self, symbol_name: str, scope: str | None) -> list[Reference]: ...
```

LSP backend implements the same protocol. SmartFileAccess tries LSP first, falls back to AST on failure or absence.

## Redesigned Agent Tools

### Tool Set

| Tool | Purpose | Typical Size |
|------|---------|-------------|
| `search_for_files` | Keyword file search (kept) | ~7.5 KB max |
| `grep_in_files` | Regex pattern search (kept) | ~2-3 KB max |
| `get_file_outline` | Symbols, signatures, imports — no bodies | ~500 bytes |
| `read_symbol` | Extract a specific function/method body | ~500-2000 bytes |
| `read_lines` | Read exact line range | variable, surgical |
| `find_references` | Cross-file symbol usage | ~2-3 KB |
| `read_file` | Full file content — last resort | ~8 KB max |
| `generate_scoped_context` | Final output generation (kept) | reads internally |

### Tools Removed

- `find_code_definitions` — replaced by `get_file_outline` (superset)
- `extract_file_imports` — folded into `get_file_outline`

### Expected Token Reduction

Current typical session: ~35-37 KB cumulative context at generation time.
New typical session: ~12 KB cumulative context at generation time.
**~65% reduction.**

### Agent Prompt — Progressive Flow

```
Step 1: SEARCH — search_for_files, grep_in_files (find candidates)
Step 2: OUTLINE — get_file_outline on top candidates (understand structure without reading)
Step 3: DRILL — read_symbol for specific functions, read_lines for targeted ranges
Step 4: CONNECT — find_references to trace cross-file relationships
Step 5: GENERATE — call generate_scoped_context when confident

COST HIERARCHY (cheapest → most expensive):
  get_file_outline  (~500 bytes)  ← prefer this
  read_symbol       (~1-2 KB)    ← when you need a specific function
  read_lines        (variable)   ← when you need an exact range
  find_references   (~2-3 KB)    ← for cross-file understanding
  read_file         (~8 KB)      ← LAST RESORT, only for config/non-code files

RULE: Never call read_file on a code file without first calling get_file_outline.
```

## Data Models

```python
@dataclass
class SymbolInfo:
    name: str
    kind: str              # "function", "class", "method", "variable"
    line: int
    line_end: int
    signature: str
    children: list[SymbolInfo]
    decorators: list[str]
    docstring: str | None  # first line only

@dataclass
class SymbolDetail(SymbolInfo):
    body: str              # actual source code
    parent: str | None     # enclosing class name if method
    char_count: int

@dataclass
class FileOutline:
    path: str
    language: str
    imports: list[str]
    symbols: list[SymbolInfo]
    line_count: int

@dataclass
class Reference:
    path: str
    line: int
    context: str           # the single line containing the reference
```

## AST Backend

**Python**: stdlib `ast` module. Already proven in codebase via `find_code_definitions`.

**JS/TS**: `tree-sitter` with Python bindings (`tree-sitter-javascript`, `tree-sitter-typescript`). Handles JSX, decorators, arrow functions, and all edge cases regex misses. Also makes adding future languages trivial.

### File Layout

```
src/agents/
    backends/
        __init__.py
        protocol.py              # FileAnalysisBackend Protocol
        ast_backend.py           # ASTFileAnalysisBackend
        parsers/
            __init__.py
            python_parser.py     # stdlib ast
            ts_parser.py         # tree-sitter
    file_access.py               # SmartFileAccess (unified layer)
    tools/
        file.py                  # updated: read_file, read_lines (new)
        search.py                # updated: grep + find_references (new)
        analysis.py              # updated: get_file_outline (replaces old tools)
```

### LSP Plug-in Point (Future)

Add `src/agents/backends/lsp_backend.py` implementing the same `FileAnalysisBackend` protocol. SmartFileAccess tries LSP first, falls back to AST automatically:

```python
class SmartFileAccess:
    def __init__(self, file_backend, analysis_backend, lsp_backend=None):
        self._files = file_backend
        self._analysis = analysis_backend
        self._lsp = lsp_backend

    def get_outline(self, path):
        if self._lsp:
            try:
                return self._lsp.get_outline(path)
            except LSPError:
                pass
        return self._analysis.get_outline(path)
```

## Conversation Context Management

Two-layer approach to keep accumulated tool results from bloating the conversation.

### Layer 1 — Proactive Summarization

- `read_file` results get a `_summary` field injected (the file's outline).
- For messages older than N turns, middleware replaces the full `content` field with `_summary`.
- Outlines, search results, and symbol reads are already compact — pass through unchanged.

### Layer 2 — Priority-Based Smart Trimming

Tag each tool result message with priority:

| Priority | Tool Results | Rationale |
|----------|-------------|-----------|
| `high` | search results, outlines | Navigational — agent needs these to know where it's been |
| `medium` | symbol reads, references | Targeted data — useful but replaceable |
| `low` | full file reads | Most verbose, least reusable |

When trimming, evict `low` first, then `medium`, preserving `high` as long as possible. System message and most recent 2-3 turns always preserved.

## Pipeline Mode Integration

Pipeline mode benefits from the same SmartFileAccess layer:

**Exploration phase** (currently reads up to 12 full files at 8K each = 96KB):
- Call `get_outline()` on all 12 candidates (~6KB total)
- LLM sees the shape of all files, responds with which symbols to examine
- Pipeline reads those symbols with `read_symbol()` (~15-25KB)
- Result: ~30KB into the exploration prompt vs ~96KB currently
- Adds one extra LLM round-trip, but each call is significantly cheaper

**Generation phase**:
- Uses `SmartFileAccess` instead of raw `backend.read_file()`
- Can include outlines for supporting files, full content only for primary files

## Dependencies

**New**: `tree-sitter`, `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`

## Not In Scope

- LSP backend implementation (future iteration)
- Language parsers beyond Python/JS/TS
- Changes to full context pipeline (only scoped pipeline affected)
