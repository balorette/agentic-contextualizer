# Developer Guide

> Agentic Contextualizer v0.1.0

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An Anthropic API key

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd agentic-contextualizer

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Project Structure

```
src/agents/
├── main.py                    # CLI entry point (Click commands)
├── config.py                  # Config from env vars (Pydantic)
├── models.py                  # Core data models
├── factory.py                 # LangChain agent factory
├── file_access.py             # SmartFileAccess (unified I/O + semantic)
├── memory.py                  # Checkpointing and state
├── observability.py           # LangSmith tracing
├── streaming.py               # Real-time streaming output
├── repo_resolver.py           # GitHub URL / local path handling
│
├── backends/                  # Semantic code analysis
│   ├── protocol.py            #   FileAnalysisBackend protocol
│   ├── models.py              #   SymbolInfo, SymbolDetail, FileOutline, Reference
│   ├── ast_backend.py         #   AST-based implementation
│   └── parsers/
│       ├── python_parser.py   #   Python stdlib ast parser
│       └── ts_parser.py       #   JS/TS tree-sitter parser
│
├── scanner/                   # Phase 1: Static analysis (no LLM)
│   ├── structure.py           #   File tree walking
│   └── metadata.py            #   Config file parsing
│
├── analyzer/                  # Phase 2: LLM code analysis
│   └── code_analyzer.py       #   Architecture/pattern detection
│
├── generator/                 # Phase 3: Context generation
│   └── context_generator.py   #   Markdown + YAML output
│
├── scoper/                    # Scoped context pipeline
│   ├── discovery.py           #   Keyword extraction + file search
│   ├── scoped_analyzer.py     #   LLM-guided exploration
│   ├── scoped_generator.py    #   Scoped markdown generation
│   └── agent.py               #   Scoped agent factory
│
├── llm/                       # LLM abstraction
│   ├── provider.py            #   AnthropicProvider
│   ├── chat_model_factory.py  #   LangChain model factory
│   ├── litellm_provider.py    #   LiteLLM provider support
│   ├── prompts.py             #   Prompt templates
│   ├── rate_limiting.py       #   TPM throttling
│   └── token_estimator.py     #   Token counting utilities
│
├── middleware/                 # Cross-cutting concerns
│   ├── token_budget.py        #   TokenBudgetMiddleware (trimming + TPM)
│   ├── context_priority.py    #   Priority tagging for tool results
│   ├── budget.py              #   Token/cost budget tracking
│   └── human_in_the_loop.py   #   Approval gates
│
└── tools/                     # LangChain tool definitions
    ├── file.py                #   read_file, search_for_files
    ├── search.py              #   grep_pattern, find_definitions
    ├── analysis.py            #   extract_imports
    ├── progressive.py         #   Progressive disclosure tools
    ├── repository_tools.py    #   Pipeline step wrappers
    ├── exploration_tools.py   #   list_key_files, read_file_snippet
    ├── schemas.py             #   All Pydantic I/O schemas
    └── backends/              #   File I/O abstraction
        ├── protocol.py        #     FileBackend protocol
        ├── local.py           #     Real filesystem
        └── memory.py          #     In-memory (for tests)
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/agents --cov-report=term-missing

# Specific test file
pytest tests/test_scanner.py

# Specific test
pytest tests/test_scanner.py::test_scan_basic -v

# Security tests
pytest tests/scoper/test_path_traversal.py
```

## Code Quality

```bash
# Format
black src/ tests/ --line-length 100

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Adding a New Tool

Tools follow a factory pattern with backend abstraction. Here's the process:

### 1. Define schemas in `tools/schemas.py`

```python
class MyToolInput(BaseModel):
    """Input schema."""
    param: str = Field(description="What this parameter does")

class MyToolOutput(BaseModel):
    """Output schema."""
    result: str = Field(description="What this returns")
    error: str | None = Field(default=None)
```

### 2. Implement the tool function

```python
# tools/my_tool.py
from .backends.protocol import FileBackend
from .schemas import MyToolOutput

def my_tool_function(backend: FileBackend, param: str) -> MyToolOutput:
    """Tool implementation using backend for file access."""
    # Use backend.read_file(), backend.walk_files(), etc.
    return MyToolOutput(result="...")
```

### 3. Create the LangChain tool factory

```python
def create_my_tools(backend: FileBackend) -> list:
    """Create LangChain tools bound to a backend."""
    from langchain_core.tools import tool

    @tool
    def my_tool(param: str) -> dict:
        """Description shown to the LLM agent."""
        result = my_tool_function(backend, param)
        return result.model_dump()

    return [my_tool]
```

### 4. Register in agent factory

Add the tool to `factory.py` or `scoper/agent.py` depending on which agent needs it.

### 5. Write tests using InMemoryFileBackend

```python
from agents.tools.backends.memory import InMemoryFileBackend

def test_my_tool():
    backend = InMemoryFileBackend("/fake/repo")
    backend.add_file("src/main.py", "print('hello')")

    result = my_tool_function(backend, param="test")
    assert result.error is None
```

## Adding a New Pipeline Phase

1. Create a new module under `src/agents/` (e.g., `src/agents/my_phase/`)
2. Define a class with a clear `analyze()` or `process()` method
3. Accept an `LLMProvider` if LLM calls are needed
4. Wire it into `main.py`'s pipeline functions
5. Add corresponding tool wrapper in `tools/repository_tools.py` for agent mode

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `LLM_PROVIDER` | No | `anthropic` | LLM provider |
| `MODEL_NAME` | No | `claude-3-5-sonnet-20241022` | Model name |
| `MAX_FILE_SIZE` | No | `1000000` | Max file size in bytes |
| `LLM_MAX_RETRIES` | No | `3` | LLM retry count |
| `LLM_TIMEOUT` | No | `60` | LLM timeout (seconds) |
| `IGNORED_DIRS` | No | — | Additional dirs to ignore (comma-separated) |
| `OUTPUT_DIR` | No | `contexts` | Output directory |
| `LANGSMITH_API_KEY` | No | — | LangSmith tracing key |
| `LANGSMITH_PROJECT` | No | — | LangSmith project name |

## Testing Strategy

### Unit Tests

Each module has a corresponding test file. Tests use `InMemoryFileBackend` to avoid filesystem dependencies:

```
tests/
├── test_scanner.py          # StructureScanner, MetadataExtractor
├── test_analyzer.py         # CodeAnalyzer (mocked LLM)
├── test_generator.py        # ContextGenerator (mocked LLM)
├── test_config.py           # Config loading
├── test_models.py           # Pydantic models
├── test_llm.py              # LLM provider
├── test_cli.py              # CLI commands
├── test_repo_resolver.py    # URL/path resolution
├── test_search_tools.py     # grep, find_definitions
│
├── scoper/
│   ├── test_discovery.py    # Keyword extraction, file search
│   ├── test_scoped_analyzer.py
│   ├── test_scoped_generator.py
│   ├── test_scoped_agent.py
│   ├── test_backends.py     # File backend implementations
│   ├── test_tools.py        # Scoped tools
│   └── test_path_traversal.py  # Security tests
│
├── agents/
│   ├── test_agent_integration.py
│   ├── test_budget.py
│   ├── test_file_access.py           # SmartFileAccess composition
│   ├── backends/
│   │   ├── test_models.py            # SymbolInfo, FileOutline, Reference
│   │   ├── test_protocol.py          # FileAnalysisBackend protocol
│   │   ├── test_ast_backend.py       # ASTFileAnalysisBackend
│   │   └── parsers/
│   │       ├── test_python_parser.py  # Python symbol extraction
│   │       └── test_ts_parser.py      # JS/TS symbol extraction
│   ├── middleware/
│   │   ├── test_token_budget_middleware.py  # Trimming + priority-aware truncation
│   │   └── test_context_priority.py        # Priority tagging
│   └── tools/
│       ├── test_exploration_tools.py
│       ├── test_repository_tools.py
│       └── test_progressive_tools.py  # Progressive disclosure tools
│
└── integration/
    ├── test_integration.py            # End-to-end full pipeline
    ├── test_scope_integration.py      # End-to-end scoped pipeline
    └── test_progressive_disclosure.py # Progressive flow (<8KB context)
```

### Integration Tests

- `test_integration.py` — End-to-end pipeline test
- `test_scope_integration.py` — End-to-end scoped pipeline
- `test_agent_integration.py` — Agent mode with mocked LLM
- `test_progressive_disclosure.py` — Simulates agent exploration with in-memory backend, verifies progressive flow produces <8 KB context (vs 35+ KB with full file reads)

### Security Tests

`test_path_traversal.py` verifies that `LocalFileBackend` blocks:
- `../` traversal attempts
- Symlinks escaping the repo
- Access to files outside repo boundary

## Debugging

### Enable debug output

```bash
python -m agents.main generate /repo -s "desc" --mode agent --debug
```

### LangSmith tracing

Set `LANGSMITH_API_KEY` in `.env` to automatically enable tracing. View traces at [smith.langchain.com](https://smith.langchain.com).

### Streaming output

Use `--stream` flag for real-time agent feedback:
```bash
python -m agents.main generate /repo -s "desc" --mode agent --stream
```

Rich formatting is used in TTY terminals; plain text in non-TTY (e.g., piped output).

## Key Constants

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `MAX_TREE_DEPTH` | `scanner/structure.py` | 6 | Max directory nesting |
| `MAX_CHILDREN_PER_DIR` | `scanner/structure.py` | 200 | Max entries per directory |
| `MAX_FILES_FOR_ANALYSIS` | `analyzer/code_analyzer.py` | 20 | Files read for analysis |
| `MAX_FILE_CHARS` | `analyzer/code_analyzer.py` | 20,000 | Chars per file in analysis |
| `DEFAULT_MAX_CHARS` | `tools/file.py` | 13,500 | Default file read limit |
| `MAX_EXPLORATION_ROUNDS` | `scoper/scoped_analyzer.py` | 2 | Scoped exploration iterations |
| `MAX_FILE_CONTENT_CHARS` | `scoper/scoped_analyzer.py` | 8,000 | Chars per file in exploration |
| `MAX_TOTAL_CONTENT_CHARS` | `scoper/scoped_analyzer.py` | 80,000 | Total content in exploration |
| `MAX_FILES_IN_PROMPT` | `scoper/scoped_analyzer.py` | 12 | Files per exploration prompt |
| Rate limit | `llm/provider.py` | 0.7 req/s | LLM request throttle |
