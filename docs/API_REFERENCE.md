# API Reference

> Agentic Contextualizer v0.1.0

## Table of Contents

- [CLI Commands](#cli-commands)
- [Python API](#python-api)
  - [Config](#config)
  - [Models](#models)
  - [Scanner](#scanner)
  - [Analyzer](#analyzer)
  - [Generator](#generator)
  - [Scoper](#scoper)
  - [LLM Provider](#llm-provider)
  - [Agent Factory](#agent-factory)
  - [Tools](#tools)
  - [Middleware](#middleware)
  - [Memory](#memory)
  - [Repo Resolver](#repo-resolver)

---

## CLI Commands

Entry point: `python -m agents.main`

### `generate`

Generate a full context file for a repository.

```
python -m agents.main generate SOURCE [OPTIONS]
```

| Argument/Option | Type | Required | Default | Description |
|----------------|------|----------|---------|-------------|
| `SOURCE` | string | Yes | — | Local path or GitHub URL |
| `--summary, -s` | string | Yes | — | Brief project description |
| `--output, -o` | path | No | `contexts/{name}/context.md` | Custom output path |
| `--mode, -m` | choice | No | `pipeline` | `pipeline` or `agent` |
| `--debug` | flag | No | `false` | Enable debug output |
| `--stream` | flag | No | `false` | Enable streaming (agent mode) |

**Examples:**
```bash
# Local repository
python -m agents.main generate /path/to/repo -s "FastAPI REST API"

# GitHub URL
python -m agents.main generate https://github.com/owner/repo -s "Web app"

# Agent mode with streaming
python -m agents.main generate /path/to/repo -s "API" --mode agent --stream
```

**Exit codes:** `0` success, `1` error

---

### `refine`

Update an existing context file with new information.

```
python -m agents.main refine CONTEXT_FILE [OPTIONS]
```

| Argument/Option | Type | Required | Default | Description |
|----------------|------|----------|---------|-------------|
| `CONTEXT_FILE` | path | Yes | — | Path to existing context.md |
| `--request, -r` | string | Yes | — | What to change/add |
| `--mode, -m` | choice | No | `pipeline` | `pipeline` or `agent` |
| `--debug` | flag | No | `false` | Enable debug output |
| `--stream` | flag | No | `false` | Enable streaming (agent mode) |

**Example:**
```bash
python -m agents.main refine contexts/myapp/context.md \
  -r "Add more details about the authentication flow"
```

---

### `scope`

Generate focused context for a specific question or topic.

```
python -m agents.main scope SOURCE [OPTIONS]
```

| Argument/Option | Type | Required | Default | Description |
|----------------|------|----------|---------|-------------|
| `SOURCE` | string | Yes | — | Repo path, context.md file, or GitHub URL |
| `--question, -q` | string | Yes | — | Question/topic to scope |
| `--output, -o` | path | No | auto-generated | Custom output path |
| `--mode, -m` | choice | No | `pipeline` | `pipeline` or `agent` |
| `--debug` | flag | No | `false` | Enable debug output |
| `--stream` | flag | No | `false` | Enable streaming (agent mode) |

**Examples:**
```bash
# Scope from repository
python -m agents.main scope /path/to/repo -q "authentication flow"

# Scope from existing context file
python -m agents.main scope contexts/repo/context.md -q "API endpoints"

# Scope from GitHub URL
python -m agents.main scope https://github.com/owner/repo -q "auth"

# Agent mode
python -m agents.main scope /path/to/repo -q "database models" --mode agent --stream
```

---

## Python API

### Config

**Module:** `src/agents/config.py`

#### `Config`

Pydantic model for application configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_provider` | `str` | `"anthropic"` | LLM provider name |
| `model_name` | `str` | `"claude-3-5-sonnet-20241022"` | Model identifier |
| `api_key` | `str \| None` | `None` | API key |
| `max_retries` | `int` | `3` | Max retry attempts |
| `timeout` | `int` | `60` | Request timeout (seconds) |
| `max_file_size` | `int` | `1_000_000` | Max file size (bytes) |
| `ignored_dirs` | `list[str]` | See below | Directories to skip |
| `output_dir` | `Path` | `Path("contexts")` | Output directory |

**Default ignored directories:** `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`, `.pytest_cache`

```python
from agents.config import Config

# From environment
config = Config.from_env()

# Manual
config = Config(api_key="sk-...", model_name="claude-3-5-sonnet-20241022")
```

#### `Config.from_env() → Config`

Loads configuration from environment variables:

| Env Variable | Config Field | Default |
|-------------|-------------|---------|
| `LLM_PROVIDER` | `llm_provider` | `"anthropic"` |
| `MODEL_NAME` | `model_name` | `"claude-3-5-sonnet-20241022"` |
| `ANTHROPIC_API_KEY` | `api_key` | `None` |
| `LLM_MAX_RETRIES` | `max_retries` | `3` |
| `LLM_TIMEOUT` | `timeout` | `60` |
| `MAX_FILE_SIZE` | `max_file_size` | `1_000_000` |
| `IGNORED_DIRS` | `ignored_dirs` | (appended to defaults) |
| `OUTPUT_DIR` | `output_dir` | `"contexts"` |

---

### Models

**Module:** `src/agents/models.py`

#### `ProjectMetadata`

Repository metadata extracted during scanning.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Repository name |
| `path` | `Path` | Filesystem path |
| `project_type` | `str \| None` | `"python"`, `"node"`, `"rust"`, etc. |
| `dependencies` | `dict[str, str]` | Package → version mapping |
| `entry_points` | `list[str]` | Main entry files |
| `key_files` | `list[str]` | Important config/metadata files |
| `readme_content` | `str \| None` | README file content |

#### `CodeAnalysis`

Results from LLM-based code analysis.

| Field | Type | Description |
|-------|------|-------------|
| `architecture_patterns` | `list[str]` | e.g., `["MVC", "Microservices"]` |
| `coding_conventions` | `dict[str, str]` | Convention name → description |
| `tech_stack` | `list[str]` | e.g., `["FastAPI", "PostgreSQL"]` |
| `insights` | `str` | Free-form analysis text |

#### `ContextMetadata`

YAML frontmatter for full context files.

| Field | Type | Description |
|-------|------|-------------|
| `source_repo` | `str` | Repository path |
| `scan_date` | `datetime` | UTC timestamp |
| `user_summary` | `str` | User-provided description |
| `model_used` | `str` | LLM model identifier |

#### `ScopedContextMetadata`

YAML frontmatter for scoped context files.

| Field | Type | Description |
|-------|------|-------------|
| `source_repo` | `str` | Repository path |
| `scope_question` | `str` | The scoping question |
| `scan_date` | `datetime` | UTC timestamp |
| `model_used` | `str` | LLM model identifier |
| `files_analyzed` | `int` | Number of files analyzed |
| `source_context` | `str \| None` | Parent context file path |

---

### Scanner

**Module:** `src/agents/scanner/`

#### `StructureScanner`

Walks the repository file tree.

```python
from agents.scanner.structure import StructureScanner
from agents.config import Config

scanner = StructureScanner(Config.from_env())
result = scanner.scan(Path("/path/to/repo"))
```

**`scan(repo_path: Path) → dict`**

Returns:
```python
{
    "tree": {...},          # Nested directory tree
    "all_files": [...],     # Flat file list
    "total_files": int,
    "total_dirs": int
}
```

**Constants:**
- `MAX_TREE_DEPTH = 6`
- `MAX_CHILDREN_PER_DIR = 200`

#### `MetadataExtractor`

Extracts project metadata from config files.

```python
from agents.scanner.metadata import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract(Path("/path/to/repo"))
# Returns: ProjectMetadata
```

**Supported project types:**
- Python (`pyproject.toml`, `setup.py`)
- Node.js (`package.json`)
- Rust (`Cargo.toml`)
- Go (`go.mod`)
- Java (`pom.xml`, `build.gradle`)

---

### Analyzer

**Module:** `src/agents/analyzer/code_analyzer.py`

#### `CodeAnalyzer`

LLM-based code analysis.

```python
from agents.analyzer.code_analyzer import CodeAnalyzer

analyzer = CodeAnalyzer(llm_provider)
analysis = analyzer.analyze(repo_path, metadata, file_tree, user_summary)
# Returns: CodeAnalysis
```

**`analyze(repo_path, metadata, file_tree, user_summary) → CodeAnalysis`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `repo_path` | `Path` | Repository path |
| `metadata` | `ProjectMetadata` | From MetadataExtractor |
| `file_tree` | `dict` | From StructureScanner |
| `user_summary` | `str` | User description |

**Constants:**
- `MAX_FILES_FOR_ANALYSIS = 20`
- `MAX_FILE_CHARS = 20_000`

---

### Generator

**Module:** `src/agents/generator/context_generator.py`

#### `ContextGenerator`

Generates and refines context markdown files.

```python
from agents.generator.context_generator import ContextGenerator

generator = ContextGenerator(llm_provider, output_dir=Path("contexts"))
```

**`generate(metadata, analysis, user_summary, model_name) → Path`**

Generates a full context file. Returns the output path.

**`refine(context_path, request) → Path`**

Updates an existing context file. Returns the updated path.

---

### Scoper

**Module:** `src/agents/scoper/`

#### Discovery Functions

```python
from agents.scoper.discovery import extract_keywords, search_relevant_files
```

**`extract_keywords(question: str) → list[str]`**

Extracts meaningful keywords from a question. Filters 150+ English stopwords, enforces minimum length 3.

**`search_relevant_files(repo_path: Path, keywords: list[str]) → list[dict]`**

Searches for files matching keywords. Returns scored results:
```python
[
    {"path": "src/auth.py", "match_type": "filename", "score": 4},
    {"path": "src/utils.py", "match_type": "content", "score": 2},
]
```

Filename matches score 2x higher than content matches. Skips files >500KB.

#### `ScopedAnalyzer`

LLM-guided file exploration.

```python
from agents.scoper.scoped_analyzer import ScopedAnalyzer

analyzer = ScopedAnalyzer(llm_provider)
result = analyzer.analyze(repo_path, question, candidate_files, file_tree)
```

Returns:
```python
{
    "relevant_files": [...],       # Files determined relevant
    "insights": "...",             # Analysis text
    "code_references": [...]       # CodeReference objects
}
```

**Constants:**
- `MAX_EXPLORATION_ROUNDS = 2`
- `MAX_FILE_CONTENT_CHARS = 8_000`
- `MAX_TOTAL_CONTENT_CHARS = 80_000`
- `MAX_FILES_IN_PROMPT = 12`

#### `ScopedGenerator`

Generates scoped context markdown.

```python
from agents.scoper.scoped_generator import ScopedGenerator

generator = ScopedGenerator(llm_provider, output_dir=Path("contexts"))
path = generator.generate(
    repo_name="my-app",
    question="authentication flow",
    relevant_files=[...],
    insights="...",
    model_name="claude-3-5-sonnet-20241022",
    source_repo="/path/to/repo",
)
```

#### `create_scoped_agent()`

Creates a LangChain agent with scoped tools.

```python
from agents.scoper.agent import create_scoped_agent

agent = create_scoped_agent(
    repo_path=Path("/path/to/repo"),
    model_name="anthropic:claude-3-5-sonnet-20241022",
    checkpointer=checkpointer,
    output_dir=Path("contexts"),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Generate scoped context answering: auth flow"}]
})
```

---

### LLM Provider

**Module:** `src/agents/llm/provider.py`

#### `LLMResponse`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Response text |
| `model` | `str` | Model that produced response |
| `tokens_used` | `int` | Total tokens consumed |

#### `AnthropicProvider`

```python
from agents.llm.provider import AnthropicProvider

llm = AnthropicProvider(model_name="claude-3-5-sonnet-20241022", api_key="sk-...")
```

**`generate(prompt: str, system_prompt: str = "") → LLMResponse`**

Basic text generation.

**`generate_structured(prompt: str, output_schema: type[BaseModel], system_prompt: str = "") → BaseModel`**

Structured output using Pydantic schema validation.

**Rate limiting:** 0.7 requests/second via `InMemoryRateLimiter`.

---

### Agent Factory

**Module:** `src/agents/factory.py`

#### `create_contextualizer_agent()`

```python
from agents.factory import create_contextualizer_agent

agent = create_contextualizer_agent(
    model_name="anthropic:claude-3-5-sonnet-20241022",
    checkpointer=None,
    debug=False,
)
```

Returns a compiled LangGraph `StateGraph` agent.

#### `create_contextualizer_agent_with_budget()`

```python
from agents.factory import create_contextualizer_agent_with_budget

agent, tracker = create_contextualizer_agent_with_budget(
    max_tokens=50_000,
    max_cost_usd=5.0,
)
```

Returns `(agent, BudgetTracker)`.

#### `create_contextualizer_agent_with_checkpointer()`

```python
from agents.factory import create_contextualizer_agent_with_checkpointer
from agents.memory import create_checkpointer

agent = create_contextualizer_agent_with_checkpointer(
    checkpointer=create_checkpointer(),  # Required
)
```

Raises `ValueError` if `checkpointer` is `None`.

---

### Tools

**Module:** `src/agents/tools/`

#### File Tools

| Tool | Parameters | Returns | Description |
|------|-----------|---------|-------------|
| `read_file` | `path: str, max_chars: int = 13500` | `ReadFileOutput` | Read file with truncation |
| `search_for_files` | `keywords: list[str]` | `SearchFilesOutput` | Search by filename and content |

#### Search Tools

| Tool | Parameters | Returns | Description |
|------|-----------|---------|-------------|
| `grep_in_files` | `pattern: str, file_pattern: str \| None` | `GrepOutput` | Regex search across files |
| `find_code_definitions` | `name: str` | `FindDefinitionsOutput` | Find function/class definitions |

#### Analysis Tools

| Tool | Parameters | Returns | Description |
|------|-----------|---------|-------------|
| `extract_file_imports` | `path: str` | `ExtractImportsOutput` | Extract imports (Python/JS/TS) |

#### Repository Tools

| Tool | Parameters | Returns | Description |
|------|-----------|---------|-------------|
| `scan_structure` | `repo_path: str` | `ScanStructureOutput` | Scan repository file tree |
| `extract_metadata` | `repo_path: str` | `ExtractMetadataOutput` | Extract project metadata |
| `analyze_code` | `repo_path, user_summary, metadata_dict, file_tree` | `AnalyzeCodeOutput` | LLM code analysis |
| `generate_context` | `repo_path, user_summary, metadata_dict, analysis_dict` | `GenerateContextOutput` | Generate context file |
| `refine_context` | `context_file_path, refinement_request` | `RefineContextOutput` | Refine existing context |

#### Exploration Tools

| Tool | Parameters | Returns | Description |
|------|-----------|---------|-------------|
| `list_key_files` | `file_tree: dict` | `ListKeyFilesOutput` | List important files |
| `read_file_snippet` | `file_path, start_line, num_lines` | `ReadFileSnippetOutput` | Read line range |

---

### Middleware

**Module:** `src/agents/middleware/`

#### `BudgetTracker`

```python
from agents.middleware.budget import BudgetTracker

tracker = BudgetTracker(max_tokens=50_000, max_cost_usd=5.0)
tracker.add_usage(prompt_tokens=1000, completion_tokens=500, operation="analyze")

tracker.total_tokens      # 1500
tracker.total_cost        # USD estimate
tracker.is_over_budget()  # bool
tracker.print_summary()   # Pretty-print
```

**Cost model:**
- Input tokens: $3.00 / 1M tokens
- Output tokens: $15.00 / 1M tokens

#### `extract_token_usage_from_response(message) → TokenUsage | None`

Extracts token counts from LangChain message metadata.

---

### Memory

**Module:** `src/agents/memory.py`

```python
from agents.memory import create_checkpointer, create_agent_config, generate_thread_id

checkpointer = create_checkpointer()
config = create_agent_config("/path/to/repo")
# config = {"configurable": {"thread_id": "<sha256-hash>"}}

thread_id = generate_thread_id("/path/to/repo")
```

---

### Repo Resolver

**Module:** `src/agents/repo_resolver.py`

```python
from agents.repo_resolver import resolve_repo, is_github_url

with resolve_repo("https://github.com/owner/repo") as local_path:
    # local_path is a Path to the cloned repo
    ...
# Temp directory cleaned up automatically

with resolve_repo("/path/to/local/repo") as local_path:
    # local_path is the same Path passed in
    ...
```

**`is_github_url(source: str) → bool`** — Validates GitHub URL format.

**`extract_repo_name(url: str) → str`** — Gets repo name from URL.

**`clone_repo(url: str, target: Path) → None`** — Shallow clone with timeout.
