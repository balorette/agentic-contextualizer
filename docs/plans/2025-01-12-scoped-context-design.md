# Scoped Context Generation Design

## Overview

Add a `scope` command that generates focused context for a specific question/domain from a repository. Designed for distributed system workflows where you need deep context on a specific topic from a dependency repo.

## Use Case

Example: Building a frontend that consumes a backend in a separate repo. Instead of loading the entire backend context, generate scoped context for just the functionality you're implementing (e.g., "weather functionality").

**Layered context model:**
- General context: High-level understanding ("backend does X, Y, Z")
- Scoped context: Deep dive on specific topic (generated on-demand)

## CLI Interface

```bash
# Scope from repo directly
python -m agents.main scope /path/to/backend \
  --question "weather functionality" \
  --output contexts/backend/scope-weather.md   # optional

# Scope from existing general context (faster, reuses scan)
python -m agents.main scope contexts/backend/context.md \
  --question "weather functionality"

# With agent mode and streaming
python -m agents.main scope /path/to/backend \
  --question "authentication flow" \
  --mode agent --stream
```

### Options

| Option | Required | Description |
|--------|----------|-------------|
| `SOURCE` | Yes | Repo path OR existing context.md file |
| `--question, -q` | Yes | The question/topic to scope to |
| `--output, -o` | No | Output file path |
| `--mode, -m` | No | `pipeline` (default) or `agent` |
| `--stream` | No | Enable streaming output (agent mode) |
| `--debug` | No | Enable debug output |

### Output Location Defaults

- From repo: `contexts/{repo-name}/scope-{sanitized-question}.md`
- From context file: Same directory as source context

## How Scoping Works (Pipeline Mode)

### Phase 1: Discovery (No LLM)

1. Parse question for keywords/terms
2. Grep codebase for content matches
3. Glob for files with matching names
4. If starting from existing context, use its metadata to narrow search
5. Output: candidate file list

### Phase 2: Exploration (LLM-guided)

1. Give LLM: file tree + candidate files + question
2. LLM reads candidates, decides what to follow (imports, tests, configs)
3. **Interactive checkpoint**: "Found N relevant files. Explore deeper or generate now?"
4. Budget middleware enforces limits
5. LLM continues until: sufficient context OR budget hit OR user says "generate now"
6. Output: final relevant file set + preliminary insights

### Phase 3: Synthesis (1 LLM Call)

1. Generate scoped context from curated file set
2. Focused on answering the original question
3. Output: scoped context markdown file

**Estimated LLM calls:** 2-4 depending on exploration depth

## Agent Mode

Reuses existing agent infrastructure:
- Same `--mode agent` flag as generate/refine
- Uses existing tools: `scan_repository`, `read_files`, `search_code`
- Interactive checkpoint via `human_in_the_loop` middleware
- Budget middleware enforces limits
- Agent decides exploration depth based on question complexity

More conversational and flexible than pipeline mode.

## Output Format

Markdown with YAML frontmatter. Structure is flexible based on question and codebase type.

```markdown
---
source_repo: /path/to/repo
source_context: contexts/backend/context.md  # if scoped from existing
scope_question: "weather functionality"
scan_date: 2025-01-22T10:30:00Z
model_used: anthropic/claude-3-5-sonnet
files_analyzed: 12
---

# Scoped Context: {Question/Topic}

## Summary
{Direct answer to the scope question}

{LLM-generated sections based on what's relevant}
{Structure varies by codebase type and question}

## Key Files
{Files the reader should examine}

## Usage Examples / Related Tests
{When available - demonstrates actual usage}
```

**Fixed sections:** Summary, Key Files, Usage Examples/Tests

**Dynamic sections:** LLM determines based on question + codebase (could be API Endpoints, Processing Pipeline, State Management, Event Contracts, etc.)

## Refinement

Existing `refine` command works unchanged on scoped context files:

```bash
python -m agents.main refine contexts/backend/scope-weather.md \
  --request "add error handling details"
```

## Implementation Components

### New Files

| File | Purpose |
|------|---------|
| `src/agents/scoper/__init__.py` | Module init |
| `src/agents/scoper/discovery.py` | Phase 1: keyword extraction, grep/glob |
| `src/agents/scoper/scoped_analyzer.py` | Phase 2: LLM exploration with checkpoints |
| `src/agents/scoper/scoped_generator.py` | Phase 3: synthesize scoped context |

### Modified Files

| File | Change |
|------|--------|
| `src/agents/main.py` | Add `scope` command |
| `src/agents/llm/prompts.py` | Add scope-related prompts |
| `src/agents/models.py` | Add `ScopedContext` model |

### Reused Components

- `StructureScanner` - file tree scanning
- `LLMProvider` - API calls
- `human_in_the_loop` middleware - interactive checkpoints
- `budget` middleware - cost control
- Agent factory and tools - for agent mode

### New Prompts

1. **Exploration prompt** - Given question and candidates, what else to read?
2. **Generation prompt** - Generate focused context for question from files

## Cost Control

- Budget middleware enforces token/call limits
- Interactive checkpoint before deep exploration
- User can say "generate now" at any point
