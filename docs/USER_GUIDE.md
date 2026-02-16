# User Guide

> Agentic Contextualizer v0.1.0

## What Is This?

Agentic Contextualizer scans your codebase and generates structured markdown files ("context files") that AI coding agents can use to understand your project quickly. Instead of your AI assistant reading hundreds of files, it reads one context file that captures your project's architecture, patterns, and key information.

## Getting Started

### 1. Install

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url>
cd agentic-contextualizer
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Generate Your First Context

```bash
python -m agents.main generate /path/to/your/repo \
  --summary "Brief description of what the project does"
```

Your context file will appear at `contexts/<repo-name>/context.md`.

## Commands

### Generate Context

Creates a comprehensive context file for an entire repository.

```bash
python -m agents.main generate SOURCE --summary "DESCRIPTION"
```

**SOURCE** can be:
- A local path: `/path/to/my-project`
- A GitHub URL: `https://github.com/owner/repo`

**What happens:**
1. Scans repository structure (no LLM call)
2. Extracts project metadata — type, dependencies, entry points (no LLM call)
3. Analyzes code patterns and architecture (1 LLM call)
4. Generates structured context markdown (1 LLM call)

**Cost:** Exactly 2 LLM API calls.

**Example output:**
```markdown
---
source_repo: /home/user/projects/my-api
scan_date: 2025-01-22T10:30:00Z
user_summary: "FastAPI REST API with PostgreSQL"
model_used: anthropic/claude-3-5-sonnet-20241022
---

# Repository Context: my-api

## Architecture Overview
This is a FastAPI-based REST API following a layered architecture...

## Key Commands
- **Run**: `uvicorn app.main:app --reload`
- **Test**: `pytest`
...
```

### Refine Context

Update an existing context file when you want more detail or corrections.

```bash
python -m agents.main refine contexts/my-api/context.md \
  --request "Add more details about the authentication flow"
```

**Cost:** 1 LLM call per refinement.

### Scope Context

Generate focused context for a specific question or topic. This is useful when you need deep information about one area of the codebase.

```bash
# From a repository
python -m agents.main scope /path/to/repo \
  --question "How does authentication work?"

# From an existing context file (faster — reuses metadata)
python -m agents.main scope contexts/my-api/context.md \
  --question "database migration strategy"

# From a GitHub URL
python -m agents.main scope https://github.com/owner/repo \
  --question "API endpoints"
```

**What happens:**
1. Extracts keywords from your question (no LLM)
2. Searches for matching files by name and content (no LLM)
3. LLM explores candidates, follows imports and dependencies (1-3 LLM calls)
4. Generates focused context answering your question (1 LLM call)

**Cost:** 2-4 LLM calls depending on exploration depth.

Output: `contexts/<repo-name>/scope-<topic>-<timestamp>.md`

## Execution Modes

### Pipeline Mode (Default)

Deterministic, predictable cost. Steps execute in order.

```bash
python -m agents.main generate /repo -s "description"
```

### Agent Mode

Uses a LangChain agent that autonomously decides which tools to call. More flexible but potentially more LLM calls.

```bash
python -m agents.main generate /repo -s "description" --mode agent
```

**Scoped agent mode** uses progressive disclosure to minimize token usage. Instead of reading full files, the agent first outlines files (~500 bytes), then reads only the specific functions it needs (~1-2 KB each). This reduces cumulative context from ~35 KB to ~12 KB per session (~65% reduction).

### Streaming

Add `--stream` for real-time output in agent mode:

```bash
python -m agents.main generate /repo -s "description" --mode agent --stream
```

## Typical Workflow

```bash
# Step 1: Generate initial context for your project
python -m agents.main generate ~/projects/my-webapp \
  --summary "React frontend with Express backend and MongoDB"

# Step 2: Review the output
cat contexts/my-webapp/context.md

# Step 3: Refine if needed
python -m agents.main refine contexts/my-webapp/context.md \
  --request "Include details about the WebSocket real-time features"

# Step 4: Create scoped context for specific topics
python -m agents.main scope contexts/my-webapp/context.md \
  --question "authentication and session management"

python -m agents.main scope contexts/my-webapp/context.md \
  --question "database schema and migrations"

# Step 5: Use with your AI coding agent
# Copy the context files to your agent's context window
```

## Using Context Files with AI Agents

The generated context files work with any AI coding assistant. Common approaches:

1. **Direct paste**: Copy the context file content into your conversation
2. **CLAUDE.md**: For Claude Code, place the content in your project's `CLAUDE.md`
3. **System prompt**: Include as part of your agent's system prompt
4. **RAG pipeline**: Index context files for retrieval-augmented generation

## Cost Guide

| Operation | LLM Calls | Approximate Cost |
|-----------|-----------|-----------------|
| Generate (pipeline) | 2 | ~$0.05-0.15 |
| Refine | 1 | ~$0.02-0.05 |
| Scope (pipeline) | 2-4 | ~$0.05-0.20 |
| Scope (agent) | 2-5 | ~$0.03-0.15 |
| Generate (agent) | 2-5 | ~$0.05-0.30 |

Costs depend on repository size and model used. The tool enforces content limits to keep costs predictable.

Scoped agent mode with progressive disclosure is more cost-efficient than before — by reading file outlines instead of full files, each LLM call processes ~65% less context, reducing both token usage and cost.

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

Make sure your `.env` file exists and contains a valid key:
```bash
echo $ANTHROPIC_API_KEY  # Should print your key
```

### "Repository path does not exist"

Use an absolute path or ensure the relative path is correct from your current directory.

### "Clone timed out"

For large GitHub repositories, try cloning manually first, then use the local path:
```bash
git clone --depth 1 https://github.com/owner/large-repo /tmp/large-repo
python -m agents.main generate /tmp/large-repo -s "description"
```

### Empty or thin context

Provide a more detailed `--summary` to guide the analysis:
```bash
# Instead of:
python -m agents.main generate /repo -s "web app"

# Try:
python -m agents.main generate /repo -s "FastAPI REST API with JWT auth, PostgreSQL, and Celery background tasks"
```

### Scoped context misses relevant files

Use more specific questions:
```bash
# Instead of:
python -m agents.main scope /repo -q "auth"

# Try:
python -m agents.main scope /repo -q "JWT authentication middleware and token refresh flow"
```
