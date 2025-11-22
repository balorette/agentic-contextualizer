# Agentic Contextualizer

Generate effective context files for codebases that AI coding agents can use to understand projects quickly.

## Status

✅ **Ready to Use** - Full pipeline implementation complete

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repo-url>
cd agentic-contextualizer

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Quick Start

**1. Set up your API key:**

```bash
# Create .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

**2. Generate context for a repository:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate context for any repository
python -m agents.main generate /path/to/your/repo \
  --summary "Brief description of what the project does"

# Example: Generate context for this project
python -m agents.main generate . \
  --summary "Tool that generates context files for AI agents"
```

**3. Find your generated context:**

The context file will be saved to `contexts/<repo-name>/context.md`

### Generate Context

The `generate` command scans a repository and creates a comprehensive context file:

```bash
python -m agents.main generate <REPO_PATH> --summary "<DESCRIPTION>"

# Options:
#   REPO_PATH          Path to the repository (required)
#   -s, --summary      Brief description of the project (required)
#   -o, --output       Custom output path (optional)
```

**What happens during generation:**
1. **Structure Scan** - Walks the file tree, respects .gitignore patterns
2. **Metadata Extraction** - Identifies project type, dependencies, entry points
3. **Code Analysis** - Uses Claude to analyze architecture and patterns (1 LLM call)
4. **Context Generation** - Creates structured markdown documentation (1 LLM call)

**Output format:**
- YAML frontmatter with metadata (source, date, model used)
- Architecture overview
- Key commands (build, test, run)
- Code patterns and conventions
- Entry points

### Refine Context

Update an existing context file based on feedback:

```bash
python -m agents.main refine contexts/myproject/context.md \
  --request "Add more details about the authentication flow"

# Options:
#   CONTEXT_FILE       Path to existing context.md file (required)
#   -r, --request      What to change or add (required)
```

### Example Workflow

```bash
# 1. Generate initial context
python -m agents.main generate ~/projects/my-app \
  --summary "FastAPI REST API with PostgreSQL backend"

# Output: contexts/my-app/context.md created

# 2. Review the generated context
cat contexts/my-app/context.md

# 3. Refine if needed
python -m agents.main refine contexts/my-app/context.md \
  --request "Include details about the database migration strategy"

# 4. Use the context with your AI coding agent
# Copy contexts/my-app/context.md to your agent's context
```

### Cost Efficiency

The tool makes exactly **2 LLM calls** per generation:
- 1 call for code analysis
- 1 call for context generation

Refinement makes **1 additional call** per request.

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture and design decisions.

## License

MIT
