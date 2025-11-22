# Agentic Contextualizer

Generate effective context files for codebases that AI coding agents can use to understand projects quickly.

## Status

🚧 **Under Active Development** - Foundation phase complete

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

*Coming soon - pipeline implementation in progress*

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
