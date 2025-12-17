"""Scoped context agent factory."""

from pathlib import Path
from typing import Optional
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from .backends import FileBackend, LocalFileBackend
from .tools.file_tools import create_file_tools
from .tools.analysis_tools import create_analysis_tools
from .scoped_generator import ScopedGenerator
from ..llm.provider import AnthropicProvider
from ..config import Config


SCOPED_AGENT_SYSTEM_PROMPT = """You are a scoped context generator agent. Your goal is to analyze a specific aspect of a codebase and produce focused documentation.

## Available Tools

You have access to the following tools:

1. **read_file** - Read file content from the repository
2. **search_for_files** - Search for files by keywords (filename or content match)
3. **extract_file_imports** - Parse imports from Python/JS/TS files to find related code
4. **generate_scoped_context** - Generate the final scoped context markdown file

## Workflow Strategy

Follow this exploration strategy:

### Step 1: Search
Use `search_for_files` with keywords from the scope question to find candidate files.

### Step 2: Read and Analyze
- Read the top candidate files with `read_file`
- Look for the code most relevant to the question
- Use `extract_file_imports` to find related files

### Step 3: Follow Dependencies
- Read imported files that seem relevant
- Look for test files (they often explain behavior)
- Check configuration files if relevant

### Step 4: Generate
When you have sufficient context (typically 5-15 relevant files), use `generate_scoped_context` to produce the final documentation.

## Guidelines

- **Budget**: Aim for 10-20 file reads maximum
- **Focus**: Stay on topic - don't explore tangential code
- **Tests**: Test files are valuable - they show expected behavior
- **Imports**: Following imports reveals architecture
- **Confidence**: Generate output when you can answer the question, not when you've read everything

## Output Format

The final scoped context should answer the user's question with:
- A clear summary
- Relevant code locations
- Key files to examine
- Usage examples if available

## Important Notes

- Don't read files you've already read
- Prioritize files that directly address the question
- If the question is ambiguous, make reasonable assumptions
- Report what you found even if it's incomplete
"""


def create_scoped_agent(
    repo_path: str | Path,
    file_backend: FileBackend | None = None,
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    output_dir: str = "contexts",
    debug: bool = False,
):
    """Create a scoped context generation agent.

    Args:
        repo_path: Path to repository to analyze
        file_backend: Optional pre-configured file backend
        model_name: LLM model identifier
        checkpointer: Optional checkpointer for state persistence
        output_dir: Directory for output files
        debug: Enable verbose logging

    Returns:
        Compiled StateGraph agent ready for invocation

    Example:
        ```python
        from src.agents.scoper.agent import create_scoped_agent

        agent = create_scoped_agent("/path/to/repo")

        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "How does authentication work in this codebase?"
            }]
        })
        ```
    """
    repo_path = Path(repo_path).resolve()

    # Create or use provided backend
    backend = file_backend or LocalFileBackend(repo_path)

    # Initialize chat model
    model = init_chat_model(model_name)

    # Create file and analysis tools bound to backend
    file_tools = create_file_tools(backend)
    analysis_tools = create_analysis_tools(backend)

    # Create the generation tool (needs LLM and output config)
    config = Config.from_env()
    llm_provider = AnthropicProvider(config.model_name, config.api_key)
    generator = ScopedGenerator(llm_provider, output_dir)

    @tool
    def generate_scoped_context(
        question: str,
        relevant_files: dict[str, str],
        insights: str,
    ) -> dict:
        """Generate the final scoped context markdown file.

        Call this when you have gathered sufficient context to answer the question.
        This will create a markdown file with focused documentation.

        Args:
            question: The original scope question being answered
            relevant_files: Dictionary mapping file paths to their content
            insights: Your analysis and insights about the code

        Returns:
            Dictionary with:
            - output_path: Path to generated markdown file
            - error: Error message if generation failed
        """
        try:
            output_path = generator.generate(
                repo_name=repo_path.name,
                question=question,
                relevant_files=relevant_files,
                insights=insights,
                model_name=config.model_name,
                source_repo=str(repo_path),
            )
            return {
                "output_path": str(output_path),
                "error": None,
            }
        except Exception as e:
            return {
                "output_path": None,
                "error": str(e),
            }

    # Combine all tools
    tools = file_tools + analysis_tools + [generate_scoped_context]

    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SCOPED_AGENT_SYSTEM_PROMPT,
        middleware=[],
        checkpointer=checkpointer,
        debug=debug,
    )

    return agent


def create_scoped_agent_with_budget(
    repo_path: str | Path,
    max_tokens: int = 30000,
    max_cost_usd: float = 2.0,
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    output_dir: str = "contexts",
    debug: bool = False,
):
    """Create scoped agent with budget tracking.

    Args:
        repo_path: Path to repository
        max_tokens: Maximum tokens allowed (default: 30k)
        max_cost_usd: Maximum cost in USD (default: $2.00)
        model_name: LLM model identifier
        checkpointer: Optional checkpointer for state persistence
        output_dir: Directory for output files
        debug: Enable verbose logging

    Returns:
        Tuple of (agent, budget_tracker)
    """
    from ..middleware import BudgetTracker

    agent = create_scoped_agent(
        repo_path=repo_path,
        model_name=model_name,
        checkpointer=checkpointer,
        output_dir=output_dir,
        debug=debug,
    )

    tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)

    return agent, tracker
