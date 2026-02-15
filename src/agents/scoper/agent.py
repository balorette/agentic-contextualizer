"""Scoped context agent factory."""

import logging
from pathlib import Path
from typing import Optional
from langchain.agents import create_agent
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

from ..tools import (
    FileBackend,
    LocalFileBackend,
    CodeReference,
    create_file_tools,
    create_analysis_tools,
    create_search_tools,
)
from .scoped_generator import ScopedGenerator
from ..llm.provider import create_llm_provider
from ..config import Config


SCOPED_AGENT_SYSTEM_PROMPT = """You are a scoped context generator agent. Your goal is to analyze a specific aspect of a codebase and produce focused documentation.

## Available Tools

You have access to the following tools:

### File Discovery
1. **search_for_files** - Search for files by keywords (filename or content match)
2. **grep_in_files** - Search for regex patterns with line numbers and context
3. **find_code_definitions** - Find function, class, or method definitions by name

### File Analysis
4. **read_file** - Read file content from the repository
5. **extract_file_imports** - Parse imports from Python/JS/TS files to find related code

### Output Generation
6. **generate_scoped_context** - Generate the final scoped context markdown file (pass file paths, not contents)

## Workflow Strategy

Follow this exploration strategy:

### Step 1: Search
- Use `search_for_files` with keywords from the scope question to find candidate files
- Use `grep_in_files` to search for specific patterns and get line numbers
- Use `find_code_definitions` to locate functions or classes by name

### Step 2: Read and Analyze
- Read the top candidate files with `read_file`
- Look for the code most relevant to the question
- **Track important line numbers** as you read
- Use `extract_file_imports` to find related files

### Step 3: Follow Dependencies
- Read imported files that seem relevant
- Look for test files (they often explain behavior)
- Check configuration files if relevant
- Use grep to find usages of key functions

### Step 4: Generate
When you have sufficient context (typically 5-15 relevant files), use `generate_scoped_context` with:
- The list of **file paths** you found relevant (the tool reads contents automatically)
- Your analysis and insights
- Code references with specific line numbers

## Guidelines

- **Budget**: Aim for 10-20 file reads maximum
- **Focus**: Stay on topic - don't explore tangential code
- **Tests**: Test files are valuable - they show expected behavior
- **Imports**: Following imports reveals architecture
- **Line Numbers**: Track and report specific line numbers for key code
- **Confidence**: Generate output when you can answer the question, not when you've read everything
- **File Paths**: When calling generate_scoped_context, pass file paths — NOT file contents. The tool reads files automatically.

## Output Format

The final scoped context should answer the user's question with:
- A clear summary
- Relevant code locations with **specific line numbers** (e.g., `src/auth.py:45-78`)
- Key files to examine
- Usage examples if available
- A Code References section listing important file:line locations

## Token Economy

Every tool result is added to the conversation and sent with each subsequent API call. Large results compound quickly.

- **grep_in_files**: Start with `max_results=5`. Only increase if you need more matches.
- **search_for_files**: Start with `max_results=5`. Narrow with more specific keywords rather than increasing results.
- **find_code_definitions**: Start with `max_results=5`.
- **read_file**: Large files are automatically truncated. If you only need a section, note the relevant lines and move on.
- Prefer targeted grep searches over reading entire files.
- If a tool returns too many results, refine the query instead of reading them all.

## Important Notes

- Don't read files you've already read
- Prioritize files that directly address the question
- Use grep to quickly find specific patterns instead of reading entire files
- Track line numbers for important code you discover
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
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
    config: Optional["Config"] = None,
):
    """Create a scoped context generation agent.

    Args:
        repo_path: Path to repository to analyze
        file_backend: Optional pre-configured file backend
        model_name: LLM model identifier
        checkpointer: Optional checkpointer for state persistence
        output_dir: Directory for output files
        debug: Enable verbose logging
        base_url: Optional custom API endpoint URL

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
    if config is None:
        config = Config.from_env()

    # Derive use_litellm from config when caller didn't explicitly set it
    if not use_litellm and config.llm_provider == "litellm":
        use_litellm = True

    from ..llm.chat_model_factory import build_chat_model, build_token_middleware
    from ..llm.rate_limiting import TPMThrottle

    model = build_chat_model(
        config=config,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        use_litellm=use_litellm,
        debug=debug,
    )

    # Create shared throttle instance so agent loop and generation LLM
    # are aware of each other's token usage
    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)

    # Create file, analysis, and code search tools bound to backend
    # Use tighter limits for agent mode to reduce token consumption
    file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
    analysis_tools = create_analysis_tools(backend)
    code_search_tools = create_search_tools(
        backend, max_grep_results=15, max_def_results=15, context_lines=1
    )

    # Create the generation tool (needs LLM and output config)
    # Reuse the config from above to ensure consistent credentials
    llm_provider = create_llm_provider(config, throttle=throttle)
    generator = ScopedGenerator(llm_provider, output_dir)

    @tool
    def generate_scoped_context(
        question: str,
        relevant_file_paths: list[str],
        insights: str,
        code_references: list[dict] | None = None,
    ) -> dict:
        """Generate the final scoped context markdown file.

        Call this when you have gathered sufficient context to answer the question.
        Pass the PATHS of relevant files — the tool reads their contents automatically.

        Args:
            question: The original scope question being answered
            relevant_file_paths: List of file paths the agent determined are relevant
            insights: Your analysis and insights about the code
            code_references: Optional list of code reference dicts with keys:
                - path: File path
                - line_start: Starting line number
                - line_end: Optional ending line number
                - description: Brief description of what this code does

        Returns:
            Dictionary with:
            - output_path: Path to generated markdown file
            - error: Error message if generation failed
        """
        try:
            if not relevant_file_paths:
                return {
                    "output_path": None,
                    "error": "relevant_file_paths must not be empty.",
                }

            # Read file contents via backend
            relevant_files = {}
            for file_path in relevant_file_paths:
                content = backend.read_file(file_path)
                if content is not None:
                    relevant_files[file_path] = content

            if not relevant_files:
                return {
                    "output_path": None,
                    "error": f"Could not read any of the {len(relevant_file_paths)} provided file paths.",
                }

            # Convert code reference dicts to CodeReference objects
            refs = None
            if code_references:
                refs = []
                for ref in code_references:
                    try:
                        refs.append(CodeReference(
                            path=ref["path"],
                            line_start=ref["line_start"],
                            line_end=ref.get("line_end"),
                            description=ref.get("description", ""),
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue  # Skip malformed references

            output_path = generator.generate(
                repo_name=repo_path.name,
                question=question,
                relevant_files=relevant_files,
                insights=insights,
                model_name=config.model_name,
                source_repo=str(repo_path),
                code_references=refs,
            )
            return {
                "output_path": str(output_path),
                "error": None,
            }
        except Exception as e:
            logger.exception("generate_scoped_context failed")
            return {
                "output_path": None,
                "error": str(e),
            }

    # Combine all tools
    tools = file_tools + analysis_tools + code_search_tools + [generate_scoped_context]

    budget_mw = build_token_middleware(config, model_name, throttle=throttle)

    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SCOPED_AGENT_SYSTEM_PROMPT,
        middleware=[budget_mw],
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
    base_url: Optional[str] = None,
    config: Optional["Config"] = None,
):
    """Create scoped agent with budget tracking wired into middleware.

    The BudgetTracker is automatically fed actual token usage via the
    middleware's after_model hook — no manual extraction needed.

    Args:
        repo_path: Path to repository
        max_tokens: Maximum tokens allowed (default: 30k)
        max_cost_usd: Maximum cost in USD (default: $2.00)
        model_name: LLM model identifier
        checkpointer: Optional checkpointer for state persistence
        output_dir: Directory for output files
        debug: Enable verbose logging
        base_url: Optional custom API endpoint URL
        config: Optional Config override

    Returns:
        Tuple of (agent, budget_tracker)
    """
    from ..middleware import BudgetTracker
    from ..llm.chat_model_factory import build_chat_model, build_token_middleware
    from ..llm.rate_limiting import TPMThrottle

    repo_path = Path(repo_path).resolve()
    backend = LocalFileBackend(repo_path)

    if config is None:
        config = Config.from_env()

    use_litellm = config.llm_provider == "litellm"

    model = build_chat_model(
        config=config,
        model_name=model_name,
        base_url=base_url,
        use_litellm=use_litellm,
        debug=debug,
    )

    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)

    file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
    analysis_tools = create_analysis_tools(backend)
    code_search_tools = create_search_tools(
        backend, max_grep_results=15, max_def_results=15, context_lines=1
    )

    llm_provider = create_llm_provider(config, throttle=throttle)
    generator = ScopedGenerator(llm_provider, output_dir)

    @tool
    def generate_scoped_context(
        question: str,
        relevant_file_paths: list[str],
        insights: str,
        code_references: list[dict] | None = None,
    ) -> dict:
        """Generate the final scoped context markdown file.

        Call this when you have gathered sufficient context to answer the question.
        Pass the PATHS of relevant files — the tool reads their contents automatically.

        Args:
            question: The original scope question being answered
            relevant_file_paths: List of file paths the agent determined are relevant
            insights: Your analysis and insights about the code
            code_references: Optional list of code reference dicts with keys:
                - path: File path
                - line_start: Starting line number
                - line_end: Optional ending line number
                - description: Brief description of what this code does

        Returns:
            Dictionary with:
            - output_path: Path to generated markdown file
            - error: Error message if generation failed
        """
        try:
            if not relevant_file_paths:
                return {"output_path": None, "error": "relevant_file_paths must not be empty."}

            relevant_files = {}
            for file_path in relevant_file_paths:
                content = backend.read_file(file_path)
                if content is not None:
                    relevant_files[file_path] = content

            if not relevant_files:
                return {
                    "output_path": None,
                    "error": f"Could not read any of the {len(relevant_file_paths)} provided file paths.",
                }

            refs = None
            if code_references:
                refs = []
                for ref in code_references:
                    try:
                        refs.append(CodeReference(
                            path=ref["path"],
                            line_start=ref["line_start"],
                            line_end=ref.get("line_end"),
                            description=ref.get("description", ""),
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue

            output_path = generator.generate(
                repo_name=repo_path.name,
                question=question,
                relevant_files=relevant_files,
                insights=insights,
                model_name=config.model_name,
                source_repo=str(repo_path),
                code_references=refs,
            )
            return {"output_path": str(output_path), "error": None}
        except Exception as e:
            logger.exception("generate_scoped_context failed")
            return {"output_path": None, "error": str(e)}

    tools = file_tools + analysis_tools + code_search_tools + [generate_scoped_context]
    budget_mw = build_token_middleware(config, model_name, throttle=throttle, budget_tracker=tracker)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SCOPED_AGENT_SYSTEM_PROMPT,
        middleware=[budget_mw],
        checkpointer=checkpointer,
        debug=debug,
    )

    return agent, tracker
