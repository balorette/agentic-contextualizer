"""Scoped context agent factory."""

from pathlib import Path
from typing import Optional
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

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
from ..factory import _format_model_name_for_langchain


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
6. **generate_scoped_context** - Generate the final scoped context markdown file

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
When you have sufficient context (typically 5-15 relevant files), use `generate_scoped_context` to produce the final documentation. **Include code references** with specific line numbers.

## Guidelines

- **Budget**: Aim for 10-20 file reads maximum
- **Focus**: Stay on topic - don't explore tangential code
- **Tests**: Test files are valuable - they show expected behavior
- **Imports**: Following imports reveals architecture
- **Line Numbers**: Track and report specific line numbers for key code
- **Confidence**: Generate output when you can answer the question, not when you've read everything

## Output Format

The final scoped context should answer the user's question with:
- A clear summary
- Relevant code locations with **specific line numbers** (e.g., `src/auth.py:45-78`)
- Key files to examine
- Usage examples if available
- A Code References section listing important file:line locations

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
    config = Config.from_env()

    # Auto-detect if we should use LiteLLM
    should_use_litellm = use_litellm or base_url is not None or config.llm_provider == "litellm"

    # Resolve API key if not explicitly provided
    if not api_key:
        from ..llm.provider import _resolve_api_key_for_model
        api_key = _resolve_api_key_for_model(model_name, config)

    if should_use_litellm:
        # Use ChatLiteLLM for custom gateways
        from langchain_litellm import ChatLiteLLM

        # Set up kwargs for ChatLiteLLM
        litellm_kwargs = {
            "model": model_name,
            "temperature": 0.0,
        }

        if api_key:
            litellm_kwargs["api_key"] = api_key
        if base_url:
            litellm_kwargs["api_base"] = base_url
        if config.max_retries:
            litellm_kwargs["max_retries"] = config.max_retries
        if config.timeout:
            litellm_kwargs["request_timeout"] = config.timeout
        if config.max_output_tokens:
            litellm_kwargs["max_tokens"] = config.max_output_tokens

        if debug:
            print(f"[DEBUG] ChatLiteLLM kwargs: {model_name}")

        model = ChatLiteLLM(**litellm_kwargs)

        if debug:
            print(f"[DEBUG] Created ChatLiteLLM for scoped agent with model: {model_name}")
    else:
        # Use standard LangChain init_chat_model
        formatted_model_name = _format_model_name_for_langchain(model_name)

        model_kwargs = {}
        if base_url:
            model_kwargs["base_url"] = base_url
        if api_key:
            model_kwargs["api_key"] = api_key

        model = init_chat_model(formatted_model_name, **model_kwargs)

    # Create file, analysis, and code search tools bound to backend
    file_tools = create_file_tools(backend)
    analysis_tools = create_analysis_tools(backend)
    code_search_tools = create_search_tools(backend)

    # Create the generation tool (needs LLM and output config)
    # Reuse the config from above to ensure consistent credentials
    llm_provider = create_llm_provider(config)
    generator = ScopedGenerator(llm_provider, output_dir)

    @tool
    def generate_scoped_context(
        question: str,
        relevant_files: dict[str, str],
        insights: str,
        code_references: list[dict] | None = None,
    ) -> dict:
        """Generate the final scoped context markdown file.

        Call this when you have gathered sufficient context to answer the question.
        This will create a markdown file with focused documentation.

        Args:
            question: The original scope question being answered
            relevant_files: Dictionary mapping file paths to their content
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
            # Convert code reference dicts to CodeReference objects
            refs = None
            if code_references:
                refs = [
                    CodeReference(
                        path=ref["path"],
                        line_start=ref["line_start"],
                        line_end=ref.get("line_end"),
                        description=ref["description"],
                    )
                    for ref in code_references
                ]

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
            return {
                "output_path": None,
                "error": str(e),
            }

    # Combine all tools
    tools = file_tools + analysis_tools + code_search_tools + [generate_scoped_context]

    # TPM-aware throttle — shared instance for this agent session
    from ..llm.rate_limiting import TPMThrottle
    from ..llm.token_estimator import LiteLLMTokenEstimator

    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)
    estimator = LiteLLMTokenEstimator()

    # Token budget middleware — trims messages, truncates tool output, and throttles TPM
    from ..middleware.token_budget import TokenBudgetMiddleware
    budget_mw = TokenBudgetMiddleware(
        max_input_tokens=config.max_input_tokens,
        max_tool_output_chars=config.max_tool_output_chars,
        throttle=throttle,
        estimator=estimator,
        model_name=model_name,
    )

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
        base_url: Optional custom API endpoint URL

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
        base_url=base_url,
    )

    tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)

    return agent, tracker
