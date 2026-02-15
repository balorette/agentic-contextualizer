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
    create_search_tools,
)
from ..tools.progressive import create_progressive_tools
from ..backends import ASTFileAnalysisBackend
from ..file_access import SmartFileAccess
from .scoped_generator import ScopedGenerator
from ..llm.provider import create_llm_provider
from ..config import Config


SCOPED_AGENT_SYSTEM_PROMPT = """You are a scoped context generator agent. Your goal is to analyze a specific aspect of a codebase and produce focused documentation.

## Available Tools

### File Discovery
1. **search_for_files** - Search for files by keywords (filename or content match)
2. **grep_in_files** - Search for regex patterns with line numbers and context

### Progressive File Analysis (cheapest -> most expensive)
3. **get_file_outline** (~500 bytes) - Get file structure: imports, symbols, signatures. ALWAYS use this before read_file.
4. **read_symbol** (~1-2 KB) - Extract a specific function/method/class body by name.
5. **read_lines** (variable) - Read an exact line range from a file.
6. **find_references** (~2-3 KB) - Find all usages of a symbol across the codebase.
7. **read_file** (~8 KB) - Read full file. LAST RESORT — only for config/non-code files.

### Output Generation
8. **generate_scoped_context** - Generate the final scoped context markdown file (pass file paths, not contents).

## Workflow Strategy

### Step 1: SEARCH
- Use `search_for_files` with keywords from the scope question to find candidate files.
- Use `grep_in_files` to search for specific patterns and get line numbers.

### Step 2: OUTLINE
- Use `get_file_outline` on the top candidates to see their structure.
- This shows imports, function names, signatures, and line numbers — WITHOUT reading file bodies.
- Decide which specific symbols are relevant from the outlines.

### Step 3: DRILL
- Use `read_symbol` to read specific functions or classes you identified in Step 2.
- Use `read_lines` when you need a specific range (from grep results or outline line numbers).
- Only use `read_file` for non-code files (config, README, etc.).

### Step 4: CONNECT
- Use `find_references` to see where key functions/classes are used across the codebase.
- Use `get_file_outline` on referenced files to understand how they fit.

### Step 5: GENERATE
When you have sufficient context (typically 5-15 relevant files), use `generate_scoped_context` with:
- The list of **file paths** (the tool reads contents automatically)
- Your analysis and insights
- Code references with specific line numbers

## Cost Hierarchy

RULE: Never call read_file on a code file without first calling get_file_outline.

| Tool | Cost | Use When |
|------|------|----------|
| get_file_outline | ~500 bytes | Understanding any file's structure |
| read_symbol | ~1-2 KB | Need a specific function/method body |
| read_lines | variable | Need an exact line range |
| find_references | ~2-3 KB | Tracing cross-file relationships |
| read_file | ~8 KB | Config/non-code files only |

## Guidelines

- **Budget**: Aim for 5-10 file outlines + 5-10 symbol reads (much cheaper than 10-20 full file reads)
- **Focus**: Stay on topic — don't explore tangential code
- **Tests**: Outline test files to see what's tested, then read specific test functions
- **Imports**: Outlines show imports — follow them to understand architecture
- **Line Numbers**: Track and report specific line numbers for key code
- **Confidence**: Generate output when you can answer the question, not when you've read everything
- **File Paths**: When calling generate_scoped_context, pass file paths — NOT file contents

## Token Economy

Every tool result is added to the conversation and sent with each subsequent API call. Progressive disclosure keeps this small:

- get_file_outline returns ~500 bytes per file (vs ~8 KB for read_file)
- read_symbol returns only the code you need (~1-2 KB)
- A typical session should accumulate ~12 KB of tool results (vs ~35 KB with full file reads)

Start with outlines. Only drill into symbols you actually need.
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

    # Progressive disclosure tools (outline -> symbol -> lines -> references -> read_file)
    analysis_backend = ASTFileAnalysisBackend()
    smart_access = SmartFileAccess(backend, analysis_backend)
    progressive_tools = create_progressive_tools(smart_access, max_read_chars=8000)

    # Keep search_for_files and grep_in_files for file discovery
    file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
    search_tools = create_search_tools(
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

    # Combine: discovery tools + progressive tools + generate
    search_for_files_tool = next(t for t in file_tools if t.name == "search_for_files")
    grep_tool = next(t for t in search_tools if t.name == "grep_in_files")
    tools = [search_for_files_tool, grep_tool] + progressive_tools + [generate_scoped_context]

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

    # Progressive disclosure tools
    analysis_backend = ASTFileAnalysisBackend()
    smart_access = SmartFileAccess(backend, analysis_backend)
    progressive_tools = create_progressive_tools(smart_access, max_read_chars=8000)

    file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
    search_tools = create_search_tools(
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

    # Combine: discovery tools + progressive tools + generate
    search_for_files_tool = next(t for t in file_tools if t.name == "search_for_files")
    grep_tool = next(t for t in search_tools if t.name == "grep_in_files")
    tools = [search_for_files_tool, grep_tool] + progressive_tools + [generate_scoped_context]
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
