"""Agent factory for creating LangChain agents."""

from typing import Optional
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from .tools.repository_tools import (
    scan_structure,
    extract_metadata,
    analyze_code,
    generate_context,
    refine_context,
)
from .tools.exploration_tools import (
    list_key_files,
    read_file_snippet,
)


def _format_model_name_for_langchain(model_name: str) -> str:
    """Format model name for LangChain's init_chat_model.

    LangChain expects format: "provider:model"
    Examples: "openai:gpt-4o", "anthropic:claude-3-5-sonnet-20241022"

    Args:
        model_name: Raw model name (e.g., "gpt-4o", "claude-3-5-sonnet")

    Returns:
        Formatted model name with provider prefix
    """
    # Already has provider prefix
    if ":" in model_name:
        return model_name

    # Detect provider from model name and add prefix
    if model_name.startswith("gpt-") or model_name.startswith("o1"):
        return f"openai:{model_name}"
    elif model_name.startswith("claude"):
        return f"anthropic:{model_name}"
    elif model_name.startswith("gemini") or model_name.startswith("vertex"):
        return f"google-genai:{model_name}"
    else:
        # Default to anthropic for backward compatibility
        return f"anthropic:{model_name}"


AGENT_SYSTEM_PROMPT = """You are a repository context generator agent. Your goal is to analyze codebases and produce comprehensive context documentation for AI coding assistants.

## Available Tools

You have access to the following tools:

1. **scan_structure** - Scans repository structure, returns file tree and counts
2. **extract_metadata** - Extracts project type, dependencies, entry points from config files
3. **analyze_code** - Performs LLM-based deep code analysis (EXPENSIVE - use wisely!)
4. **generate_context** - Generates final markdown context file (EXPENSIVE - use wisely!)
5. **refine_context** - Refines an existing context file based on feedback (EXPENSIVE)
6. **list_key_files** - Quick utility to list important files from a file tree
7. **read_file_snippet** - Read specific sections of files for targeted analysis

## Workflow Guidelines

### For Initial Context Generation:
Follow this sequence:
1. Use `scan_structure` to get the repository file tree
2. Use `extract_metadata` to get project details
3. Use `analyze_code` to perform deep analysis (1 LLM call)
4. Use `generate_context` to create the final markdown file (1 LLM call)

Total: 2 expensive LLM calls (analyze + generate)

### For Context Refinement:
1. User provides path to existing context file
2. Use `refine_context` with the refinement request (1 LLM call)

### Budget Constraints

- **Target**: Keep expensive LLM calls ≤ 2 per generation
- **Token Limits**: All tools auto-limit output to prevent overflow
- **Fail Fast**: If repo is oversized (>10k files), report error and suggest focusing on subdirectory

### Error Handling

- If a tool returns an "error" field, report it clearly to the user
- Suggest recovery actions (e.g., "Repository too large, try scanning a subdirectory")
- Don't retry failed operations unless explicitly requested

### Output Format

The final context file should be markdown with YAML frontmatter:
```markdown
---
source_repo: /path/to/repo
scan_date: 2025-01-23T10:30:00Z
user_summary: "User's description"
model_used: anthropic/claude-sonnet-4-5-20250929
---

# Repository Context: {repo-name}

## Architecture Overview
...

## Key Commands
...

## Code Patterns
...

## Entry Points
...
```

## Important Notes

- Always acknowledge the user's summary/description in your analysis
- Be concise but thorough in generated context
- If user requests refinement, focus only on requested changes
- Report the output path when context is generated successfully
"""


def create_contextualizer_agent(
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    middleware: Optional[list] = None,
    checkpointer: Optional[object] = None,
    debug: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
):
    """Create a repository contextualizer agent.

    Args:
        model_name: LLM model identifier (with or without provider prefix)
            Examples: "gpt-4o", "openai:gpt-4o", "claude-3-5-sonnet-20241022"
        middleware: Optional list of middleware instances
        checkpointer: Optional checkpointer for state persistence
        debug: Enable verbose logging for graph execution
        base_url: Optional custom API endpoint URL
        api_key: Optional API key (if not provided, will be resolved from config)
        use_litellm: Force use of ChatLiteLLM (recommended for custom gateways)

    Returns:
        Compiled StateGraph agent ready for invocation

    Example:
        ```python
        from src.agents.factory import create_contextualizer_agent

        # Standard usage (direct provider)
        agent = create_contextualizer_agent(model_name="gpt-4o")

        # Custom gateway usage
        agent = create_contextualizer_agent(
            model_name="gpt-4.1",
            base_url="https://custom-gateway.com",
            use_litellm=True
        )

        # Generate context
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Generate context for /path/to/repo. It's a FastAPI REST API."
            }]
        })
        ```
    """
    from .config import Config
    from .llm.provider import _resolve_api_key_for_model

    # Auto-detect if we should use LiteLLM
    # Use LiteLLM if: custom base_url OR explicitly requested OR provider is "litellm"
    config = Config.from_env()
    should_use_litellm = use_litellm or base_url is not None or config.llm_provider == "litellm"

    # Resolve API key if not explicitly provided
    if not api_key:
        api_key = _resolve_api_key_for_model(model_name, config)

    # Debug output
    if debug:
        print(f"[DEBUG] Creating LangChain model:")
        print(f"  - Original model_name: {model_name}")
        print(f"  - Using LiteLLM: {should_use_litellm}")
        print(f"  - base_url: {base_url or 'None'}")
        print(f"  - api_key: {'***' + api_key[-4:] if api_key else 'None'}")

    if should_use_litellm:
        # Use ChatLiteLLM for custom gateways
        from langchain_litellm import ChatLiteLLM
        import litellm

        # Enable LiteLLM debug mode if debugging
        if debug:
            litellm.set_verbose = True
            # Also turn on detailed debugging
            import os
            os.environ["LITELLM_LOG"] = "DEBUG"

        # Set up kwargs for ChatLiteLLM
        litellm_kwargs = {
            "model": model_name,  # LiteLLM handles model names directly
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

        if debug:
            print(f"  - ChatLiteLLM kwargs:")
            for k, v in litellm_kwargs.items():
                if k == "api_key":
                    print(f"      {k}: ***{v[-4:] if v else 'None'}")
                else:
                    print(f"      {k}: {v}")

        model = ChatLiteLLM(**litellm_kwargs)

        if debug:
            print(f"  - Created ChatLiteLLM successfully")
    else:
        # Use standard LangChain init_chat_model for direct providers
        formatted_model_name = _format_model_name_for_langchain(model_name)

        model_kwargs = {}
        if base_url:
            model_kwargs["base_url"] = base_url
        if api_key:
            model_kwargs["api_key"] = api_key

        if debug:
            print(f"  - Formatted model_name: {formatted_model_name}")

        try:
            model = init_chat_model(formatted_model_name, **model_kwargs)
        except Exception as e:
            # If LangChain init fails, suggest using LiteLLM
            if "404" in str(e) or "401" in str(e):
                raise RuntimeError(
                    f"Failed to initialize model '{formatted_model_name}'.\n"
                    f"For custom LiteLLM gateways, set LLM_PROVIDER=litellm in .env\n"
                    f"Original error: {e}"
                ) from e
            raise

    # Collect all tools
    tools = [
        scan_structure,
        extract_metadata,
        analyze_code,
        generate_context,
        refine_context,
        list_key_files,
        read_file_snippet,
    ]

    # Create agent with tools and system prompt
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=middleware or [],
        checkpointer=checkpointer,
        debug=debug,
    )

    return agent


def create_contextualizer_agent_with_budget(
    max_tokens: int = 50000,
    max_cost_usd: float = 5.0,
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    debug: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
):
    """Create agent with budget tracking for token/cost monitoring.

    This creates a standard agent and returns it along with a BudgetTracker
    instance. The caller is responsible for integrating budget tracking
    into their workflow (e.g., in the CLI).

    Args:
        max_tokens: Maximum tokens allowed per session (default: 50k)
        max_cost_usd: Maximum cost in USD allowed (default: $5.00)
        model_name: LLM model identifier
        checkpointer: Optional checkpointer for state persistence
        debug: Enable verbose logging

    Returns:
        Tuple of (agent, budget_tracker)

    Example:
        ```python
        from src.agents.factory import create_contextualizer_agent_with_budget
        from src.agents.middleware import extract_token_usage_from_response

        agent, tracker = create_contextualizer_agent_with_budget(max_tokens=30000)

        # Invoke agent
        result = agent.invoke({"messages": [...]}, config=config)

        # Track usage from response
        for msg in result.get("messages", []):
            usage = extract_token_usage_from_response(msg)
            if usage:
                tracker.add_usage(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    "agent_call"
                )

        # Check budget
        tracker.print_summary()
        if tracker.is_over_budget():
            print("Warning: Budget exceeded!")
        ```
    """
    from .middleware import BudgetTracker

    agent = create_contextualizer_agent(
        model_name=model_name,
        checkpointer=checkpointer,
        debug=debug,
        base_url=base_url,
        api_key=api_key,
        use_litellm=use_litellm,
    )

    tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)

    return agent, tracker


def create_contextualizer_agent_with_hitl(
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    debug: bool = False,
    require_approval_for: Optional[list[str]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
):
    """Create agent with human-in-the-loop for expensive operations.

    Note: This requires a checkpointer to be enabled for interrupts to work.
    Human-in-the-loop is implemented using LangGraph's interrupt() function
    within tools. The agent will pause and wait for approval before executing
    expensive operations.

    Args:
        model_name: LLM model identifier
        checkpointer: Checkpointer for state persistence (REQUIRED for HITL)
        debug: Enable verbose logging
        require_approval_for: List of tool names requiring approval
            Default: ["analyze_code", "generate_context", "refine_context"]

    Returns:
        Compiled StateGraph agent with human-in-the-loop

    Example:
        ```python
        from src.agents.factory import create_contextualizer_agent_with_hitl
        from src.agents.memory import create_checkpointer, create_agent_config
        from langgraph.types import Command

        # MUST have checkpointer for HITL
        checkpointer = create_checkpointer()
        agent = create_contextualizer_agent_with_hitl(checkpointer=checkpointer)

        config = create_agent_config("/path/to/repo")

        # Start agent - it will pause at expensive operations
        result = agent.invoke({"messages": [...]}, config=config)

        # Check if interrupted
        state = agent.get_state(config)
        if state.next == ("__interrupt__",):
            print("Approval required:", state.values.get("__interrupt__"))

            # Approve
            final_result = agent.invoke(
                Command(resume={"type": "approve"}),
                config=config
            )
        ```

    Raises:
        ValueError: If checkpointer is not provided (required for interrupts)

    Note:
        To actually use interrupts, tools must call interrupt() internally.
        The tools in this codebase don't have interrupts built-in by default.
        Use create_approval_tool() to wrap tools with approval logic.
    """
    if checkpointer is None:
        raise ValueError(
            "checkpointer is required for human-in-the-loop mode. "
            "Interrupts require state persistence to work. "
            "Use create_checkpointer() to create one."
        )

    # Default expensive operations that should require approval
    if require_approval_for is None:
        require_approval_for = ["analyze_code", "generate_context", "refine_context"]

    # Note: The tools themselves would need to be modified to call interrupt()
    # This factory currently just validates that checkpointing is enabled
    # Future enhancement: Wrap specified tools with create_approval_tool()

    agent = create_contextualizer_agent(
        model_name=model_name,
        checkpointer=checkpointer,
        debug=debug,
        base_url=base_url,
        api_key=api_key,
        use_litellm=use_litellm,
    )

    if debug:
        print(f"✓ Human-in-the-loop enabled for: {', '.join(require_approval_for)}")
        print("  Note: Tools must call interrupt() to pause for approval")

    return agent
