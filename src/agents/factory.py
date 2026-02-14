"""Agent factory for creating LangChain agents."""

from typing import Optional
from langchain.agents import create_agent

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


AGENT_SYSTEM_PROMPT = """You are a repository context generator. Analyze codebases and produce context documentation.

## Workflow

For initial generation, call tools in this order:
1. scan_structure - get file listing
2. extract_metadata - get project type, deps, entry points
3. analyze_code - deep LLM analysis (pass file_list from step 1)
4. generate_context - produce final markdown

For refinement: use refine_context with the request.

Keep LLM calls ≤ 2. If a tool returns "error", report it. If >10k files, suggest a subdirectory.
"""


def create_contextualizer_agent(
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    middleware: Optional[list] = None,
    checkpointer: Optional[object] = None,
    debug: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
    config: Optional["Config"] = None,
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
    from .llm.chat_model_factory import build_chat_model, build_token_middleware

    if config is None:
        config = Config.from_env()

    # Derive use_litellm from config when caller didn't explicitly set it
    if not use_litellm and config.llm_provider == "litellm":
        use_litellm = True
    model = build_chat_model(
        config=config,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        use_litellm=use_litellm,
        debug=debug,
    )

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

    budget_mw = build_token_middleware(config, model_name)
    all_middleware = [budget_mw] + (middleware or [])

    # Create agent with tools and system prompt
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=all_middleware,
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
    config: Optional["Config"] = None,
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
        config=config,
    )

    tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)

    return agent, tracker


def create_contextualizer_agent_with_checkpointer(
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    debug: bool = False,
    require_approval_for: Optional[list[str]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
    config: Optional["Config"] = None,
):
    """Create agent with state persistence via checkpointer.

    This creates an agent with checkpointing enabled, which is a
    prerequisite for future human-in-the-loop approval gates.

    Note: This does NOT currently implement automatic approval gates.
    Tools do not call interrupt() by default. See docs/roadmap/hitl-approval-gates.md
    for the planned HITL implementation.

    # ROADMAP: Wire create_approval_tool() wrappers for tools listed in
    # require_approval_for. See docs/roadmap/hitl-approval-gates.md

    Args:
        model_name: LLM model identifier
        checkpointer: Checkpointer for state persistence (REQUIRED)
        debug: Enable verbose logging
        require_approval_for: Reserved for future HITL implementation
        base_url: Optional custom API endpoint URL
        api_key: Optional API key
        use_litellm: Force use of ChatLiteLLM

    Returns:
        Compiled StateGraph agent with checkpointing

    Raises:
        ValueError: If checkpointer is not provided
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
        config=config,
    )

    if debug:
        print(f"✓ Checkpointing enabled. Approval gates reserved for: {', '.join(require_approval_for)}")
        print("  Note: HITL approval gates are not yet wired. See docs/roadmap/hitl-approval-gates.md")

    return agent
