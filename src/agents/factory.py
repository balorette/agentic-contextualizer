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
):
    """Create a repository contextualizer agent.

    Args:
        model_name: LLM model identifier (default: Claude Sonnet 4.5)
        middleware: Optional list of middleware instances
        checkpointer: Optional checkpointer for state persistence
        debug: Enable verbose logging for graph execution

    Returns:
        Compiled StateGraph agent ready for invocation

    Example:
        ```python
        from src.agents.factory import create_contextualizer_agent

        agent = create_contextualizer_agent()

        # Generate context
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Generate context for /path/to/repo. It's a FastAPI REST API."
            }]
        })

        # Refine context
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Refine /path/to/context.md: add more auth details"
            }]
        })
        ```
    """
    # Initialize chat model
    model = init_chat_model(model_name)

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
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    debug: bool = False,
):
    """Create agent with budget middleware for token tracking.

    Note: BudgetMiddleware will be implemented in Phase 4.
    For now, this is a placeholder that creates a standard agent.

    Args:
        max_tokens: Maximum tokens allowed per session (default: 50k)
        model_name: LLM model identifier
        checkpointer: Optional checkpointer for state persistence
        debug: Enable verbose logging

    Returns:
        Compiled StateGraph agent with budget tracking
    """
    # TODO: Implement BudgetMiddleware in Phase 4
    # For now, return standard agent
    return create_contextualizer_agent(
        model_name=model_name, checkpointer=checkpointer, debug=debug
    )
