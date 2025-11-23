"""Human-in-the-loop utilities for agent workflows."""

from typing import Any, Dict, Callable, Optional
from langchain_core.tools import tool
from langgraph.types import interrupt


def create_approval_tool(
    base_tool: Callable,
    approval_message: Optional[str] = None,
    require_approval: bool = True,
) -> Callable:
    """Wrap a tool to require human approval before execution.

    Args:
        base_tool: The original tool function to wrap
        approval_message: Custom message to show for approval
        require_approval: If False, tool runs without approval (for testing)

    Returns:
        Wrapped tool function that requests approval

    Example:
        ```python
        from src.agents.middleware.human_in_the_loop import create_approval_tool
        from src.agents.tools.repository_tools import generate_context

        # Wrap generate_context to require approval
        generate_with_approval = create_approval_tool(
            generate_context,
            approval_message="About to generate context file. Approve?"
        )
        ```

    Note:
        Requires checkpointer to be enabled on the agent for interrupts to work.
    """

    def wrapped_tool(*args, **kwargs):
        """Tool wrapper that requests approval before execution."""
        if not require_approval:
            # Skip approval for testing
            return base_tool(*args, **kwargs)

        # Prepare approval request
        tool_name = getattr(base_tool, "name", base_tool.__name__)
        message = approval_message or f"Approve {tool_name} with these arguments?"

        approval_data = {
            "tool": tool_name,
            "message": message,
            "args": kwargs if kwargs else {"positional_args": args},
            "action": "approve_or_edit",
        }

        # Request approval via interrupt
        response = interrupt(approval_data)

        # Process response
        if response is None or response.get("type") == "reject":
            raise ToolRejectedError(f"Tool {tool_name} was rejected by user")

        elif response.get("type") == "edit":
            # User provided edited arguments
            edited_kwargs = response.get("args", {})
            return base_tool(**edited_kwargs)

        elif response.get("type") == "approve":
            # Use original arguments
            return base_tool(*args, **kwargs)

        else:
            # Default to approve for backwards compatibility
            return base_tool(*args, **kwargs)

    # Preserve tool metadata
    wrapped_tool.__name__ = getattr(base_tool, "name", base_tool.__name__)
    wrapped_tool.__doc__ = getattr(base_tool, "__doc__", "")

    return wrapped_tool


def get_expensive_tool_approval(
    tool_name: str,
    estimated_cost: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Request approval for expensive LLM operations.

    Args:
        tool_name: Name of the tool requiring approval
        estimated_cost: Estimated cost in USD (if known)
        **kwargs: Tool arguments to show in approval request

    Returns:
        Approval response from user

    Example:
        ```python
        from src.agents.middleware.human_in_the_loop import get_expensive_tool_approval

        # Inside a tool function
        approval = get_expensive_tool_approval(
            "analyze_code",
            estimated_cost=0.50,
            repo_path="/path/to/repo",
            user_summary="FastAPI REST API"
        )

        if approval.get("type") != "approve":
            return {"error": "Operation cancelled by user"}
        ```
    """
    approval_data = {
        "tool": tool_name,
        "type": "expensive_operation",
        "message": f"⚠️  {tool_name} will call the LLM (expensive operation)",
        "args": kwargs,
        "action": "Please approve to continue",
    }

    if estimated_cost is not None:
        approval_data["estimated_cost_usd"] = estimated_cost

    return interrupt(approval_data)


class ToolRejectedError(Exception):
    """Raised when a tool is rejected by the user."""

    pass


# Pre-configured approval messages for common operations
APPROVAL_MESSAGES = {
    "analyze_code": "📊 Analyze codebase with LLM? This will use ~2-5k tokens.",
    "generate_context": "📝 Generate final context file? This will use ~3-8k tokens.",
    "refine_context": "🔄 Refine existing context? This will use ~2-4k tokens.",
}


def should_approve_expensive_operation(operation: str, **kwargs) -> bool:
    """Helper to request approval for expensive operations.

    Args:
        operation: Name of the operation
        **kwargs: Operation parameters

    Returns:
        True if approved, False if rejected

    Raises:
        ToolRejectedError: If operation is rejected

    Example:
        ```python
        if not should_approve_expensive_operation("analyze_code", repo="/path"):
            return {"error": "Operation cancelled"}
        ```
    """
    approval_data = {
        "operation": operation,
        "message": APPROVAL_MESSAGES.get(
            operation, f"Approve {operation}?"
        ),
        "parameters": kwargs,
    }

    response = interrupt(approval_data)

    if response is None or response.get("type") == "reject":
        raise ToolRejectedError(f"Operation {operation} was rejected")

    return response.get("type") == "approve"
