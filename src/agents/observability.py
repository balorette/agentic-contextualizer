"""LangSmith tracing and observability configuration."""

import os
from typing import Optional
from functools import wraps
from langsmith import traceable, Client


def configure_tracing(
    project_name: str = "agentic-contextualizer",
    api_key: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> None:
    """Configure LangSmith tracing for the application.

    This should be called once at application startup to enable
    automatic tracing of all agent invocations and tool calls.

    Args:
        project_name: LangSmith project name (default: "agentic-contextualizer")
        api_key: LangSmith API key (if not in environment)
        enabled: Explicitly enable/disable tracing (if not in environment)

    Environment Variables:
        LANGSMITH_API_KEY: API key for LangSmith (required for tracing)
        LANGSMITH_TRACING: Set to "true" to enable tracing
        LANGSMITH_PROJECT: Project name for organizing traces

    Example:
        ```python
        from src.agents.observability import configure_tracing

        # Configure at startup
        configure_tracing(project_name="my-contextualizer")

        # Now all agent invocations will be traced
        agent = create_contextualizer_agent()
        result = agent.invoke({"messages": [...]})
        ```

    Note:
        If LANGSMITH_API_KEY is not set, tracing will be disabled
        and a warning will be printed. This allows the application
        to run without LangSmith configured.
    """
    # Set API key if provided
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key

    # Check if API key is available
    api_key_env = os.getenv("LANGSMITH_API_KEY")
    if not api_key_env:
        print(
            "⚠️  LangSmith API key not found. Tracing disabled.\n"
            "   To enable tracing, set LANGSMITH_API_KEY environment variable.\n"
            "   Get your API key at: https://smith.langchain.com"
        )
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    # Set project name
    os.environ["LANGSMITH_PROJECT"] = project_name

    # Enable tracing if specified or not already set
    if enabled is not None:
        os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    elif "LANGSMITH_TRACING" not in os.environ:
        os.environ["LANGSMITH_TRACING"] = "true"

    # Verify connection
    try:
        client = Client()
        # Try to verify the connection works
        print(f"✅ LangSmith tracing enabled")
        print(f"   Project: {project_name}")
        project = client.read_project(project_name)
        project_id = project["id"]
        print(f"   View traces at: https://smith.langchain.com/o/{client.info()['tenant_id']}/projects/p/{project_id}")
    except Exception as e:
        print(f"   (Could not retrieve project ID for direct link: {e})")
    print(f"⚠️  LangSmith connection failed: {e}")
    print("   Tracing will be disabled.")
    os.environ["LANGSMITH_TRACING"] = "false"



def disable_tracing() -> None:
    """Disable LangSmith tracing.

    Useful for testing or when you want to temporarily disable
    tracing without changing environment variables.

    Example:
        ```python
        from src.agents.observability import disable_tracing

        disable_tracing()
        # Now agent invocations won't be traced
        ```
    """
    os.environ["LANGSMITH_TRACING"] = "false"
    print("LangSmith tracing disabled")


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently enabled.

    Returns:
        True if tracing is enabled, False otherwise

    Example:
        ```python
        from src.agents.observability import is_tracing_enabled

        if is_tracing_enabled():
            print("Traces will be sent to LangSmith")
        ```
    """
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"


def trace_function(name: Optional[str] = None, **trace_kwargs):
    """Decorator to trace a function with LangSmith.

    This is a thin wrapper around langsmith.traceable that provides
    consistent naming and metadata for traced functions.

    Args:
        name: Custom name for the trace (default: function name)
        **trace_kwargs: Additional arguments passed to @traceable

    Example:
        ```python
        from src.agents.observability import trace_function

        @trace_function(name="my_analysis")
        def analyze_repository(repo_path: str) -> dict:
            # Function will be traced with name "my_analysis"
            return {"analysis": "results"}

        @trace_function(tags=["expensive"])
        def expensive_operation():
            # Function traced with tags
            pass
        ```

    Note:
        Functions are only traced if tracing is enabled.
        If disabled, this decorator has no effect (no overhead).
    """

    def decorator(func):
        # Use function name if no custom name provided
        trace_name = name or func.__name__

        # Apply traceable decorator
        traced_func = traceable(name=trace_name, **trace_kwargs)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Only trace if enabled
            if is_tracing_enabled():
                return traced_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_trace_url(run_id: str) -> str:
    """Get the URL to view a specific trace in LangSmith.

    Args:
        run_id: The run ID from an agent invocation

    Returns:
        URL to the trace in LangSmith UI

    Example:
        ```python
        result = agent.invoke({"messages": [...]})
        run_id = result.get("run_id")
        if run_id:
            url = get_trace_url(run_id)
            print(f"View trace: {url}")
        ```

    Note:
        Returns empty string if tracing is disabled or client can't be initialized.
    """
    if not is_tracing_enabled():
        return ""

    try:
        client = Client()
        tenant_id = client.info().get("tenant_id", "")
        return f"https://smith.langchain.com/o/{tenant_id}/projects/p?trace={run_id}"
    except Exception:
        return ""


# Auto-configure tracing on module import (reads from environment)
# This allows tracing to work automatically if env vars are set
if os.getenv("LANGSMITH_API_KEY"):
    configure_tracing()
