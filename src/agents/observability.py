"""LangSmith tracing and observability configuration."""

import logging
import os
from typing import Optional

from langsmith import Client, traceable

logger = logging.getLogger(__name__)


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
    """
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.info("LangSmith API key not found. Tracing disabled.")
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    os.environ["LANGSMITH_PROJECT"] = project_name

    if enabled is not None:
        os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    elif "LANGSMITH_TRACING" not in os.environ:
        os.environ["LANGSMITH_TRACING"] = "true"

    # Verify connection — non-fatal, tracing still works if project doesn't exist yet
    try:
        client = Client()
        project = client.read_project(project_name=project_name)
        trace_url = (
            f"https://smith.langchain.com/o/{project.tenant_id}"
            f"/projects/p/{project.id}"
        )
        logger.info("LangSmith tracing enabled — Project: %s", project_name)
        logger.info("  Traces: %s", trace_url)
    except Exception as e:
        # Project may not exist yet — LangSmith creates it on first trace
        logger.info(
            "LangSmith tracing enabled (project lookup skipped: %s)", e
        )


def disable_tracing() -> None:
    """Disable LangSmith tracing."""
    os.environ["LANGSMITH_TRACING"] = "false"
    logger.info("LangSmith tracing disabled")


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently enabled.

    Returns:
        True if tracing is enabled, False otherwise
    """
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"


def trace_function(name: Optional[str] = None, **trace_kwargs):
    """Decorator to trace a function with LangSmith.

    Thin wrapper around langsmith.traceable that provides consistent
    naming. The @traceable decorator already no-ops when tracing is
    disabled, so no manual gating is needed.

    Args:
        name: Custom name for the trace (default: function name)
        **trace_kwargs: Additional arguments passed to @traceable
    """

    def decorator(func):
        trace_name = name or func.__name__
        return traceable(name=trace_name, **trace_kwargs)(func)

    return decorator


def get_trace_url(run_id: str) -> str:
    """Get the URL to view a specific trace in LangSmith.

    Args:
        run_id: The run ID from an agent invocation

    Returns:
        URL to the trace in LangSmith UI, or empty string if unavailable.
    """
    if not is_tracing_enabled():
        return ""

    try:
        client = Client()
        project_name = os.getenv("LANGSMITH_PROJECT", "agentic-contextualizer")
        project = client.read_project(project_name=project_name)
        return (
            f"https://smith.langchain.com/o/{project.tenant_id}"
            f"/projects/p/{project.id}?trace={run_id}"
        )
    except Exception:
        return ""
