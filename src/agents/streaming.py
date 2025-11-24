"""Streaming output handlers for agent execution."""

import json
from typing import Any, Optional, Iterator
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.status import Status


def stream_agent_execution(
    agent,
    messages: list[dict],
    config: dict,
    verbose: bool = False,
) -> dict[str, Any]:
    """Stream agent execution with real-time feedback.

    Args:
        agent: The LangChain agent to invoke
        messages: List of message dictionaries
        config: Agent configuration (must include thread_id in configurable)
        verbose: Whether to show detailed token usage and metadata

    Returns:
        Final agent result after streaming completes

    Example:
        ```python
        from src.agents.factory import create_contextualizer_agent
        from src.agents.memory import create_checkpointer, create_agent_config
        from src.agents.streaming import stream_agent_execution

        agent = create_contextualizer_agent(checkpointer=create_checkpointer())
        config = create_agent_config("/path/to/repo")

        result = stream_agent_execution(
            agent,
            messages=[{"role": "user", "content": "Generate context for /path/to/repo"}],
            config=config,
            verbose=True
        )
        ```

    Note:
        Uses rich for formatting. Falls back to plain text if rich is not available.
        Shows tool calls, intermediate results, and final response.
    """
    console = Console()

    try:
        # Stream agent execution
        with console.status("[bold green]Agent executing...", spinner="dots") as status:
            for i, chunk in enumerate(
                agent.stream({"messages": messages}, config=config, stream_mode="updates")
            ):
                _process_stream_chunk(chunk, console, status, verbose)

        # Get final result
        result = agent.invoke({"messages": messages}, config=config)

        # Display final message
        final_message = result.get("messages", [])[-1]
        output_content = (
            final_message.content if hasattr(final_message, "content") else str(final_message)
        )

        console.print("\n[bold green]✓ Agent execution complete[/bold green]")
        console.print(Panel(output_content, title="Final Response", border_style="green"))

        return result

    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Agent execution interrupted by user[/bold yellow]")
        raise
    except Exception as e:
        console.print(f"\n[bold red]✗ Agent execution failed: {e}[/bold red]")
        raise


def _process_stream_chunk(
    chunk: dict[str, Any],
    console: Console,
    status: Status,
    verbose: bool,
) -> None:
    """Process a single stream chunk and display it.

    Args:
        chunk: Stream chunk from agent.stream()
        console: Rich console for output
        status: Rich status for spinner updates
        verbose: Whether to show detailed metadata
    """
    # Extract node name and data from chunk
    # LangGraph stream chunks have structure: {node_name: node_data}
    if not chunk:
        return

    for node_name, node_data in chunk.items():
        # Update status message
        status.update(f"[bold cyan]{node_name}[/bold cyan]")

        # Handle tool calls
        if "messages" in node_data:
            messages = node_data["messages"]
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        _display_tool_call(tool_call, console, verbose)

                # Handle tool responses
                if hasattr(msg, "content") and msg.content:
                    # Check if this is a tool response
                    if hasattr(msg, "type") and msg.type == "tool":
                        _display_tool_response(msg, console, verbose)


def _display_tool_call(tool_call: Any, console: Console, verbose: bool) -> None:
    """Display a tool call in a formatted way.

    Args:
        tool_call: Tool call object from agent message
        console: Rich console for output
        verbose: Whether to show full arguments
    """
    tool_name = tool_call.get("name", "unknown")
    tool_args = tool_call.get("args", {})

    # Format arguments nicely
    if verbose:
        args_display = json.dumps(tool_args, indent=2)
        console.print(
            Panel(
                Syntax(args_display, "json", theme="monokai"),
                title=f"🔧 Calling: {tool_name}",
                border_style="cyan",
            )
        )
    else:
        # Compact display - just show tool name and key args
        key_args = _extract_key_args(tool_name, tool_args)
        console.print(f"  [cyan]→ {tool_name}[/cyan] {key_args}")


def _display_tool_response(msg: Any, console: Console, verbose: bool) -> None:
    """Display a tool response in a formatted way.

    Args:
        msg: Tool response message
        console: Rich console for output
        verbose: Whether to show full response content
    """
    tool_name = getattr(msg, "name", "unknown")
    content = msg.content

    # Try to parse content as JSON for prettier display
    try:
        content_dict = json.loads(content) if isinstance(content, str) else content

        # Check for errors
        if isinstance(content_dict, dict) and "error" in content_dict:
            console.print(
                f"  [red]✗ {tool_name} failed:[/red] {content_dict['error']}"
            )
            return

        # Display success
        if verbose:
            formatted = json.dumps(content_dict, indent=2)
            console.print(
                Panel(
                    Syntax(formatted, "json", theme="monokai"),
                    title=f"✓ {tool_name} response",
                    border_style="green",
                )
            )
        else:
            # Compact display
            summary = _summarize_tool_response(tool_name, content_dict)
            console.print(f"  [green]✓ {tool_name}:[/green] {summary}")

    except (json.JSONDecodeError, TypeError):
        # Not JSON, display as-is
        if verbose:
            console.print(
                Panel(
                    str(content)[:500],  # Truncate long responses
                    title=f"✓ {tool_name} response",
                    border_style="green",
                )
            )
        else:
            console.print(f"  [green]✓ {tool_name}[/green] completed")


def _extract_key_args(tool_name: str, args: dict[str, Any]) -> str:
    """Extract and format key arguments for compact display.

    Args:
        tool_name: Name of the tool
        args: Tool arguments dictionary

    Returns:
        Formatted string with key arguments
    """
    # Tool-specific key argument extraction
    if tool_name == "scan_structure":
        return f"({args.get('repo_path', '')})"
    elif tool_name == "extract_metadata":
        return "(analyzing file tree)"
    elif tool_name == "analyze_code":
        return f"(summary: {args.get('user_summary', '')[:30]}...)"
    elif tool_name == "generate_context":
        return "(generating markdown)"
    elif tool_name == "refine_context":
        return f"(request: {args.get('refinement_request', '')[:30]}...)"
    elif tool_name == "list_key_files":
        return "(listing files)"
    elif tool_name == "read_file_snippet":
        path = args.get('file_path', '')
        return f"({path.split('/')[-1] if path else 'file'})"
    else:
        # Generic display
        return f"({len(args)} args)"


def _summarize_tool_response(tool_name: str, response: Any) -> str:
    """Summarize tool response for compact display.

    Args:
        tool_name: Name of the tool
        response: Tool response data

    Returns:
        Summary string
    """
    if not isinstance(response, dict):
        return str(response)[:100]

    # Tool-specific summaries
    if tool_name == "scan_structure":
        total_files = response.get("total_files", 0)
        total_dirs = response.get("total_dirs", 0)
        return f"{total_files} files in {total_dirs} directories"

    elif tool_name == "extract_metadata":
        project_type = response.get("project_type", "Unknown")
        num_deps = len(response.get("dependencies", []))
        return f"type={project_type}, {num_deps} dependencies"

    elif tool_name == "analyze_code":
        patterns = response.get("architecture_patterns", [])
        return f"architecture: {', '.join(patterns[:3])}"

    elif tool_name == "generate_context":
        output_path = response.get("output_path", "")
        return f"saved to {output_path.split('/')[-1] if output_path else 'file'}"

    elif tool_name == "refine_context":
        return "context updated"

    elif tool_name == "list_key_files":
        all_files = response.get("all_key_files", [])
        return f"{len(all_files)} key files found"

    elif tool_name == "read_file_snippet":
        total_lines = response.get("total_lines", 0)
        start = response.get("start_line", 0)
        end = response.get("end_line", 0)
        return f"lines {start}-{end} of {total_lines}"

    else:
        # Generic summary
        return f"{len(response)} fields"


def simple_stream_agent_execution(
    agent,
    messages: list[dict],
    config: dict,
) -> dict[str, Any]:
    """Simple streaming without rich formatting (fallback for non-TTY).

    Args:
        agent: The LangChain agent to invoke
        messages: List of message dictionaries
        config: Agent configuration

    Returns:
        Final agent result
    """
    print("🔄 Agent executing...")

    for chunk in agent.stream({"messages": messages}, config=config, stream_mode="updates"):
        for node_name, node_data in chunk.items():
            print(f"  → {node_name}")

            # Display tool calls
            if "messages" in node_data:
                for msg in node_data["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            print(f"     🔧 {tool_call.get('name', 'unknown')}")

    # Get final result
    result = agent.invoke({"messages": messages}, config=config)

    # Display final message
    final_message = result.get("messages", [])[-1]
    output_content = (
        final_message.content if hasattr(final_message, "content") else str(final_message)
    )

    print("\n✅ Agent execution complete")
    print("\n📋 Final Response:")
    print(output_content)

    return result
