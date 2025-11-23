"""Main CLI entry point for Agentic Contextualizer."""

import click
from pathlib import Path
from .config import Config
from .scanner.structure import StructureScanner
from .scanner.metadata import MetadataExtractor
from .analyzer.code_analyzer import CodeAnalyzer
from .generator.context_generator import ContextGenerator
from .llm.provider import AnthropicProvider


@click.group()
def cli():
    """Agentic Contextualizer - Generate AI-friendly codebase context."""
    pass


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--summary", "-s", required=True, help="Brief description of the project")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--debug", is_flag=True, help="Enable debug output for agent mode")
def generate(repo_path: str, summary: str, output: str | None, mode: str, debug: bool):
    """Generate context for a repository.

    Examples:
        # Pipeline mode (default, deterministic)
        python -m agents.main generate /path/to/repo -s "FastAPI REST API"

        # Agent mode (agentic, uses LangChain agents)
        python -m agents.main generate /path/to/repo -s "FastAPI REST API" --mode agent

        # Agent mode with debug output
        python -m agents.main generate /path/to/repo -s "API" --mode agent --debug
    """
    repo = Path(repo_path).resolve()
    config = Config.from_env()

    if not config.api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set in environment", err=True)
        return 1

    if mode == "agent":
        return _generate_agent_mode(repo, summary, config, debug)
    else:
        return _generate_pipeline_mode(repo, summary, config)


def _generate_pipeline_mode(repo: Path, summary: str, config: Config) -> int:
    """Generate context using deterministic pipeline mode."""
    click.echo(f"🔍 Scanning repository: {repo}")

    # Step 1: Structure Scan
    scanner = StructureScanner(config)
    structure = scanner.scan(repo)
    click.echo(
        f"   Found {structure['total_files']} files in {structure['total_dirs']} directories"
    )

    # Step 2: Metadata Extraction
    extractor = MetadataExtractor()
    metadata = extractor.extract(repo)
    click.echo(f"   Project type: {metadata.project_type or 'Unknown'}")
    click.echo(
        f"   Entry points: {', '.join(metadata.entry_points) if metadata.entry_points else 'None'}"
    )

    # Step 3: Code Analysis (LLM Call 1)
    click.echo(f"\n🤖 Analyzing code with {config.model_name}...")
    llm = AnthropicProvider(config.model_name, config.api_key)
    analyzer = CodeAnalyzer(llm)
    analysis = analyzer.analyze(repo, metadata, structure["tree"], summary)
    click.echo(f"   Architecture: {', '.join(analysis.architecture_patterns)}")

    # Step 4: Context Generation (LLM Call 2)
    click.echo("\n📝 Generating context file...")
    generator = ContextGenerator(llm, config.output_dir)
    output_path = generator.generate(metadata, analysis, summary, config.model_name)

    click.echo(f"\n✅ Context generated: {output_path}")
    return 0


def _generate_agent_mode(repo: Path, summary: str, config: Config, debug: bool) -> int:
    """Generate context using agent mode."""
    from .factory import create_contextualizer_agent
    from .memory import create_checkpointer, create_agent_config
    from .observability import configure_tracing, is_tracing_enabled

    click.echo(f"🤖 Agent mode: Analyzing repository: {repo}")

    # Configure tracing
    configure_tracing()

    # Create checkpointer for state persistence
    checkpointer = create_checkpointer()

    # Create agent
    agent = create_contextualizer_agent(
        model_name=f"anthropic:{config.model_name}", checkpointer=checkpointer, debug=debug
    )

    # Create agent configuration with thread ID
    agent_config = create_agent_config(str(repo))

    # Build user message
    user_message = f"Generate context for {repo}. User description: {summary}"

    click.echo(f"   Thread ID: {agent_config['configurable']['thread_id']}")
    if is_tracing_enabled():
        click.echo(f"   Tracing: Enabled")

    # Invoke agent
    try:
        click.echo("\n🔄 Agent executing...")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}, config=agent_config
        )

        # Extract final message
        final_message = result.get("messages", [])[-1]
        output_content = final_message.content if hasattr(final_message, "content") else str(final_message)

        click.echo("\n📋 Agent Response:")
        click.echo(output_content)
        click.echo("\n✅ Agent execution complete")

        return 0

    except Exception as e:
        click.echo(f"\n❌ Agent execution failed: {e}", err=True)
        if debug:
            import traceback

            traceback.print_exc()
        return 1


@cli.command()
@click.argument("context_file", type=click.Path(exists=True))
@click.option("--request", "-r", required=True, help="What to change")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--debug", is_flag=True, help="Enable debug output for agent mode")
def refine(context_file: str, request: str, mode: str, debug: bool):
    """Refine an existing context file.

    Examples:
        # Pipeline mode (default)
        python -m agents.main refine contexts/myapp/context.md -r "Add auth details"

        # Agent mode (uses conversation history from generation)
        python -m agents.main refine contexts/myapp/context.md -r "Add auth" --mode agent
    """
    context_path = Path(context_file)
    config = Config.from_env()

    if not config.api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set in environment", err=True)
        return 1

    if mode == "agent":
        return _refine_agent_mode(context_path, request, config, debug)
    else:
        return _refine_pipeline_mode(context_path, request, config)


def _refine_pipeline_mode(context_path: Path, request: str, config: Config) -> int:
    """Refine context using deterministic pipeline mode."""
    click.echo(f"🔄 Refining: {context_path}")

    llm = AnthropicProvider(config.model_name, config.api_key)
    generator = ContextGenerator(llm, config.output_dir)

    updated_path = generator.refine(context_path, request)

    click.echo(f"✅ Context updated: {updated_path}")
    return 0


def _refine_agent_mode(context_path: Path, request: str, config: Config, debug: bool) -> int:
    """Refine context using agent mode."""
    from .factory import create_contextualizer_agent
    from .memory import create_checkpointer, create_agent_config
    from .observability import configure_tracing, is_tracing_enabled

    click.echo(f"🤖 Agent mode: Refining context: {context_path}")

    # Try to infer repo path from context file location
    # Context files are stored as contexts/{repo-name}/context.md
    repo_path = context_path.parent.parent  # Go up two levels

    # Configure tracing
    configure_tracing()

    # Create checkpointer (same instance will restore previous conversation)
    checkpointer = create_checkpointer()

    # Create agent
    agent = create_contextualizer_agent(
        model_name=f"anthropic:{config.model_name}", checkpointer=checkpointer, debug=debug
    )

    # Use same thread ID as generation (based on repo path)
    agent_config = create_agent_config(str(repo_path))

    # Build user message
    user_message = f"Refine the context file at {context_path}. Refinement request: {request}"

    click.echo(f"   Thread ID: {agent_config['configurable']['thread_id']}")
    click.echo(f"   (Using same thread as generation for context continuity)")
    if is_tracing_enabled():
        click.echo(f"   Tracing: Enabled")

    # Invoke agent
    try:
        click.echo("\n🔄 Agent executing...")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}, config=agent_config
        )

        # Extract final message
        final_message = result.get("messages", [])[-1]
        output_content = final_message.content if hasattr(final_message, "content") else str(final_message)

        click.echo("\n📋 Agent Response:")
        click.echo(output_content)
        click.echo("\n✅ Agent execution complete")

        return 0

    except Exception as e:
        click.echo(f"\n❌ Agent execution failed: {e}", err=True)
        if debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    cli()
