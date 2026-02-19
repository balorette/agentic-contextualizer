"""Main CLI entry point for Agentic Contextualizer."""

import logging
import subprocess
import click
import yaml

logger = logging.getLogger(__name__)
from pathlib import Path
from .config import Config
from .scanner.structure import StructureScanner
from .scanner.metadata import MetadataExtractor
from .analyzer.code_analyzer import CodeAnalyzer
from .generator.context_generator import ContextGenerator
from .llm.provider import create_llm_provider
from .scoper.discovery import extract_keywords, search_relevant_files
from .scoper.scoped_analyzer import ScopedAnalyzer
from .scoper.scoped_generator import ScopedGenerator
from .repo_resolver import resolve_repo


def _extract_repo_from_context(context_path: Path) -> str | None:
    """Extract source_repo from context file frontmatter.

    Args:
        context_path: Path to context.md file

    Returns:
        Source repo path from frontmatter, or None if not found
    """
    try:
        content = context_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None

        # Find end of frontmatter
        end_marker = content.find("---", 3)
        if end_marker == -1:
            return None

        frontmatter_text = content[3:end_marker].strip()
        frontmatter = yaml.safe_load(frontmatter_text)

        if isinstance(frontmatter, dict):
            return frontmatter.get("source_repo")
    except yaml.YAMLError:
        logger.warning("Failed to parse frontmatter from %s", context_path)
    except OSError as e:
        logger.warning("Failed to read context file %s: %s", context_path, e)
    except Exception:
        logger.warning("Unexpected error extracting repo from %s", context_path, exc_info=True)
    return None


def _validate_api_key(config: Config) -> tuple[bool, str]:
    """Validate that required API key is set for the chosen provider/model.

    Args:
        config: Application configuration

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    # Strip provider prefix (e.g. "openai:gpt-4o" -> "gpt-4o") for matching
    from .llm.provider import _strip_provider_prefix
    normalized_model = _strip_provider_prefix(config.model_name)

    # Local models don't need API keys
    if normalized_model.startswith(("ollama", "lmstudio")):
        return True, ""

    # When using a custom gateway (base_url), the gateway handles auth.
    # Any API key (even a gateway token) is sufficient.
    if config.llm_provider == "litellm":
        if config.api_base_url:
            # Gateway mode - any key set is fine, the gateway handles routing
            has_any_key = config.openai_api_key or config.anthropic_api_key or config.google_api_key or config.api_key
            if not has_any_key:
                return False, "Error: No API key set. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for gateway auth."
            return True, ""

        # Direct LiteLLM (no gateway) - need provider-specific keys
        if normalized_model.startswith("gpt-") or normalized_model.startswith("o1"):
            if not config.openai_api_key:
                return False, "Error: OPENAI_API_KEY not set. Required for OpenAI models with LiteLLM."
        elif normalized_model.startswith("claude"):
            if not config.anthropic_api_key:
                return False, "Error: ANTHROPIC_API_KEY not set. Required for Claude models with LiteLLM."
        elif normalized_model.startswith(("gemini", "vertex")):
            if not config.google_api_key:
                return False, "Error: GOOGLE_API_KEY not set. Required for Google models with LiteLLM."
        return True, ""

    # For direct Anthropic provider
    if not config.anthropic_api_key and not config.api_key:
        return False, "Error: ANTHROPIC_API_KEY not set in environment"

    return True, ""


def _prepare_config(provider: str | None, model: str | None) -> Config | None:
    """Build config with CLI overrides and validate API key.

    Returns Config on success, None on validation failure (error already printed).
    """
    cli_overrides = {}
    if provider:
        cli_overrides["llm_provider"] = provider
    if model:
        cli_overrides["model_name"] = model

    config = Config.from_env(cli_overrides=cli_overrides)

    is_valid, error_msg = _validate_api_key(config)
    if not is_valid:
        click.echo(error_msg, err=True)
        return None
    return config


def _run_agent(agent, user_message: str, agent_config: dict, stream: bool, debug: bool) -> int:
    """Execute an agent with invoke or stream, returning exit code.

    Handles TTY detection for streaming and common exception handling.
    """
    # Inject Langfuse callback handler (no-op when Langfuse is not configured)
    from .observability import get_langfuse_callback_handler

    langfuse_handler = get_langfuse_callback_handler()
    if langfuse_handler:
        existing_callbacks = agent_config.get("callbacks", [])
        agent_config["callbacks"] = existing_callbacks + [langfuse_handler]

    try:
        if stream:
            from .streaming import stream_agent_execution, simple_stream_agent_execution
            import sys

            if sys.stdout.isatty():
                stream_agent_execution(
                    agent,
                    messages=[{"role": "user", "content": user_message}],
                    config=agent_config,
                    verbose=debug,
                )
            else:
                simple_stream_agent_execution(
                    agent,
                    messages=[{"role": "user", "content": user_message}],
                    config=agent_config,
                )
        else:
            click.echo("\n🔄 Agent executing...")
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=agent_config,
            )
            messages = result.get("messages", [])
            if not messages:
                click.echo("\n⚠️ Agent returned no messages.", err=True)
                return 1
            final_message = messages[-1]
            output_content = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )
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


@click.group()
def cli():
    """Agentic Contextualizer - Generate AI-friendly codebase context."""
    pass


@cli.command()
@click.argument("source")
@click.option("--summary", "-s", required=True, help="Brief description of the project")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--provider", help="LLM provider: anthropic or litellm")
@click.option("--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
@click.option("--debug", is_flag=True, help="Enable debug output for agent mode")
@click.option("--stream", is_flag=True, help="Enable streaming output for agent mode (real-time feedback)")
@click.pass_context
def generate(ctx, source: str, summary: str, output: str | None, mode: str, provider: str | None, model: str | None, debug: bool, stream: bool):
    """Generate context for a repository.

    SOURCE can be a local path or a GitHub URL.

    Examples:
        # From local path (pipeline mode, default)
        python -m agents.main generate /path/to/repo -s "FastAPI REST API"

        # From GitHub URL
        python -m agents.main generate https://github.com/owner/repo -s "FastAPI REST API"

        # Agent mode (agentic, uses LangChain agents)
        python -m agents.main generate /path/to/repo -s "FastAPI REST API" --mode agent

        # Agent mode with streaming output (real-time feedback)
        python -m agents.main generate /path/to/repo -s "API" --mode agent --stream
    """
    config = _prepare_config(provider, model)
    if config is None:
        ctx.exit(1)
        return

    try:
        with resolve_repo(source) as repo:
            if mode == "agent":
                rc = _generate_agent_mode(repo, summary, config, debug, stream)
            else:
                rc = _generate_pipeline_mode(repo, summary, config)
            if rc:
                ctx.exit(rc)
    except subprocess.CalledProcessError as e:
        click.echo("Error: Failed to clone repository. Check the URL and your git credentials.", err=True)
        if debug:
            click.echo(f"Details: {e.stderr}", err=True)
        ctx.exit(1)
    except subprocess.TimeoutExpired:
        click.echo("Error: Clone timed out. The repository may be too large or the network is slow.", err=True)
        ctx.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


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
    llm = create_llm_provider(config)
    analyzer = CodeAnalyzer(llm)
    analysis = analyzer.analyze(repo, metadata, structure["tree"], summary)
    click.echo(f"   Architecture: {', '.join(analysis.architecture_patterns)}")

    # Step 4: Context Generation (LLM Call 2)
    click.echo("\n📝 Generating context file...")
    generator = ContextGenerator(llm, config.output_dir)
    output_path = generator.generate(metadata, analysis, summary, config.model_name)

    click.echo(f"\n✅ Context generated: {output_path}")
    return 0


def _generate_agent_mode(repo: Path, summary: str, config: Config, debug: bool, stream: bool) -> int:
    """Generate context using agent mode."""
    from .factory import create_contextualizer_agent
    from .memory import create_checkpointer, create_agent_config
    from .observability import configure_tracing, is_tracing_enabled
    from .observability import configure_langfuse_tracing, is_langfuse_tracing_enabled
    from .tools.repository_tools import set_tool_config, set_allowed_repo_root

    click.echo(f"🤖 Agent mode: Analyzing repository: {repo}")

    if stream:
        click.echo("   Streaming: Enabled")

    # Set tool config so tools use the same config as the agent
    # This ensures CLI flags propagate to tools
    set_tool_config(config)
    set_allowed_repo_root(repo)

    # Configure tracing (LangSmith + Langfuse)
    configure_tracing()
    configure_langfuse_tracing()

    # Create checkpointer for state persistence
    checkpointer = create_checkpointer()

    # Resolve API key for the selected model
    from .llm.provider import _resolve_api_key_for_model
    api_key = _resolve_api_key_for_model(config.model_name, config)

    # Create agent
    agent = create_contextualizer_agent(
        model_name=config.model_name,
        checkpointer=checkpointer,
        debug=debug,
        base_url=config.api_base_url,
        api_key=api_key,
        config=config,
    )

    # Create agent configuration with thread ID
    agent_config = create_agent_config(str(repo))

    # Build user message
    user_message = f"Generate context for {repo}. User description: {summary}"

    click.echo(f"   Thread ID: {agent_config['configurable']['thread_id']}")
    if is_tracing_enabled():
        click.echo("   Tracing: LangSmith enabled")
    if is_langfuse_tracing_enabled():
        click.echo("   Tracing: Langfuse enabled")

    return _run_agent(agent, user_message, agent_config, stream, debug)


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
@click.option("--provider", help="LLM provider: anthropic or litellm")
@click.option("--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
@click.option("--debug", is_flag=True, help="Enable debug output for agent mode")
@click.option("--stream", is_flag=True, help="Enable streaming output for agent mode (real-time feedback)")
@click.pass_context
def refine(ctx, context_file: str, request: str, mode: str, provider: str | None, model: str | None, debug: bool, stream: bool):
    """Refine an existing context file.

    Examples:
        # Pipeline mode (default)
        python -m agents.main refine contexts/myapp/context.md -r "Add auth details"

        # Agent mode (uses conversation history from generation)
        python -m agents.main refine contexts/myapp/context.md -r "Add auth" --mode agent

        # Agent mode with streaming output
        python -m agents.main refine contexts/myapp/context.md -r "Add auth" --mode agent --stream
    """
    context_path = Path(context_file)
    config = _prepare_config(provider, model)
    if config is None:
        ctx.exit(1)
        return

    if mode == "agent":
        rc = _refine_agent_mode(context_path, request, config, debug, stream)
    else:
        rc = _refine_pipeline_mode(context_path, request, config)
    if rc:
        ctx.exit(rc)


def _refine_pipeline_mode(context_path: Path, request: str, config: Config) -> int:
    """Refine context using deterministic pipeline mode."""
    click.echo(f"🔄 Refining: {context_path}")

    llm = create_llm_provider(config)
    generator = ContextGenerator(llm, config.output_dir)

    updated_path = generator.refine(context_path, request)

    click.echo(f"✅ Context updated: {updated_path}")
    return 0


def _refine_agent_mode(context_path: Path, request: str, config: Config, debug: bool, stream: bool) -> int:
    """Refine context using agent mode."""
    from .factory import create_contextualizer_agent
    from .memory import create_checkpointer, create_agent_config
    from .observability import configure_tracing, is_tracing_enabled
    from .observability import configure_langfuse_tracing, is_langfuse_tracing_enabled
    from .tools.repository_tools import set_tool_config, set_allowed_repo_root

    click.echo(f"🤖 Agent mode: Refining context: {context_path}")

    # Set tool config for consistency
    set_tool_config(config)

    if stream:
        click.echo("   Streaming: Enabled")

    # Try to get repo path from context file frontmatter, fall back to parent dir
    source_repo = _extract_repo_from_context(context_path)
    if source_repo and Path(source_repo).is_dir():
        repo_path = Path(source_repo).resolve()
    else:
        # Fallback: context files are stored as contexts/{repo-name}/context.md
        repo_path = context_path.parent

    # Restrict tool file access to the repo being refined
    if repo_path.is_dir():
        set_allowed_repo_root(repo_path)

    # Configure tracing (LangSmith + Langfuse)
    configure_tracing()
    configure_langfuse_tracing()

    # Create checkpointer (same instance will restore previous conversation)
    checkpointer = create_checkpointer()

    # Resolve API key for the selected model
    from .llm.provider import _resolve_api_key_for_model
    api_key = _resolve_api_key_for_model(config.model_name, config)

    # Create agent
    agent = create_contextualizer_agent(
        model_name=config.model_name,
        checkpointer=checkpointer,
        debug=debug,
        base_url=config.api_base_url,
        api_key=api_key,
        config=config,
    )

    # Use same thread ID as generation (based on repo path)
    agent_config = create_agent_config(str(repo_path))

    # Build user message
    user_message = f"Refine the context file at {context_path}. Refinement request: {request}"

    click.echo(f"   Thread ID: {agent_config['configurable']['thread_id']}")
    click.echo(f"   (Using same thread as generation for context continuity)")
    if is_tracing_enabled():
        click.echo("   Tracing: LangSmith enabled")
    if is_langfuse_tracing_enabled():
        click.echo("   Tracing: Langfuse enabled")

    return _run_agent(agent, user_message, agent_config, stream, debug)


@cli.command()
@click.argument("source")
@click.option("--question", "-q", required=True, help="Question/topic to scope to")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["pipeline", "agent"]),
    default="pipeline",
    help="Execution mode: pipeline (deterministic) or agent (agentic)",
)
@click.option("--provider", help="LLM provider: anthropic or litellm")
@click.option("--model", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--stream", is_flag=True, help="Enable streaming output (agent mode)")
@click.pass_context
def scope(ctx, source: str, question: str, output: str | None, mode: str, provider: str | None, model: str | None, debug: bool, stream: bool):
    """Generate scoped context for a specific question.

    SOURCE can be:
    - A repository path: scopes directly from the repo
    - A context.md file: uses existing context as starting point
    - A GitHub URL: clones the repo, scopes, then cleans up

    Examples:
        # Scope from repo
        python -m agents.main scope /path/to/repo -q "weather functionality"

        # Scope from GitHub URL
        python -m agents.main scope https://github.com/owner/repo -q "auth flow"

        # Scope from existing context
        python -m agents.main scope contexts/repo/context.md -q "auth flow"

        # Agent mode with streaming
        python -m agents.main scope /path/to/repo -q "auth" --mode agent --stream
    """
    config = _prepare_config(provider, model)
    if config is None:
        ctx.exit(1)
        return

    # Context files are handled directly (no resolve_repo needed)
    source_path = Path(source)
    if source_path.is_file() and source_path.suffix == ".md":
        source_path = source_path.resolve()
        if mode == "agent":
            rc = _scope_agent_mode(source_path, question, config, True, debug, stream)
        else:
            rc = _scope_pipeline_mode(source_path, question, config, True, output)
        if rc:
            ctx.exit(rc)
        return

    # Repo path or GitHub URL - resolve it
    try:
        with resolve_repo(source) as repo:
            if mode == "agent":
                rc = _scope_agent_mode(repo, question, config, False, debug, stream)
            else:
                rc = _scope_pipeline_mode(repo, question, config, False, output)
            if rc:
                ctx.exit(rc)
    except subprocess.CalledProcessError as e:
        click.echo("Error: Failed to clone repository. Check the URL and your git credentials.", err=True)
        if debug:
            click.echo(f"Details: {e.stderr}", err=True)
        ctx.exit(1)
    except subprocess.TimeoutExpired:
        click.echo("Error: Clone timed out. The repository may be too large or the network is slow.", err=True)
        ctx.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _scope_pipeline_mode(
    source_path: Path,
    question: str,
    config: Config,
    is_context_file: bool,
    output: str | None,
) -> int:
    """Generate scoped context using pipeline mode."""
    click.echo(f"🔍 Scoping: {question}")

    # Determine repo path
    if is_context_file:
        # Extract repo path from context file's frontmatter
        source_repo = _extract_repo_from_context(source_path)
        if source_repo and Path(source_repo).exists():
            repo_path = Path(source_repo)
            repo_name = repo_path.name
            click.echo(f"   Source: context file (repo: {repo_name})")
        else:
            # Fallback: infer from directory structure
            repo_name = source_path.parent.name
            repo_path = source_path.parent.parent.parent
            click.echo(f"   Source: context file for {repo_name} (inferred)")
            if not repo_path.exists():
                click.echo(f"   Warning: Could not locate repository at {repo_path}", err=True)
                click.echo("   Consider using --repo flag or ensure source_repo is set in context file", err=True)
        source_context = str(source_path)
    else:
        repo_path = source_path
        repo_name = source_path.name
        source_context = None
        click.echo(f"   Source: repository {repo_name}")

    # Phase 1: Discovery
    click.echo("\n📂 Phase 1: Discovery...")
    keywords = extract_keywords(question)
    click.echo(f"   Keywords: {', '.join(keywords)}")

    # Always scan repository structure for file discovery
    scanner = StructureScanner(config)
    structure = scanner.scan(repo_path)
    file_tree = structure["tree"]

    # Search for relevant files
    candidates = search_relevant_files(repo_path, keywords)
    click.echo(f"   Found {len(candidates)} candidate files")

    if not candidates:
        click.echo("   No matching files found. Using fallback search...")
        # Fallback: use all files from structure scan
        candidates = [
            {"path": f, "match_type": "fallback", "score": 1}
            for f in structure.get("all_files", [])[:20]
        ]

    # Phase 2: LLM-guided exploration
    click.echo(f"\n🤖 Phase 2: Exploration with {config.model_name}...")
    llm = create_llm_provider(config)
    analyzer = ScopedAnalyzer(llm)

    analysis_result = analyzer.analyze(
        repo_path=repo_path,
        question=question,
        candidate_files=candidates,
        file_tree=file_tree,
    )

    click.echo(f"   Analyzed {len(analysis_result['relevant_files'])} files")

    # Phase 3: Generate scoped context
    click.echo("\n📝 Phase 3: Generating scoped context...")
    generator = ScopedGenerator(llm, config.output_dir)

    output_path = Path(output) if output else None
    result_path = generator.generate(
        repo_name=repo_name,
        question=question,
        relevant_files=analysis_result["relevant_files"],
        insights=analysis_result["insights"],
        model_name=config.model_name,
        source_repo=str(repo_path),
        source_context=source_context,
        output_path=output_path,
        code_references=analysis_result.get("code_references"),
    )

    click.echo(f"\n✅ Scoped context generated: {result_path}")
    return 0


def _scope_agent_mode(
    source_path: Path,
    question: str,
    config: Config,
    is_context_file: bool,
    debug: bool,
    stream: bool,
) -> int:
    """Generate scoped context using agent mode with dedicated scoped agent."""
    from .scoper import create_scoped_agent
    from .memory import create_checkpointer, create_agent_config
    from .tools.repository_tools import set_tool_config, set_allowed_repo_root

    # Set tool config for consistency
    set_tool_config(config)
    from .observability import configure_tracing, is_tracing_enabled
    from .observability import configure_langfuse_tracing, is_langfuse_tracing_enabled

    click.echo(f"🤖 Agent mode: Scoping '{question}'")

    if stream:
        click.echo("   Streaming: Enabled")

    # Determine repo path
    if is_context_file:
        source_repo = _extract_repo_from_context(source_path)
        if source_repo and Path(source_repo).exists():
            repo_path = Path(source_repo)
        else:
            # Fallback: infer from directory structure
            repo_path = source_path.parent.parent.parent
        click.echo(f"   Repository: {repo_path}")
    else:
        repo_path = source_path

    # Restrict tool file access to the target repo
    if repo_path.is_dir():
        set_allowed_repo_root(repo_path)

    # Configure tracing (LangSmith + Langfuse)
    configure_tracing()
    configure_langfuse_tracing()

    # Create checkpointer
    checkpointer = create_checkpointer()

    # Resolve API key for the selected model
    from .llm.provider import _resolve_api_key_for_model
    api_key = _resolve_api_key_for_model(config.model_name, config)

    # Create scoped agent with dedicated tools
    agent = create_scoped_agent(
        repo_path=repo_path,
        model_name=config.model_name,
        checkpointer=checkpointer,
        output_dir=config.output_dir,
        debug=debug,
        api_key=api_key,
        base_url=config.api_base_url,
        config=config,
    )

    # Create agent configuration
    agent_config = create_agent_config(f"scope-{repo_path}-{question[:20]}")

    # Build user message
    user_message = f"Generate scoped context answering: {question}"

    click.echo(f"   Thread ID: {agent_config['configurable']['thread_id']}")
    if is_tracing_enabled():
        click.echo("   Tracing: LangSmith enabled")
    if is_langfuse_tracing_enabled():
        click.echo("   Tracing: Langfuse enabled")

    return _run_agent(agent, user_message, agent_config, stream, debug)


if __name__ == "__main__":
    cli()
