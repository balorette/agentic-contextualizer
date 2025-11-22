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
def generate(repo_path: str, summary: str, output: str | None):
    """Generate context for a repository.

    Example:
        python -m agents.main generate /path/to/repo -s "FastAPI REST API"
    """
    repo = Path(repo_path).resolve()
    config = Config.from_env()

    if not config.api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set in environment", err=True)
        return 1

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


@cli.command()
@click.argument("context_file", type=click.Path(exists=True))
@click.option("--request", "-r", required=True, help="What to change")
def refine(context_file: str, request: str):
    """Refine an existing context file.

    Example:
        python -m agents.main refine contexts/myapp/context.md -r "Add auth details"
    """
    context_path = Path(context_file)
    config = Config.from_env()

    if not config.api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set in environment", err=True)
        return 1

    click.echo(f"🔄 Refining: {context_path}")

    llm = AnthropicProvider(config.model_name, config.api_key)
    generator = ContextGenerator(llm, config.output_dir)

    updated_path = generator.refine(context_path, request)

    click.echo(f"✅ Context updated: {updated_path}")
    return 0


if __name__ == "__main__":
    cli()
