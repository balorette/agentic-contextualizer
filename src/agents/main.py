"""Main CLI entry point for Agentic Contextualizer."""

import click


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
        agentic-contextualizer generate /path/to/repo -s "FastAPI REST API"
    """
    click.echo(f"Generating context for: {repo_path}")
    click.echo(f"Summary: {summary}")
    # TODO: Implement pipeline
    raise NotImplementedError("Pipeline implementation coming in next phase")


@cli.command()
@click.argument("context_file", type=click.Path(exists=True))
@click.option("--request", "-r", required=True, help="What to change")
def refine(context_file: str, request: str):
    """Refine an existing context file.

    Example:
        agentic-contextualizer refine contexts/myapp/context.md -r "Add auth details"
    """
    click.echo(f"Refining: {context_file}")
    click.echo(f"Request: {request}")
    # TODO: Implement refinement
    raise NotImplementedError("Refinement implementation coming in next phase")


if __name__ == "__main__":
    cli()
