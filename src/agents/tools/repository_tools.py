"""LangChain tool wrappers for repository analysis pipeline."""

from pathlib import Path
from typing import Any
from contextvars import ContextVar
from langchain_core.tools import tool

from ..config import Config
from ..scanner.structure import StructureScanner
from ..scanner.metadata import MetadataExtractor
from ..analyzer.code_analyzer import CodeAnalyzer
from ..generator.context_generator import ContextGenerator
from ..llm.provider import LLMProvider, create_llm_provider
from ..models import ProjectMetadata, CodeAnalysis

# Context variable for runtime config (set by agent invocation)
# This allows CLI flags to propagate to tools without breaking tool signatures
_tool_config: ContextVar[Config] = ContextVar('tool_config', default=None)

# Fallback config (used if context var not set)
_default_config = Config.from_env()


def set_tool_config(config: Config) -> None:
    """Set the config to be used by tools in this context.

    This should be called before invoking the agent to ensure tools
    use the correct configuration (including CLI overrides).

    Args:
        config: Configuration instance with all settings
    """
    _tool_config.set(config)


def _get_config() -> Config:
    """Get current config from context or fallback to default.

    Returns:
        Configuration instance
    """
    config = _tool_config.get(None)
    if config is None:
        # Fallback to default config if not set
        # This maintains backward compatibility
        return _default_config
    return config


def _get_llm_provider() -> LLMProvider:
    """Get LLM provider instance from current config.

    Returns:
        Configured LLM provider
    """
    config = _get_config()
    return create_llm_provider(config)


def _get_scanner() -> StructureScanner:
    """Get StructureScanner instance from current config.

    Returns:
        Configured scanner
    """
    config = _get_config()
    return StructureScanner(config)


def _get_context_generator() -> ContextGenerator:
    """Get ContextGenerator instance from current config.

    Returns:
        Configured context generator
    """
    config = _get_config()
    llm = create_llm_provider(config)
    return ContextGenerator(llm, config.output_dir)


@tool
def scan_structure(repo_path: str) -> dict[str, Any]:
    """Scan repository structure and identify key files.

    Use this tool to get a complete overview of the repository's file structure,
    including directory tree, file counts, and all file paths. This is typically
    the first step in analyzing a repository.

    Args:
        repo_path: Absolute path to the repository root directory

    Returns:
        Dictionary containing:
        - tree: Nested directory structure
        - all_files: List of all file paths (limited to first 100 for efficiency)
        - total_files: Total count of files
        - total_dirs: Total count of directories
        - error: Error message if scan failed
    """
    try:
        scanner = _get_scanner()
        result = scanner.scan(Path(repo_path))
        return {
            "tree": result["tree"],
            "all_files": result["all_files"][:100],  # Limit to avoid token overflow
            "total_files": result["total_files"],
            "total_dirs": result["total_dirs"],
        }
    except Exception as e:
        return {"error": f"Failed to scan repository: {str(e)}"}


@tool
def extract_metadata(repo_path: str) -> dict[str, Any]:
    """Extract project metadata from configuration files.

    Analyzes package.json, pyproject.toml, Cargo.toml, and other config files
    to determine project type, dependencies, entry points, and key files.
    Call this after scan_structure to get project-specific information.

    Args:
        repo_path: Absolute path to the repository root directory

    Returns:
        Dictionary containing:
        - project_type: Detected type (python, node, rust, go, java, or None)
        - dependencies: Dictionary of dependency name -> version (limited to 20)
        - entry_points: List of identified entry point files
        - key_files: List of important configuration files (limited to 20)
        - error: Error message if extraction failed
    """
    try:
        # MetadataExtractor is stateless, can be instantiated per call
        extractor = MetadataExtractor()
        metadata = extractor.extract(Path(repo_path))
        return {
            "project_type": metadata.project_type,
            "dependencies": dict(list(metadata.dependencies.items())[:20]),  # Limit for tokens
            "entry_points": metadata.entry_points,
            "key_files": metadata.key_files[:20],  # Limit for tokens
        }
    except Exception as e:
        return {"error": f"Failed to extract metadata: {str(e)}"}


@tool
def analyze_code(
    repo_path: str, user_summary: str, metadata_dict: dict[str, Any], file_tree: dict[str, Any]
) -> dict[str, Any]:
    """Analyze code using LLM to extract architectural insights.

    This tool performs deep code analysis using an LLM to identify architecture
    patterns, coding conventions, tech stack, and generate insights. This is
    an expensive operation (LLM call). Call after extract_metadata.

    Args:
        repo_path: Absolute path to the repository root directory
        user_summary: User's description of what the project does
        metadata_dict: Output from extract_metadata tool
        file_tree: Output from scan_structure tool (the 'tree' field)

    Returns:
        Dictionary containing:
        - architecture_patterns: List of identified patterns (e.g., "MVC", "Microservices")
        - coding_conventions: Dictionary of convention type -> description
        - tech_stack: List of technologies and frameworks used
        - insights: Additional insights about the codebase
        - error: Error message if analysis failed
    """
    try:
        # Reconstruct ProjectMetadata from dict
        metadata = ProjectMetadata(
            name=Path(repo_path).name,
            path=Path(repo_path),
            project_type=metadata_dict.get("project_type"),
            dependencies=metadata_dict.get("dependencies", {}),
            entry_points=metadata_dict.get("entry_points", []),
            key_files=metadata_dict.get("key_files", []),
        )

        llm = _get_llm_provider()
        analyzer = CodeAnalyzer(llm)
        analysis = analyzer.analyze(Path(repo_path), metadata, file_tree, user_summary)

        return {
            "architecture_patterns": analysis.architecture_patterns,
            "coding_conventions": analysis.coding_conventions,
            "tech_stack": analysis.tech_stack,
            "insights": analysis.insights,
        }
    except Exception as e:
        return {"error": f"Failed to analyze code: {str(e)}"}


@tool
def generate_context(
    repo_path: str,
    user_summary: str,
    metadata_dict: dict[str, Any],
    analysis_dict: dict[str, Any],
) -> dict[str, Any]:
    """Generate final context markdown file.

    Synthesizes all gathered information (metadata + analysis) into a
    comprehensive markdown context file with YAML frontmatter. This performs
    an LLM call to generate the final documentation. Call after analyze_code.

    Args:
        repo_path: Absolute path to the repository root directory
        user_summary: User's description of the project
        metadata_dict: Output from extract_metadata tool
        analysis_dict: Output from analyze_code tool

    Returns:
        Dictionary containing:
        - context_md: Generated markdown content (truncated preview)
        - output_path: Path where context file was saved
        - error: Error message if generation failed
    """
    try:
        config = _get_config()

        # Reconstruct models from dicts
        metadata = ProjectMetadata(
            name=Path(repo_path).name,
            path=Path(repo_path),
            project_type=metadata_dict.get("project_type"),
            dependencies=metadata_dict.get("dependencies", {}),
            entry_points=metadata_dict.get("entry_points", []),
            key_files=metadata_dict.get("key_files", []),
        )

        analysis = CodeAnalysis(
            architecture_patterns=analysis_dict.get("architecture_patterns", []),
            coding_conventions=analysis_dict.get("coding_conventions", {}),
            tech_stack=analysis_dict.get("tech_stack", []),
            insights=analysis_dict.get("insights", ""),
        )

        generator = _get_context_generator()
        output_path = generator.generate(metadata, analysis, user_summary, config.model_name)

        # Read generated content for confirmation
        content = output_path.read_text(encoding="utf-8")

        return {
            "context_md": content[:500] + "..." if len(content) > 500 else content,
            "output_path": str(output_path),
        }
    except Exception as e:
        return {"error": f"Failed to generate context: {str(e)}"}


@tool
def refine_context(context_file_path: str, refinement_request: str) -> dict[str, Any]:
    """Refine an existing context file based on user feedback.

    Updates an existing context markdown file with requested changes.
    Performs an LLM call to intelligently update the content while
    preserving structure and other information.

    Args:
        context_file_path: Absolute path to the existing context.md file
        refinement_request: Description of what to change or add

    Returns:
        Dictionary containing:
        - updated_context: Preview of updated markdown content
        - output_path: Path where updated context was saved
        - error: Error message if refinement failed
    """
    try:
        generator = _get_context_generator()
        updated_path = generator.refine(Path(context_file_path), refinement_request)

        # Read updated content
        content = updated_path.read_text(encoding="utf-8")

        return {
            "updated_context": content[:500] + "..." if len(content) > 500 else content,
            "output_path": str(updated_path),
        }
    except Exception as e:
        return {"error": f"Failed to refine context: {str(e)}"}
