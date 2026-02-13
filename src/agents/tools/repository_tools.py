"""LangChain tool wrappers for repository analysis pipeline."""

from pathlib import Path, PurePosixPath
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


def _flatten_tree(root: dict) -> list[str]:
    """Flatten nested tree dict into a compact list of repo-relative paths.

    The root directory name is excluded from output paths so that results
    are repo-relative (e.g. ``src/main.py`` instead of ``myrepo/src/main.py``).
    """

    def _walk(node: dict, prefix: str) -> list[str]:
        paths: list[str] = []
        name = node.get("name", "")
        # Use PurePosixPath for consistent forward-slash separators
        # regardless of host OS (context files are portable)
        path = str(PurePosixPath(prefix) / name) if prefix else name
        if node.get("type") == "file":
            paths.append(path)
        elif node.get("type") == "directory":
            for child in node.get("children", []):
                paths.extend(_walk(child, path))
        return paths

    # Skip root directory name — start from its children
    paths: list[str] = []
    for child in root.get("children", []):
        paths.extend(_walk(child, ""))
    return paths


@tool
def scan_structure(repo_path: str) -> dict[str, Any]:
    """Scan repository structure. Returns a flat file listing, file counts, and directory counts.

    Args:
        repo_path: Absolute path to the repository root directory

    Returns:
        Dictionary with file_list, total_files, total_dirs, or error.
    """
    try:
        config = _get_config()
        scanner = _get_scanner()
        result = scanner.scan(Path(repo_path))
        # Flatten tree to compact list of relative paths (saves ~80% tokens vs nested JSON)
        flat_files = _flatten_tree(result["tree"])
        limit = config.max_scan_files
        return {
            "file_list": flat_files[:limit],
            "total_files": result["total_files"],
            "total_dirs": result["total_dirs"],
        }
    except Exception as e:
        return {"error": f"Failed to scan repository: {str(e)}"}


@tool
def extract_metadata(repo_path: str) -> dict[str, Any]:
    """Extract project type, dependencies, entry points from config files.

    Args:
        repo_path: Absolute path to the repository root directory

    Returns:
        Dictionary with project_type, dependencies, entry_points, key_files, or error.
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
    repo_path: str, user_summary: str, metadata_dict: dict[str, Any], file_list: list[str]
) -> dict[str, Any]:
    """Analyze code using LLM for architectural insights. Expensive (LLM call).

    Args:
        repo_path: Absolute path to the repository root directory
        user_summary: User's description of what the project does
        metadata_dict: Output from extract_metadata tool
        file_list: List of file paths from scan_structure's file_list field

    Returns:
        Dictionary with architecture_patterns, coding_conventions, tech_stack, insights, or error.
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

        # Build a simple tree dict from flat file list for the analyzer
        tree = {"name": Path(repo_path).name, "type": "directory", "children": []}
        for fp in file_list:
            tree["children"].append({"name": fp, "type": "file"})

        llm = _get_llm_provider()
        analyzer = CodeAnalyzer(llm)
        analysis = analyzer.analyze(Path(repo_path), metadata, tree, user_summary)

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
    """Generate final context markdown file. Expensive (LLM call). Call after analyze_code.

    Args:
        repo_path: Absolute path to the repository root directory
        user_summary: User's description of the project
        metadata_dict: Output from extract_metadata tool
        analysis_dict: Output from analyze_code tool

    Returns:
        Dictionary with context_md preview, output_path, or error.
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
    """Refine an existing context file. Expensive (LLM call).

    Args:
        context_file_path: Absolute path to the existing context.md file
        refinement_request: Description of what to change or add

    Returns:
        Dictionary with updated_context preview, output_path, or error.
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
