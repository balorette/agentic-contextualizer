"""LLM-based code analysis."""

import json
from pathlib import Path
from ..models import ProjectMetadata, CodeAnalysis
from ..llm.provider import LLMProvider
from ..llm.prompts import CODE_ANALYSIS_PROMPT

MAX_FILES_FOR_ANALYSIS = 20
MAX_FILE_CHARS = 20_000


class CodeAnalyzer:
    """Analyzes code using LLM to extract architectural insights."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize analyzer with LLM provider.

        Args:
            llm_provider: LLM provider instance for making API calls
        """
        self.llm = llm_provider

    def analyze(
        self, repo_path: Path, metadata: ProjectMetadata, file_tree: dict, user_summary: str
    ) -> CodeAnalysis:
        """Analyze repository code.

        Args:
            repo_path: Path to repository
            metadata: Extracted project metadata
            file_tree: File tree structure from scanner
            user_summary: User's description of the project

        Returns:
            CodeAnalysis with extracted insights
        """
        # Read content of key files
        key_files_content = self._read_key_files(
            repo_path, metadata.key_files, metadata.entry_points
        )

        # Format file tree for prompt
        file_tree_str = self._format_tree(file_tree)

        # Build prompt
        prompt = CODE_ANALYSIS_PROMPT.format(
            file_tree=file_tree_str, key_files_content=key_files_content, user_summary=user_summary
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            system="You are an expert software architect analyzing codebases. Respond with valid JSON only.",
        )

        # Parse response
        try:
            analysis_data = json.loads(response.content)
            return CodeAnalysis(
                architecture_patterns=analysis_data.get("architecture_patterns", []),
                coding_conventions=analysis_data.get("coding_conventions", {}),
                tech_stack=analysis_data.get("tech_stack", []),
                insights=analysis_data.get("insights", ""),
            )
        except json.JSONDecodeError as e:
            snippet = response.content[:200]
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}. Response snippet: {snippet}"
            ) from e

    def _read_key_files(
        self, repo_path: Path, key_files: list[str], entry_points: list[str]
    ) -> str:
        """Read content of important files.

        Args:
            repo_path: Repository root
            key_files: List of key configuration files
            entry_points: List of entry point files

        Returns:
            Formatted string with file contents
        """
        files_to_read = sorted(set(key_files + entry_points))[:MAX_FILES_FOR_ANALYSIS]
        content_parts = []

        for file_path_str in files_to_read:
            file_path = repo_path / file_path_str
            if file_path.exists() and file_path.is_file():
                try:
                    # Limit file size to avoid token overflow
                    if file_path.stat().st_size < 50_000:  # 50KB limit
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        snippet = content[:MAX_FILE_CHARS]
                        content_parts.append(f"=== {file_path_str} ===\n{snippet}\n")
                except OSError:
                    pass

        return "\n".join(content_parts) if content_parts else "No key files found."

    def _format_tree(self, tree: dict, indent: int = 0) -> str:
        """Format file tree as indented string.

        Args:
            tree: File tree dictionary
            indent: Current indentation level

        Returns:
            Formatted tree string
        """
        lines = []
        prefix = "  " * indent

        if tree.get("type") == "directory":
            lines.append(f"{prefix}{tree['name']}/")
            for child in tree.get("children", []):
                lines.append(self._format_tree(child, indent + 1))
        else:
            lines.append(f"{prefix}{tree['name']}")

        return "\n".join(lines)
