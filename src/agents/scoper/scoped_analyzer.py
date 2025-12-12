"""LLM-guided exploration for scoped context generation."""

from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from ..llm.provider import LLMProvider
from ..llm.prompts import SCOPE_EXPLORATION_PROMPT

MAX_EXPLORATION_ROUNDS = 3
MAX_FILE_CONTENT_CHARS = 15_000


class ScopeExplorationOutput(BaseModel):
    """Schema for scope exploration LLM output."""

    additional_files_needed: List[str] = Field(
        description="List of additional file paths to examine"
    )
    reasoning: str = Field(description="Why these files are needed")
    sufficient_context: bool = Field(
        description="Whether we have enough context to generate scoped output"
    )
    preliminary_insights: str = Field(
        description="What has been learned so far about the scope question"
    )


class ScopedAnalyzer:
    """Analyzes repository with LLM guidance for scoped context."""

    def __init__(self, llm_provider: LLMProvider, max_rounds: int = MAX_EXPLORATION_ROUNDS):
        """Initialize analyzer.

        Args:
            llm_provider: LLM provider for API calls
            max_rounds: Maximum exploration rounds before forcing synthesis
        """
        self.llm = llm_provider
        self.max_rounds = max_rounds

    def analyze(
        self,
        repo_path: Path,
        question: str,
        candidate_files: List[Dict],
        file_tree: Dict,
    ) -> Dict[str, Any]:
        """Analyze repository files guided by the scope question.

        Args:
            repo_path: Path to repository
            question: The scope question
            candidate_files: Initial candidate files from discovery
            file_tree: Repository file tree structure

        Returns:
            Dict with relevant_files and insights
        """
        # Track all files we've examined
        examined_files: Dict[str, str] = {}
        all_insights: List[str] = []

        # Start with candidate files
        files_to_examine = [f["path"] for f in candidate_files]

        for round_num in range(self.max_rounds):
            # Read files we haven't examined yet
            for file_path in files_to_examine:
                if file_path not in examined_files:
                    content = self._read_file(repo_path, file_path)
                    if content:
                        examined_files[file_path] = content

            # Ask LLM if we need more files
            exploration_result = self._explore(
                question=question,
                file_tree=file_tree,
                candidate_files=candidate_files,
                examined_contents=examined_files,
            )

            all_insights.append(exploration_result.preliminary_insights)

            if exploration_result.sufficient_context:
                break

            # Queue additional files for next round
            files_to_examine = [
                f for f in exploration_result.additional_files_needed
                if f not in examined_files
            ]

            if not files_to_examine:
                break

        return {
            "relevant_files": examined_files,
            "insights": "\n".join(all_insights),
        }

    def _explore(
        self,
        question: str,
        file_tree: Dict,
        candidate_files: List[Dict],
        examined_contents: Dict[str, str],
    ) -> ScopeExplorationOutput:
        """Ask LLM what additional files to examine."""
        # Format file tree
        tree_str = self._format_tree(file_tree)

        # Format candidate files
        candidates_str = "\n".join(
            f"- {f['path']} (match: {f['match_type']}, score: {f['score']})"
            for f in candidate_files
        )

        # Format examined contents (truncated)
        contents_str = ""
        for path, content in examined_contents.items():
            truncated = content[:MAX_FILE_CONTENT_CHARS]
            contents_str += f"\n=== {path} ===\n{truncated}\n"

        prompt = SCOPE_EXPLORATION_PROMPT.format(
            scope_question=question,
            file_tree=tree_str,
            candidate_files=candidates_str,
            candidate_contents=contents_str,
        )

        return self.llm.generate_structured(
            prompt=prompt,
            system="You are analyzing code to find files relevant to a specific question.",
            schema=ScopeExplorationOutput,
        )

    def _read_file(self, repo_path: Path, file_path: str) -> str | None:
        """Read file content safely."""
        full_path = repo_path / file_path
        try:
            if full_path.exists() and full_path.is_file():
                if full_path.stat().st_size < 500_000:  # 500KB limit
                    return full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
        return None

    def _format_tree(self, tree: Dict, indent: int = 0) -> str:
        """Format file tree as indented string."""
        lines = []
        prefix = "  " * indent

        if tree.get("type") == "directory":
            lines.append(f"{prefix}{tree['name']}/")
            for child in tree.get("children", []):
                lines.append(self._format_tree(child, indent + 1))
        else:
            lines.append(f"{prefix}{tree['name']}")

        return "\n".join(lines)
