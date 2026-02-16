"""LLM-guided exploration for scoped context generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from ..llm.provider import LLMProvider
from ..llm.prompts import SCOPE_EXPLORATION_PROMPT
from ..tools import FileBackend, LocalFileBackend, CodeReference

if TYPE_CHECKING:
    from ..file_access import SmartFileAccess

# Maximum number of LLM-guided exploration rounds before forcing synthesis.
# This limits cost and latency by preventing excessive iterations.
MAX_EXPLORATION_ROUNDS = 2

# Maximum number of characters from each file to include in the LLM prompt.
# This helps avoid exceeding the LLM's context window and keeps prompts efficient.
# Value chosen based on typical LLM context limits and empirical prompt size testing.
MAX_FILE_CONTENT_CHARS = 8_000

# Maximum total characters for ALL file contents combined in a single prompt.
# With ~4 chars/token, 80k chars ≈ 20k tokens, leaving room for prompt/tree/overhead.
# This prevents rate limit errors (50k tokens/min) on APIs like Anthropic.
MAX_TOTAL_CONTENT_CHARS = 80_000

# Maximum characters for the file tree representation.
# Large repos can have huge trees; we truncate to leave room for file contents.
MAX_TREE_CHARS = 8_000

# Maximum number of files to include content for in exploration phase.
# Prioritizes higher-scoring files when this limit is exceeded.
MAX_FILES_IN_PROMPT = 12


class KeyLocation(BaseModel):
    """A key code location discovered during exploration."""

    path: str = Field(description="File path")
    line_start: int = Field(description="Starting line number")
    line_end: Optional[int] = Field(default=None, description="Ending line number")
    description: str = Field(description="Brief description of what this code does")


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
    key_locations: List[KeyLocation] = Field(
        default_factory=list,
        description="Important code locations discovered with line numbers",
    )


class ScopedAnalyzer:
    """Analyzes repository with LLM guidance for scoped context."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        file_backend: FileBackend | None = None,
        max_rounds: int = MAX_EXPLORATION_ROUNDS,
        smart_access: SmartFileAccess | None = None,
    ):
        """Initialize analyzer.

        Args:
            llm_provider: LLM provider for API calls
            file_backend: Backend for file access (created lazily if not provided)
            max_rounds: Maximum exploration rounds before forcing synthesis
            smart_access: Optional SmartFileAccess for progressive disclosure
        """
        self.llm = llm_provider
        self._file_backend = file_backend
        self.max_rounds = max_rounds
        self._smart_access = smart_access

    def _get_backend(self, repo_path: Path) -> FileBackend:
        """Get or create file backend for the repository.

        Args:
            repo_path: Path to repository (used if no backend was provided)

        Returns:
            FileBackend instance
        """
        if self._file_backend is not None:
            return self._file_backend
        return LocalFileBackend(repo_path)

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
            Dict with relevant_files, insights, and code_references
        """
        backend = self._get_backend(repo_path)

        # Track all files we've examined
        examined_files: Dict[str, str] = {}
        # When using progressive disclosure, track outlines separately for exploration
        examined_outlines: Dict[str, str] = {}
        all_insights: List[str] = []
        all_key_locations: List[KeyLocation] = []

        # Start with candidate files
        files_to_examine = [f["path"] for f in candidate_files]
        use_outlines = self._smart_access is not None

        for round_num in range(self.max_rounds):
            # Read files we haven't examined yet
            for file_path in files_to_examine:
                if file_path not in examined_files and file_path not in examined_outlines:
                    if use_outlines:
                        # Progressive: use outlines for exploration (much smaller)
                        outline = self._smart_access.get_outline(file_path)
                        if outline:
                            examined_outlines[file_path] = self._format_outline(outline)
                    else:
                        content = backend.read_file(file_path)
                        if content:
                            examined_files[file_path] = content

            # Ask LLM if we need more files — use outlines if available, else full contents
            exploration_contents = examined_outlines if use_outlines else examined_files
            exploration_result = self._explore(
                question=question,
                file_tree=file_tree,
                candidate_files=candidate_files,
                examined_contents=exploration_contents,
            )

            all_insights.append(exploration_result.preliminary_insights)

            # Collect key locations from this round
            all_key_locations.extend(exploration_result.key_locations)

            if exploration_result.sufficient_context:
                break

            # Queue additional files for next round
            files_to_examine = [
                f for f in exploration_result.additional_files_needed
                if f not in examined_files and f not in examined_outlines
            ]

            if not files_to_examine:
                break

        # After exploration, read full contents of all relevant files for synthesis
        if use_outlines:
            all_paths = list(examined_outlines.keys()) + list(examined_files.keys())
            for file_path in all_paths:
                if file_path not in examined_files:
                    content = backend.read_file(file_path)
                    if content:
                        examined_files[file_path] = content

        # Convert key locations to CodeReference objects
        code_references = self._deduplicate_references(all_key_locations)

        return {
            "relevant_files": examined_files,
            "insights": "\n".join(all_insights),
            "code_references": code_references,
        }

    def _format_outline(self, outline) -> str:
        """Format a FileOutline as a compact string for exploration prompts.

        Args:
            outline: FileOutline from SmartFileAccess.get_outline()

        Returns:
            Compact string representation of the file structure
        """
        lines = [f"[{outline.language}] {outline.path} ({outline.line_count} lines)"]
        if outline.imports:
            lines.append(f"  imports: {', '.join(outline.imports[:10])}")
        for sym in outline.symbols:
            sig = f" — {sym.signature}" if sym.signature else ""
            lines.append(f"  {sym.kind} {sym.name}{sig} (L{sym.line}-{sym.line_end})")
            for child in (sym.children or []):
                child_sig = f" — {child.signature}" if child.signature else ""
                lines.append(f"    {child.kind} {child.name}{child_sig} (L{child.line}-{child.line_end})")
        return "\n".join(lines)

    def _deduplicate_references(
        self, locations: List[KeyLocation]
    ) -> List[CodeReference]:
        """Convert KeyLocations to CodeReferences, removing duplicates.

        Deduplicates by (path, line_start) to avoid repeated references
        to the same code location.

        Args:
            locations: List of KeyLocation objects from exploration

        Returns:
            List of unique CodeReference objects
        """
        seen: set[tuple[str, int]] = set()
        references: List[CodeReference] = []

        for loc in locations:
            key = (loc.path, loc.line_start)
            if key not in seen:
                seen.add(key)
                references.append(
                    CodeReference(
                        path=loc.path,
                        line_start=loc.line_start,
                        line_end=loc.line_end,
                        description=loc.description,
                    )
                )

        return references

    def _explore(
        self,
        question: str,
        file_tree: Dict,
        candidate_files: List[Dict],
        examined_contents: Dict[str, str],
    ) -> ScopeExplorationOutput:
        """Ask LLM what additional files to examine."""
        # Format file tree (with truncation)
        tree_str = self._format_tree(file_tree)
        if len(tree_str) > MAX_TREE_CHARS:
            tree_str = tree_str[:MAX_TREE_CHARS] + "\n... (tree truncated)"

        # Format candidate files
        candidates_str = "\n".join(
            f"- {f['path']} (match: {f['match_type']}, score: {f['score']})"
            for f in candidate_files
        )

        # Format examined contents with strict limits to prevent rate limit errors
        contents_str = self._format_contents_with_limits(
            examined_contents, candidate_files
        )

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

    def _format_contents_with_limits(
        self,
        examined_contents: Dict[str, str],
        candidate_files: List[Dict],
    ) -> str:
        """Format file contents with strict size limits.

        Prioritizes files by their discovery score and limits both per-file
        and total content size to avoid exceeding API rate limits.

        Args:
            examined_contents: Dict mapping file paths to their contents
            candidate_files: List of candidate file dicts with scores

        Returns:
            Formatted string with file contents, respecting size limits
        """
        # Build score lookup for prioritization
        score_lookup = {f["path"]: f.get("score", 0) for f in candidate_files}

        # Sort files by score (highest first), with examined-but-not-in-candidates last
        sorted_paths = sorted(
            examined_contents.keys(),
            key=lambda p: score_lookup.get(p, -1),
            reverse=True,
        )

        # Limit number of files
        paths_to_include = sorted_paths[:MAX_FILES_IN_PROMPT]
        omitted_count = len(sorted_paths) - len(paths_to_include)

        # Build content string with total size limit
        contents_parts = []
        total_chars = 0

        for path in paths_to_include:
            content = examined_contents[path]

            # Per-file truncation
            if len(content) > MAX_FILE_CONTENT_CHARS:
                content = content[:MAX_FILE_CONTENT_CHARS] + "\n... (file truncated)"

            # Check if adding this file would exceed total limit
            file_block = f"\n=== {path} ===\n{content}\n"
            if total_chars + len(file_block) > MAX_TOTAL_CONTENT_CHARS:
                # Truncate this file to fit remaining space
                remaining = MAX_TOTAL_CONTENT_CHARS - total_chars - len(f"\n=== {path} ===\n") - 50
                if remaining > 500:  # Only include if we can show something useful
                    content = content[:remaining] + "\n... (truncated to fit)"
                    file_block = f"\n=== {path} ===\n{content}\n"
                    contents_parts.append(file_block)
                # Stop adding more files
                omitted_count += len(paths_to_include) - len(contents_parts)
                break

            contents_parts.append(file_block)
            total_chars += len(file_block)

        result = "".join(contents_parts)

        if omitted_count > 0:
            result += f"\n... ({omitted_count} additional files omitted to fit context limit)\n"

        return result

    def _format_tree(self, tree: Dict[str, Any], indent: int = 0) -> str:
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
