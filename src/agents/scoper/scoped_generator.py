"""Scoped context file generation."""

from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING
from ..llm.provider import LLMProvider
from ..llm.prompts import SCOPE_GENERATION_PROMPT
from ..models import ScopedContextMetadata
from ..tools import CodeReference

if TYPE_CHECKING:
    from ..file_access import SmartFileAccess

# Maximum number of characters from each file to include in the generation prompt.
# Files exceeding this limit are truncated to prevent LLM context overflow.
MAX_CONTENT_PER_FILE = 15_000

# Maximum total characters for all file contents combined.
# With ~4 chars/token, 120k chars ≈ 30k tokens, leaving room for prompt overhead.
# This prevents rate limit errors on APIs with per-request token limits.
MAX_TOTAL_CONTENT_CHARS = 120_000


class ScopedGenerator:
    """Generates scoped context markdown files."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        output_dir: Path,
        smart_access: SmartFileAccess | None = None,
    ):
        """Initialize generator.

        Args:
            llm_provider: LLM provider for text generation
            output_dir: Directory to write context files
            smart_access: Optional SmartFileAccess (reserved for future use)
        """
        self.llm = llm_provider
        self.output_dir = output_dir
        self._smart_access = smart_access

    def generate(
        self,
        repo_name: str,
        question: str,
        relevant_files: Dict[str, str],
        insights: str,
        model_name: str,
        source_repo: str | None = None,
        source_context: str | None = None,
        output_path: Path | None = None,
        code_references: List[CodeReference] | None = None,
    ) -> Path:
        """Generate scoped context file.

        Args:
            repo_name: Name of the repository
            question: The scope question
            relevant_files: Dict of file_path -> content
            insights: Insights from analysis
            model_name: Name of LLM model used
            source_repo: Path to source repository
            source_context: Path to source context file (if scoping from context)
            output_path: Optional custom output path
            code_references: Optional list of CodeReference objects for the output

        Returns:
            Path to generated scoped context file
        """
        # Format file contents for prompt
        files_str = self._format_files(relevant_files)

        # Generate content via LLM
        prompt = SCOPE_GENERATION_PROMPT.format(
            scope_question=question,
            relevant_files=files_str,
            insights=insights,
        )

        response = self.llm.generate(
            prompt=prompt,
            system="You are generating focused context documentation for a specific question about a codebase.",
        )

        # Build metadata
        metadata = ScopedContextMetadata(
            source_repo=source_repo or f"/{repo_name}",
            source_context=source_context,
            scope_question=question,
            model_used=model_name,
            files_analyzed=len(relevant_files),
        )

        # Build full content
        full_content = self._build_context_file(
            metadata, response.content, code_references
        )

        # Determine output path
        if output_path is None:
            sanitized_name = self._sanitize_filename(question)
            output_path = self.output_dir / repo_name / f"scope-{sanitized_name}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_content, encoding="utf-8")

        return output_path

    def _format_files(self, files: Dict[str, str]) -> str:
        """Format file contents for prompt with size limits.

        Enforces both per-file and total content limits to prevent
        exceeding API rate limits.
        """
        parts = []
        total_chars = 0
        omitted_count = 0

        for path, content in files.items():
            # Per-file truncation
            truncated = content[:MAX_CONTENT_PER_FILE]
            if len(content) > MAX_CONTENT_PER_FILE:
                truncated += "\n... (file truncated)"

            file_block = f"=== {path} ===\n{truncated}"

            # Check total limit
            if total_chars + len(file_block) > MAX_TOTAL_CONTENT_CHARS:
                remaining = MAX_TOTAL_CONTENT_CHARS - total_chars - len(f"=== {path} ===\n") - 50
                if remaining > 500:
                    truncated = content[:remaining] + "\n... (truncated to fit)"
                    file_block = f"=== {path} ===\n{truncated}"
                    parts.append(file_block)
                omitted_count = len(files) - len(parts)
                break

            parts.append(file_block)
            total_chars += len(file_block)

        result = "\n\n".join(parts)
        if omitted_count > 0:
            result += f"\n\n... ({omitted_count} additional files omitted to fit context limit)"

        return result

    def _sanitize_filename(self, question: str) -> str:
        """Convert question to safe, unique filename.

        Uses first 4 words of question plus timestamp suffix to ensure uniqueness
        even when multiple questions share the same prefix.
        """
        from datetime import datetime
        # Take first few words, remove special chars
        sanitized = re.sub(r"[^a-zA-Z0-9\s-]", "", question.lower())
        words = sanitized.split()[:4]
        base_name = "-".join(words) if words else "context"
        # Add timestamp suffix for uniqueness (HHMMSS format)
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{base_name}-{timestamp}"

    def _format_code_references(
        self, references: List[CodeReference] | None
    ) -> str:
        """Format code references as markdown section.

        Args:
            references: List of CodeReference objects

        Returns:
            Formatted markdown section, or empty string if no references
        """
        if not references:
            return ""

        lines = ["", "## Code References", ""]
        for ref in references:
            if ref.line_end and ref.line_end != ref.line_start:
                line_range = f"{ref.line_start}-{ref.line_end}"
            else:
                line_range = str(ref.line_start)
            lines.append(f"- `{ref.path}:{line_range}` - {ref.description}")

        return "\n".join(lines)

    def _build_context_file(
        self,
        metadata: ScopedContextMetadata,
        content: str,
        code_references: List[CodeReference] | None = None,
    ) -> str:
        """Build final context file with frontmatter and code references."""
        frontmatter_dict = {
            "source_repo": metadata.source_repo,
            "scope_question": metadata.scope_question,
            "scan_date": metadata.scan_date.isoformat(),
            "model_used": metadata.model_used,
            "files_analyzed": metadata.files_analyzed,
        }

        if metadata.source_context:
            frontmatter_dict["source_context"] = metadata.source_context

        frontmatter = yaml.dump(frontmatter_dict, default_flow_style=False)

        # Build final content with optional code references section
        final_content = content.strip()
        references_section = self._format_code_references(code_references)
        if references_section:
            if not final_content.endswith("\n"):
                final_content += "\n"
            final_content += references_section

        return f"---\n{frontmatter}---\n\n{final_content}\n"
