"""Scoped context file generation."""

import re
import yaml
from pathlib import Path
from typing import Dict
from ..llm.provider import LLMProvider
from ..llm.prompts import SCOPE_GENERATION_PROMPT
from ..models import ScopedContextMetadata

# Maximum number of characters from each file to include in the generation prompt.
# Files exceeding this limit are truncated to prevent LLM context overflow.
MAX_CONTENT_PER_FILE = 10_000


class ScopedGenerator:
    """Generates scoped context markdown files."""

    def __init__(self, llm_provider: LLMProvider, output_dir: Path):
        """Initialize generator.

        Args:
            llm_provider: LLM provider for text generation
            output_dir: Directory to write context files
        """
        self.llm = llm_provider
        self.output_dir = output_dir

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
        full_content = self._build_context_file(metadata, response.content)

        # Determine output path
        if output_path is None:
            sanitized_name = self._sanitize_filename(question)
            output_path = self.output_dir / repo_name / f"scope-{sanitized_name}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_content, encoding="utf-8")

        return output_path

    def _format_files(self, files: Dict[str, str]) -> str:
        """Format file contents for prompt."""
        parts = []
        for path, content in files.items():
            truncated = content[:MAX_CONTENT_PER_FILE]
            parts.append(f"=== {path} ===\n{truncated}")
        return "\n\n".join(parts)

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

    def _build_context_file(self, metadata: ScopedContextMetadata, content: str) -> str:
        """Build final context file with frontmatter."""
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
        return f"---\n{frontmatter}---\n\n{content.strip()}\n"
