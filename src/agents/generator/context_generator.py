"""Context file generation."""

import yaml
from pathlib import Path
from ..models import ProjectMetadata, CodeAnalysis, ContextMetadata
from ..llm.provider import LLMProvider
from ..llm.prompts import CONTEXT_GENERATION_PROMPT, REFINEMENT_PROMPT


class ContextGenerator:
    """Generates context markdown files."""

    def __init__(self, llm_provider: LLMProvider, output_dir: Path):
        """Initialize generator.

        Args:
            llm_provider: LLM provider for text generation
            output_dir: Directory to write context files
        """
        self.llm = llm_provider
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_model_name = getattr(llm_provider, "model_name", "unknown-model")

    def generate(
        self,
        metadata: ProjectMetadata,
        analysis: CodeAnalysis,
        user_summary: str,
        model_name: str | None,
    ) -> Path:
        """Generate context file.

        Args:
            metadata: Project metadata
            analysis: Code analysis results
            user_summary: User's project description
            model_name: Name of LLM model used

        Returns:
            Path to generated context file
        """
        # Build prompt
        prompt = CONTEXT_GENERATION_PROMPT.format(
            project_metadata=self._format_metadata(metadata),
            code_analysis=self._format_analysis(analysis),
            user_summary=user_summary,
        )

        # Generate content
        response = self.llm.generate(
            prompt=prompt,
            system="You are generating context documentation for AI agents. Be concise and structured.",
        )

        # Build complete context
        effective_model = model_name or self.default_model_name

        context_metadata = ContextMetadata(
            source_repo=str(metadata.path), user_summary=user_summary, model_used=effective_model
        )

        # Combine frontmatter and content
        full_content = self._build_context_file(context_metadata, response.content)

        # Write to file
        output_path = self.output_dir / metadata.name / "context.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_content, encoding="utf-8")

        return output_path

    def refine(self, context_file: Path, user_request: str) -> Path:
        """Refine existing context file.

        Args:
            context_file: Path to existing context file
            user_request: What to change

        Returns:
            Path to updated context file
        """
        current_content = context_file.read_text(encoding="utf-8")
        existing_metadata, body = self._extract_frontmatter(current_content)

        prompt = REFINEMENT_PROMPT.format(current_context=body, user_request=user_request)

        response = self.llm.generate(
            prompt=prompt, system="You are updating context documentation based on user feedback."
        )

        source_repo = existing_metadata.get("source_repo") or str(context_file.parent.parent)
        user_summary = existing_metadata.get("user_summary", "")

        updated_metadata = ContextMetadata(
            source_repo=source_repo,
            user_summary=user_summary,
            model_used=self.default_model_name,
        )

        full_content = self._build_context_file(updated_metadata, response.content.strip())
        context_file.write_text(full_content, encoding="utf-8")

        return context_file

    def _format_metadata(self, metadata: ProjectMetadata) -> str:
        """Format metadata for prompt."""
        parts = [
            f"Project: {metadata.name}",
            f"Type: {metadata.project_type or 'Unknown'}",
            f"Entry Points: {', '.join(metadata.entry_points) if metadata.entry_points else 'None found'}",
            f"Key Files: {', '.join(metadata.key_files[:10])}",  # Limit to avoid overflow
        ]

        if metadata.dependencies:
            dep_list = [f"{k}: {v}" for k, v in list(metadata.dependencies.items())[:15]]
            parts.append(f"Dependencies: {', '.join(dep_list)}")

        return "\n".join(parts)

    def _format_analysis(self, analysis: CodeAnalysis) -> str:
        """Format analysis for prompt."""
        parts = [
            f"Architecture: {', '.join(analysis.architecture_patterns)}",
            f"Tech Stack: {', '.join(analysis.tech_stack)}",
            f"Insights: {analysis.insights}",
        ]

        if analysis.coding_conventions:
            conv = [f"{k}: {v}" for k, v in analysis.coding_conventions.items()]
            parts.append(f"Conventions: {', '.join(conv)}")

        return "\n".join(parts)

    def _build_context_file(self, metadata: ContextMetadata, content: str) -> str:
        """Build final context file with frontmatter.

        Args:
            metadata: Context metadata for frontmatter
            content: Generated markdown content

        Returns:
            Complete file content with YAML frontmatter
        """
        frontmatter = yaml.dump(
            {
                "source_repo": metadata.source_repo,
                "scan_date": metadata.scan_date.isoformat(),
                "user_summary": metadata.user_summary,
                "model_used": metadata.model_used,
            },
            default_flow_style=False,
        )

        return f"---\n{frontmatter}---\n\n{content.strip()}\n"

    def _extract_frontmatter(self, content: str) -> tuple[dict, str]:
        """Split YAML frontmatter from markdown body."""
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        _, raw_meta, rest = parts
        try:
            metadata = yaml.safe_load(raw_meta) or {}
        except yaml.YAMLError:
            metadata = {}

        return metadata, rest.strip()
