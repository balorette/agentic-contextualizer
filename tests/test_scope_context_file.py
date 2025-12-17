"""Tests for scoping from context files."""

from src.agents.main import _extract_repo_from_context


class TestExtractRepoFromContext:
    """Test extracting repo path from context file frontmatter."""

    def test_extract_repo_from_valid_context(self, tmp_path):
        """Test extracting source_repo from context file."""
        context_file = tmp_path / "context.md"
        context_file.write_text("""---
source_repo: /path/to/original/repo
scan_date: 2025-01-22T10:30:00Z
user_summary: "Test project"
model_used: claude-3-5-sonnet
---

# Repository Context
""")

        result = _extract_repo_from_context(context_file)

        assert result == "/path/to/original/repo"

    def test_extract_repo_returns_none_for_missing_field(self, tmp_path):
        """Test that missing source_repo returns None."""
        context_file = tmp_path / "context.md"
        context_file.write_text("""---
scan_date: 2025-01-22T10:30:00Z
---

# No source_repo field
""")

        result = _extract_repo_from_context(context_file)

        assert result is None

    def test_extract_repo_returns_none_for_invalid_yaml(self, tmp_path):
        """Test that invalid YAML returns None gracefully."""
        context_file = tmp_path / "context.md"
        context_file.write_text("# No frontmatter at all")

        result = _extract_repo_from_context(context_file)

        assert result is None
