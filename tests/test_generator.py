"""Tests for the context generator."""

from agents.generator.context_generator import ContextGenerator
from agents.models import ProjectMetadata, CodeAnalysis
from agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM for testing."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return mock markdown content."""
        return LLMResponse(content="# Architecture\n\nWell-structured project.", model="mock")


def test_context_generator_creates_file(temp_repo, tmp_path, config):
    """Test context file generation."""
    output_dir = tmp_path / "output"

    metadata = ProjectMetadata(
        name="testproject", path=temp_repo, project_type="python", entry_points=["main.py"]
    )

    analysis = CodeAnalysis(
        architecture_patterns=["Modular"], tech_stack=["Python"], insights="Clean code"
    )

    mock_llm = MockLLMProvider()
    generator = ContextGenerator(mock_llm, output_dir)

    output_path = generator.generate(
        metadata=metadata, analysis=analysis, user_summary="Test project", model_name="mock-model"
    )

    assert output_path.exists()
    assert output_path.name == "context.md"

    content = output_path.read_text()
    assert "---" in content  # YAML frontmatter
    assert "source_repo" in content
    assert "Architecture" in content
