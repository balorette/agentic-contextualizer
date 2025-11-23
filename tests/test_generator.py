"""Tests for the context generator."""

import yaml
from agents.generator.context_generator import ContextGenerator
from agents.models import ProjectMetadata, CodeAnalysis
from agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM for testing."""

    def __init__(self):
        self.response_content = "# Architecture\n\nWell-structured project."
        self.model_name = "mock-model"

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return configurable markdown content."""
        return LLMResponse(content=self.response_content, model=self.model_name)


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


def test_context_refine_preserves_frontmatter(temp_repo, tmp_path):
    """Refine should keep YAML metadata while updating body."""
    output_dir = tmp_path / "output"
    metadata = ProjectMetadata(
        name="testproject",
        path=temp_repo,
        project_type="python",
        entry_points=["main.py"],
    )

    analysis = CodeAnalysis(architecture_patterns=["Modular"], tech_stack=["Python"], insights="")

    mock_llm = MockLLMProvider()
    generator = ContextGenerator(mock_llm, output_dir)

    output_path = generator.generate(metadata, analysis, "Test project", model_name="mock-model")

    mock_llm.response_content = "## Architecture Overview\nUpdated details."

    generator.refine(output_path, "Add more details")

    content = output_path.read_text()
    frontmatter, body = content.split("---", 2)[1:]
    metadata_yaml = yaml.safe_load(frontmatter)

    assert metadata_yaml["user_summary"] == "Test project"
    assert metadata_yaml["model_used"] == "mock-model"
    assert "Updated details" in body
