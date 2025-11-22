"""Integration tests for full pipeline."""

import json
from agents.scanner.structure import StructureScanner
from agents.scanner.metadata import MetadataExtractor
from agents.analyzer.code_analyzer import CodeAnalyzer
from agents.generator.context_generator import ContextGenerator
from agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM for integration testing."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return appropriate mock response based on call."""
        self.call_count += 1

        if self.call_count == 1:  # Code analysis
            response = {
                "architecture_patterns": ["Linear Pipeline"],
                "tech_stack": ["Python", "LangChain"],
                "coding_conventions": {"style": "PEP8"},
                "insights": "Well-structured tool",
            }
            content = json.dumps(response)
        else:  # Context generation
            content = """# Repository Context: Test Repo

## Architecture Overview
Linear pipeline architecture.

## Key Commands
- `pytest`: Run tests

## Code Patterns
PEP8 style guide followed.

## Entry Points
- main.py
"""

        return LLMResponse(content=content, model="mock")


def test_full_pipeline_integration(tmp_path, config):
    """Test complete pipeline from scan to context generation."""
    # Setup test repository
    repo = tmp_path / "test_repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Test Repo")
    (repo / "main.py").write_text("def main():\n    pass")

    pyproject = """
[project]
name = "test-repo"
dependencies = ["click>=8.0.0"]
"""
    (repo / "pyproject.toml").write_text(pyproject)

    # Setup components
    scanner = StructureScanner(config)
    extractor = MetadataExtractor()
    mock_llm = MockLLMProvider()
    analyzer = CodeAnalyzer(mock_llm)

    output_dir = tmp_path / "output"
    generator = ContextGenerator(mock_llm, output_dir)

    # Run pipeline
    structure = scanner.scan(repo)
    assert structure["total_files"] == 3

    metadata = extractor.extract(repo)
    assert metadata.project_type == "python"
    assert "main.py" in metadata.entry_points

    analysis = analyzer.analyze(repo, metadata, structure["tree"], "Test project")
    assert "Linear Pipeline" in analysis.architecture_patterns

    output_path = generator.generate(metadata, analysis, "Test project", "mock")
    assert output_path.exists()

    # Verify output
    content = output_path.read_text()
    assert "source_repo" in content
    assert "Architecture Overview" in content
    assert "Key Commands" in content
