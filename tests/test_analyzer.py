"""Tests for the code analyzer."""

import json
from agents.analyzer.code_analyzer import CodeAnalyzer
from agents.llm.provider import LLMProvider, LLMResponse
from agents.models import ProjectMetadata, CodeAnalysis


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return mock response."""
        mock_response = {
            "architecture_patterns": ["Modular"],
            "tech_stack": ["Python"],
            "coding_conventions": {"style": "PEP8"},
            "insights": "Well-structured Python project",
        }
        return LLMResponse(content=json.dumps(mock_response), model="mock-model")

    def generate_structured(self, prompt: str, system: str | None = None, schema=None):
        """Return mock structured response."""
        from agents.analyzer.code_analyzer import CodeAnalysisOutput
        return CodeAnalysisOutput(
            architecture_patterns=["Modular"],
            tech_stack=["Python"],
            coding_conventions={"style": "PEP8"},
            insights="Well-structured Python project",
        )


def test_code_analyzer_basic(temp_repo, config):
    """Test code analyzer with mock LLM."""
    metadata = ProjectMetadata(
        name="test",
        path=temp_repo,
        project_type="python",
        key_files=["README.md"],
        entry_points=["main.py"],
    )

    file_tree = {
        "name": "test_repo",
        "type": "directory",
        "children": [{"name": "main.py", "type": "file"}],
    }

    mock_llm = MockLLMProvider()
    analyzer = CodeAnalyzer(mock_llm)

    analysis = analyzer.analyze(
        repo_path=temp_repo, metadata=metadata, file_tree=file_tree, user_summary="Test project"
    )

    assert isinstance(analysis, CodeAnalysis)
    assert "Modular" in analysis.architecture_patterns
    assert "Python" in analysis.tech_stack
