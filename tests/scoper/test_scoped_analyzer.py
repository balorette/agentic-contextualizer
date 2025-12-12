"""Tests for scoped analyzer."""

import pytest
import json
from pathlib import Path
from agents.scoper.scoped_analyzer import ScopedAnalyzer, ScopeExplorationOutput
from agents.llm.provider import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Return mock response."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return LLMResponse(content=json.dumps(response), model="mock")
        return LLMResponse(content="{}", model="mock")

    def generate_structured(self, prompt: str, system: str | None, schema):
        """Return mock structured response."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return schema(**response)
        return schema(
            additional_files_needed=[],
            reasoning="Done",
            sufficient_context=True,
            preliminary_insights="Test insights",
        )


class TestScopedAnalyzer:
    """Test ScopedAnalyzer."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "weather.py").write_text("def get_weather(): pass")
        (repo / "src" / "utils.py").write_text("def helper(): pass")
        return repo

    def test_analyze_returns_relevant_files(self, sample_repo):
        """Test that analyze returns list of relevant files."""
        mock_llm = MockLLMProvider(responses=[
            {
                "additional_files_needed": [],
                "reasoning": "Found all needed files",
                "sufficient_context": True,
                "preliminary_insights": "Weather module handles forecasts",
            }
        ])

        analyzer = ScopedAnalyzer(mock_llm)
        result = analyzer.analyze(
            repo_path=sample_repo,
            question="weather functionality",
            candidate_files=[{"path": "src/weather.py", "match_type": "filename", "score": 2}],
            file_tree={"name": "repo", "type": "directory", "children": []},
        )

        assert "relevant_files" in result
        assert "insights" in result
        assert len(result["relevant_files"]) > 0

    def test_analyze_expands_with_additional_files(self, sample_repo):
        """Test that analyzer requests and incorporates additional files."""
        mock_llm = MockLLMProvider(responses=[
            {
                "additional_files_needed": ["src/utils.py"],
                "reasoning": "Utils is imported by weather",
                "sufficient_context": False,
                "preliminary_insights": "Need to check utils",
            },
            {
                "additional_files_needed": [],
                "reasoning": "Now have full context",
                "sufficient_context": True,
                "preliminary_insights": "Weather uses utils for helpers",
            },
        ])

        analyzer = ScopedAnalyzer(mock_llm)
        result = analyzer.analyze(
            repo_path=sample_repo,
            question="weather functionality",
            candidate_files=[{"path": "src/weather.py", "match_type": "filename", "score": 2}],
            file_tree={"name": "repo", "type": "directory", "children": []},
        )

        # Should have called LLM twice (initial + expansion)
        assert mock_llm.call_count == 2
        assert "insights" in result
