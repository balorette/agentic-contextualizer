"""Integration tests for scoped context generation."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, Mock
from src.agents.main import cli
from src.agents.llm.provider import LLMResponse


class TestScopeIntegration:
    """End-to-end tests for scope command."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a realistic sample repository."""
        repo = tmp_path / "weather-api"
        repo.mkdir()

        # Create structure
        (repo / "src").mkdir()
        (repo / "src" / "weather").mkdir()
        (repo / "src" / "auth").mkdir()
        (repo / "tests").mkdir()

        # Weather module
        (repo / "src" / "weather" / "__init__.py").write_text('"""Weather module for forecast services."""\n')
        (repo / "src" / "weather" / "service.py").write_text("""
class WeatherService:
    def get_forecast(self, location: str) -> dict:
        '''Get weather forecast for a location.'''
        return {"location": location, "temp": 72}

    def get_alerts(self, location: str) -> list:
        '''Get weather alerts for a location.'''
        return []
""")
        (repo / "src" / "weather" / "models.py").write_text("""
from dataclasses import dataclass

@dataclass
class Forecast:
    location: str
    temperature: float
    conditions: str
""")

        # Auth module (unrelated)
        (repo / "src" / "auth" / "__init__.py").write_text('"""Authentication module."""\n')
        (repo / "src" / "auth" / "login.py").write_text("""
def authenticate(username: str, password: str) -> bool:
    return True
""")

        # Tests
        (repo / "tests" / "test_weather.py").write_text("""
from src.weather.service import WeatherService

def test_get_forecast():
    service = WeatherService()
    result = service.get_forecast("NYC")
    assert result["location"] == "NYC"
""")

        # README
        (repo / "README.md").write_text("# Weather API\n\nA weather forecasting service.")

        return repo

    @patch("src.agents.main.AnthropicProvider")
    def test_scope_pipeline_end_to_end(self, mock_provider_class, runner, sample_repo, tmp_path):
        """Test full pipeline scope execution."""
        # Mock LLM responses
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider

        # Mock exploration response
        mock_provider.generate_structured.return_value = Mock(
            additional_files_needed=[],
            reasoning="Found all weather files",
            sufficient_context=True,
            preliminary_insights="Weather service provides forecast and alerts",
        )

        # Mock generation response
        mock_provider.generate.return_value = LLMResponse(
            content="""## Summary

The weather module provides forecast and alert functionality.

## API Surface

- `WeatherService.get_forecast(location)` - Returns forecast data
- `WeatherService.get_alerts(location)` - Returns weather alerts

## Key Files

- src/weather/service.py - Main service implementation
- src/weather/models.py - Data models

## Usage Examples

See tests/test_weather.py for usage examples.
""",
            model="claude-3-5-sonnet",
        )

        # Set up environment
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.agents.main.Config.from_env") as mock_config:
                mock_config.return_value = Mock(
                    api_key="test-key",
                    model_name="claude-3-5-sonnet",
                    output_dir=output_dir,
                    ignored_dirs=[".git", "__pycache__"],
                    max_file_size=1_000_000,
                )

                result = runner.invoke(
                    cli,
                    ["scope", str(sample_repo), "-q", "weather functionality"],
                )

        # Verify execution
        assert result.exit_code == 0
        assert "Scoping:" in result.output
        assert "Phase 1: Discovery" in result.output
        assert "Phase 2: Exploration" in result.output
        assert "Phase 3: Generating" in result.output
        assert "Scoped context generated" in result.output

    def test_scope_discovery_finds_relevant_files(self, sample_repo):
        """Test that discovery phase finds weather-related files."""
        from src.agents.scoper.discovery import extract_keywords, search_relevant_files

        keywords = extract_keywords("weather forecast functionality")
        results = search_relevant_files(sample_repo, keywords)

        paths = [r["path"] for r in results]

        # Should find weather-related files
        assert any("weather" in p for p in paths)
        # Should not prioritize auth files
        auth_paths = [p for p in paths if "auth" in p]
        weather_paths = [p for p in paths if "weather" in p]
        assert len(weather_paths) >= len(auth_paths)
