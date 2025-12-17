"""Tests for discovery module."""

import pytest
from src.agents.scoper.discovery import extract_keywords, search_relevant_files


class TestExtractKeywords:
    """Test keyword extraction from questions."""

    def test_extract_simple_keywords(self):
        """Test extracting keywords from a simple question."""
        keywords = extract_keywords("weather functionality")

        assert "weather" in keywords
        assert "functionality" not in keywords  # stopword-like

    def test_extract_technical_terms(self):
        """Test extracting technical terms."""
        keywords = extract_keywords("authentication flow with OAuth2")

        assert "authentication" in keywords
        assert "oauth2" in keywords
        assert "flow" in keywords
        assert "with" not in keywords  # stopword

    def test_extract_handles_case(self):
        """Test that extraction is case-insensitive."""
        keywords = extract_keywords("UserService API endpoints")

        # Should normalize to lowercase
        assert "userservice" in keywords or "user" in keywords
        assert "api" in keywords
        assert "endpoints" in keywords

    def test_extract_filters_short_words(self):
        """Test that very short words are filtered."""
        keywords = extract_keywords("how do I use the API")

        assert "api" in keywords
        assert "how" not in keywords
        assert "do" not in keywords
        assert "i" not in keywords


class TestSearchRelevantFiles:
    """Test file search based on keywords."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository with various files."""
        repo = tmp_path / "sample_repo"
        repo.mkdir()

        # Create directory structure
        (repo / "src").mkdir()
        (repo / "src" / "weather").mkdir()
        (repo / "src" / "auth").mkdir()
        (repo / "tests").mkdir()

        # Create files
        (repo / "src" / "weather" / "service.py").write_text(
            "class WeatherService:\n    def get_forecast(self): pass"
        )
        (repo / "src" / "weather" / "models.py").write_text(
            "class WeatherData:\n    temperature: float"
        )
        (repo / "src" / "auth" / "login.py").write_text(
            "def authenticate(user): pass"
        )
        (repo / "tests" / "test_weather.py").write_text(
            "def test_weather_service(): pass"
        )
        (repo / "README.md").write_text("# Project\n\nWeather API service")

        return repo

    def test_search_finds_files_by_name(self, sample_repo):
        """Test that search finds files with matching names."""
        results = search_relevant_files(sample_repo, ["weather"])

        file_names = [r["path"] for r in results]
        assert any("weather" in f for f in file_names)

    def test_search_finds_files_by_content(self, sample_repo):
        """Test that search finds files with matching content."""
        results = search_relevant_files(sample_repo, ["forecast"])

        file_names = [r["path"] for r in results]
        assert any("service.py" in f for f in file_names)

    def test_search_returns_match_info(self, sample_repo):
        """Test that search results include match information."""
        results = search_relevant_files(sample_repo, ["weather"])

        assert len(results) > 0
        result = results[0]
        assert "path" in result
        assert "match_type" in result  # "filename" or "content"
        assert "score" in result

    def test_search_excludes_ignored_dirs(self, sample_repo):
        """Test that search respects ignored directories."""
        # Create a node_modules directory
        (sample_repo / "node_modules").mkdir()
        (sample_repo / "node_modules" / "weather.js").write_text("weather")

        results = search_relevant_files(sample_repo, ["weather"])

        file_paths = [r["path"] for r in results]
        assert not any("node_modules" in f for f in file_paths)
