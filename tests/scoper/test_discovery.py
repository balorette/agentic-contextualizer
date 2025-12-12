"""Tests for discovery module."""

import pytest
from agents.scoper.discovery import extract_keywords


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
