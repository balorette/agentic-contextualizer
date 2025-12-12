"""Tests for data models."""

import pytest
from datetime import datetime, UTC
from pathlib import Path
from agents.models import ScopedContextMetadata


class TestScopedContextMetadata:
    """Test ScopedContextMetadata model."""

    def test_create_scoped_metadata_minimal(self):
        """Test creating scoped metadata with required fields."""
        metadata = ScopedContextMetadata(
            source_repo="/path/to/repo",
            scope_question="authentication flow",
            model_used="claude-3-5-sonnet",
            files_analyzed=5,
        )

        assert metadata.source_repo == "/path/to/repo"
        assert metadata.scope_question == "authentication flow"
        assert metadata.model_used == "claude-3-5-sonnet"
        assert metadata.files_analyzed == 5
        assert metadata.source_context is None
        assert metadata.scan_date is not None

    def test_create_scoped_metadata_from_context(self):
        """Test creating scoped metadata when scoping from existing context."""
        metadata = ScopedContextMetadata(
            source_repo="/path/to/repo",
            source_context="contexts/repo/context.md",
            scope_question="weather functionality",
            model_used="claude-3-5-sonnet",
            files_analyzed=12,
        )

        assert metadata.source_context == "contexts/repo/context.md"
        assert metadata.scope_question == "weather functionality"
