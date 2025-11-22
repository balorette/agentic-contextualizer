"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path
from agents.config import Config


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository structure for testing."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Create sample files
    (repo / "README.md").write_text("# Test Project")
    (repo / "main.py").write_text("print('hello')")

    return repo


@pytest.fixture
def config() -> Config:
    """Provide a test configuration."""
    return Config(
        llm_provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        api_key="test-key",
    )
