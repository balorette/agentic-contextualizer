"""Tests for configuration loading."""

from pathlib import Path
from agents.config import Config, DEFAULT_IGNORED_DIRS


def test_config_from_env_overrides(monkeypatch, tmp_path):
    """Environment variables should override defaults."""
    monkeypatch.setenv("MODEL_NAME", "custom-model")
    monkeypatch.setenv("MAX_FILE_SIZE", "2048")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "contexts"))
    monkeypatch.setenv("IGNORED_DIRS", "coverage,.mypy_cache")

    config = Config.from_env()

    assert config.model_name == "custom-model"
    assert config.max_file_size == 2048
    assert Path(config.output_dir) == tmp_path / "contexts"

    # Ensure new ignored directories are appended to defaults
    assert set(DEFAULT_IGNORED_DIRS).issubset(set(config.ignored_dirs))
    assert "coverage" in config.ignored_dirs
    assert ".mypy_cache" in config.ignored_dirs
