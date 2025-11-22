"""Tests for the structure scanner."""

import pytest
from pathlib import Path
from agents.scanner.structure import StructureScanner
from agents.config import Config


def test_structure_scanner_basic(temp_repo, config):
    """Test basic structure scanning."""
    scanner = StructureScanner(config)
    result = scanner.scan(temp_repo)

    assert 'tree' in result
    assert 'all_files' in result
    assert result['total_files'] == 2  # README.md and main.py
    assert 'README.md' in result['all_files']
    assert 'main.py' in result['all_files']


def test_structure_scanner_ignores_dirs(tmp_path, config):
    """Test that scanner ignores configured directories."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "node_modules").mkdir()
    (repo / "src" / "file.py").write_text("code")
    (repo / "node_modules" / "lib.js").write_text("code")

    scanner = StructureScanner(config)
    result = scanner.scan(repo)

    assert 'src/file.py' in result['all_files']
    assert 'node_modules/lib.js' not in result['all_files']


def test_structure_scanner_file_size_limit(tmp_path, config):
    """Test that scanner respects file size limits."""
    repo = tmp_path / "repo"
    repo.mkdir()

    small_file = repo / "small.txt"
    small_file.write_text("small")

    large_file = repo / "large.txt"
    large_file.write_text("x" * (config.max_file_size + 1))

    scanner = StructureScanner(config)
    result = scanner.scan(repo)

    assert 'small.txt' in result['all_files']
    assert 'large.txt' not in result['all_files']
