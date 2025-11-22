"""Tests for the structure scanner."""

import json
import pytest
from pathlib import Path
from agents.scanner.structure import StructureScanner
from agents.scanner.metadata import MetadataExtractor
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


def test_metadata_extractor_python_project(tmp_path, config):
    """Test metadata extraction for Python project."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Create pyproject.toml
    pyproject_content = """
[project]
name = "test-project"
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0.0"
]
"""
    (repo / "pyproject.toml").write_text(pyproject_content)
    (repo / "README.md").write_text("# Test Project")
    (repo / "main.py").write_text("print('hello')")

    extractor = MetadataExtractor()
    metadata = extractor.extract(repo)

    assert metadata.name == "repo"
    assert metadata.project_type == "python"
    assert "requests" in metadata.dependencies
    assert "pydantic" in metadata.dependencies
    assert "main.py" in metadata.entry_points
    assert metadata.readme_content == "# Test Project"


def test_metadata_extractor_node_project(tmp_path, config):
    """Test metadata extraction for Node project."""
    repo = tmp_path / "repo"
    repo.mkdir()

    package_json = {
        "name": "test-app",
        "main": "index.js",
        "dependencies": {
            "express": "^4.18.0"
        }
    }
    (repo / "package.json").write_text(json.dumps(package_json))
    (repo / "index.js").write_text("console.log('hello')")

    extractor = MetadataExtractor()
    metadata = extractor.extract(repo)

    assert metadata.project_type == "node"
    assert "express" in metadata.dependencies
    assert "index.js" in metadata.entry_points
