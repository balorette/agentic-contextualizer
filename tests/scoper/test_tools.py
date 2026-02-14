"""Tests for scoped context tools."""

import pytest
from src.agents.tools import (
    InMemoryFileBackend,
    LocalFileBackend,
    read_file,
    search_files,
    extract_imports,
    create_file_tools,
    create_analysis_tools,
)


class TestReadScopedFile:
    """Tests for read_file function."""

    def test_read_existing_file(self):
        """Test reading an existing file."""
        backend = InMemoryFileBackend(files={
            "src/main.py": "def main(): pass",
        })

        result = read_file(backend, "src/main.py")

        assert result.content == "def main(): pass"
        assert result.path == "src/main.py"
        assert result.char_count == 16
        assert result.truncated is False
        assert result.error is None

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        backend = InMemoryFileBackend()

        result = read_file(backend, "missing.py")

        assert result.content is None
        assert result.error is not None
        assert "missing.py" in result.error

    def test_truncation(self):
        """Test that content is truncated when exceeding max_chars."""
        backend = InMemoryFileBackend(files={
            "large.py": "x" * 1000,
        })

        result = read_file(backend, "large.py", max_chars=100)

        assert result.content == "x" * 100
        assert result.char_count == 100
        assert result.truncated is True


class TestSearchFiles:
    """Tests for search_files function."""

    def test_search_finds_by_filename(self, tmp_path):
        """Test searching finds files by filename match."""
        # Need LocalFileBackend for search (uses discovery.py)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "auth.py").write_text("# auth module")
        (repo / "main.py").write_text("# main module")

        backend = LocalFileBackend(repo)
        result = search_files(backend, ["auth"])

        assert result.error is None
        assert result.total_found >= 1
        assert any(m.path == "auth.py" for m in result.matches)

    def test_search_finds_by_content(self, tmp_path):
        """Test searching finds files by content match."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "utils.py").write_text("def authenticate(): pass")

        backend = LocalFileBackend(repo)
        result = search_files(backend, ["authenticate"])

        assert result.error is None
        assert any("utils.py" in m.path for m in result.matches)

    def test_search_returns_keywords(self, tmp_path):
        """Test that search returns the keywords used."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("")

        backend = LocalFileBackend(repo)
        result = search_files(backend, ["foo", "bar"])

        assert result.keywords_used == ["foo", "bar"]


class TestExtractImports:
    """Tests for extract_imports function."""

    def test_python_simple_import(self):
        """Test extracting simple Python imports."""
        backend = InMemoryFileBackend(files={
            "main.py": "import os\nimport sys",
        })

        result = extract_imports(backend, "main.py")

        assert result.language == "python"
        assert result.error is None
        assert len(result.imports) == 2
        assert result.imports[0].module == "os"
        assert result.imports[1].module == "sys"

    def test_python_from_import(self):
        """Test extracting Python from imports."""
        backend = InMemoryFileBackend(files={
            "main.py": "from pathlib import Path, PurePath",
        })

        result = extract_imports(backend, "main.py")

        assert len(result.imports) == 1
        assert result.imports[0].module == "pathlib"
        assert "Path" in result.imports[0].names
        assert "PurePath" in result.imports[0].names

    def test_python_relative_import(self):
        """Test extracting Python relative imports."""
        backend = InMemoryFileBackend(files={
            "src/utils/helpers.py": "from ..models import User\nfrom . import config",
        })

        result = extract_imports(backend, "src/utils/helpers.py")

        assert len(result.imports) == 2

        # from ..models import User
        assert result.imports[0].is_relative is True
        assert result.imports[0].module == "..models"

        # from . import config
        assert result.imports[1].is_relative is True

    def test_python_import_with_alias(self):
        """Test extracting Python imports with aliases."""
        backend = InMemoryFileBackend(files={
            "main.py": "import numpy as np",
        })

        result = extract_imports(backend, "main.py")

        assert result.imports[0].module == "numpy"
        assert result.imports[0].alias == "np"

    def test_javascript_es6_import(self):
        """Test extracting ES6 JavaScript imports."""
        backend = InMemoryFileBackend(files={
            "main.js": "import React from 'react';\nimport { useState, useEffect } from 'react';",
        })

        result = extract_imports(backend, "main.js")

        assert result.language == "javascript"
        assert len(result.imports) == 2
        assert result.imports[0].module == "react"
        assert result.imports[0].alias == "React"
        assert "useState" in result.imports[1].names

    def test_javascript_require(self):
        """Test extracting CommonJS require statements."""
        backend = InMemoryFileBackend(files={
            "main.js": "const express = require('express');",
        })

        result = extract_imports(backend, "main.js")

        assert result.imports[0].module == "express"
        assert result.imports[0].alias == "express"

    def test_typescript_import(self):
        """Test extracting TypeScript imports."""
        backend = InMemoryFileBackend(files={
            "main.ts": "import { Component } from '@angular/core';",
        })

        result = extract_imports(backend, "main.ts")

        assert result.language == "typescript"
        assert result.imports[0].module == "@angular/core"

    def test_unsupported_language(self):
        """Test that unsupported languages return error."""
        backend = InMemoryFileBackend(files={
            "main.go": "package main\nimport \"fmt\"",
        })

        result = extract_imports(backend, "main.go")

        assert result.language == "unknown"
        assert result.error is not None
        assert "not supported" in result.error

    def test_nonexistent_file(self):
        """Test extracting from nonexistent file."""
        backend = InMemoryFileBackend()

        result = extract_imports(backend, "missing.py")

        assert result.error is not None


class TestCreateFileTools:
    """Tests for create_file_tools factory."""

    def test_creates_tools(self, tmp_path):
        """Test that factory creates the expected tools."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("content")

        backend = LocalFileBackend(repo)
        tools = create_file_tools(backend)

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "search_for_files" in tool_names

    def test_tools_are_bound_to_backend(self, tmp_path):
        """Test that created tools use the bound backend."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("test content")

        backend = LocalFileBackend(repo)
        tools = create_file_tools(backend)

        # Find the read_file tool
        read_tool = next(t for t in tools if t.name == "read_file")

        # Invoke it
        result = read_tool.invoke({"file_path": "test.py"})

        assert result["content"] == "test content"

    def test_custom_max_chars(self, tmp_path):
        """Test that create_file_tools respects custom max_chars."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "big.py").write_text("x" * 500)

        backend = LocalFileBackend(repo)
        tools = create_file_tools(backend, max_chars=100)

        read_tool = next(t for t in tools if t.name == "read_file")
        result = read_tool.invoke({"file_path": "big.py"})

        assert result["truncated"] is True
        assert result["char_count"] == 100

    def test_custom_max_search_results(self, tmp_path):
        """Test that create_file_tools respects custom max_search_results."""
        repo = tmp_path / "repo"
        repo.mkdir()
        for i in range(10):
            (repo / f"match_{i}.py").write_text("keyword content")

        backend = LocalFileBackend(repo)
        tools = create_file_tools(backend, max_search_results=3)

        search_tool = next(t for t in tools if t.name == "search_for_files")
        result = search_tool.invoke({"keywords": ["match"]})

        assert result["total_found"] <= 3

    def test_default_limits_unchanged(self, tmp_path):
        """Test that default limits are preserved when no overrides given."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("content")

        backend = LocalFileBackend(repo)
        tools = create_file_tools(backend)

        read_tool = next(t for t in tools if t.name == "read_file")
        # Default max_chars should still be 13_500
        result = read_tool.invoke({"file_path": "test.py"})
        assert result["truncated"] is False  # 7 chars < 13500


class TestCreateAnalysisTools:
    """Tests for create_analysis_tools factory."""

    def test_creates_tools(self):
        """Test that factory creates the expected tools."""
        backend = InMemoryFileBackend(files={"test.py": "import os"})
        tools = create_analysis_tools(backend)

        assert len(tools) == 1
        assert tools[0].name == "extract_file_imports"

    def test_tool_works(self):
        """Test that the created tool works."""
        backend = InMemoryFileBackend(files={
            "main.py": "import json\nfrom pathlib import Path",
        })
        tools = create_analysis_tools(backend)

        result = tools[0].invoke({"file_path": "main.py"})

        assert result["language"] == "python"
        assert len(result["imports"]) == 2
