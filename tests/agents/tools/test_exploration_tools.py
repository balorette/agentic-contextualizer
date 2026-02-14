"""Tests for exploration utility tools."""

import pytest
from src.agents.tools.exploration_tools import list_key_files, read_file_snippet


class TestListKeyFilesFlat:
    """Tests for list_key_files tool with flat file list input."""

    def test_list_key_files_accepts_flat_list(self):
        """list_key_files should work with a flat file list from scan_structure."""
        flat_list = [
            "README.md",
            "pyproject.toml",
            "src/main.py",
            "src/__init__.py",
            "docs/guide.md",
            "Dockerfile",
        ]
        result = list_key_files.invoke({"file_list": flat_list})
        assert "README.md" in result["docs"]
        assert "pyproject.toml" in result["configs"]


class TestListKeyFiles:
    """Tests for list_key_files tool."""

    def test_list_key_files_finds_configs(self):
        """Test that configuration files are identified."""
        file_list = ["package.json", "pyproject.toml", "src/app.py"]

        result = list_key_files.invoke({"file_list": file_list})

        assert "package.json" in result["configs"]
        assert "pyproject.toml" in result["configs"]
        assert len(result["all_key_files"]) >= 2

    def test_list_key_files_finds_entry_points(self):
        """Test that entry point files are identified."""
        file_list = ["main.py", "index.js"]

        result = list_key_files.invoke({"file_list": file_list})

        assert "main.py" in result["entry_points"]
        assert "index.js" in result["entry_points"]

    def test_list_key_files_finds_docs(self):
        """Test that documentation files are identified."""
        file_list = ["README.md", "LICENSE", "CLAUDE.md"]

        result = list_key_files.invoke({"file_list": file_list})

        assert "README.md" in result["docs"]
        assert "LICENSE" in result["docs"]
        assert "CLAUDE.md" in result["docs"]

    def test_list_key_files_handles_nested_structure(self):
        """Test that key files in subdirectories are found."""
        file_list = ["src/main.py", "docs/index.md"]

        result = list_key_files.invoke({"file_list": file_list})

        assert "src/main.py" in result["entry_points"]
        assert "docs/index.md" in result["docs"]

    def test_list_key_files_empty_list(self):
        """Test handling of empty file list."""
        result = list_key_files.invoke({"file_list": []})

        assert len(result["all_key_files"]) == 0
        assert len(result["configs"]) == 0
        assert len(result["entry_points"]) == 0
        assert len(result["docs"]) == 0


class TestReadFileSnippet:
    """Tests for read_file_snippet tool."""

    def test_read_file_snippet_basic(self, tmp_path):
        """Test reading basic file snippet."""
        # Setup
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        # Execute
        result = read_file_snippet.invoke({
            "file_path": str(test_file),
            "start_line": 0,
            "num_lines": 3,
        })

        # Assert
        assert "error" not in result
        assert result["start_line"] == 0
        assert result["end_line"] == 2  # 0-indexed, inclusive
        assert result["total_lines"] == 5
        assert "line1" in result["content"]
        assert "line3" in result["content"]
        assert "line4" not in result["content"]

    def test_read_file_snippet_middle_of_file(self, tmp_path):
        """Test reading from middle of file."""
        # Setup
        test_file = tmp_path / "test.txt"
        lines = [f"line{i}\n" for i in range(1, 11)]
        test_file.write_text("".join(lines))

        # Execute - start at line 5, read 3 lines
        result = read_file_snippet.invoke({
            "file_path": str(test_file),
            "start_line": 5,
            "num_lines": 3,
        })

        # Assert
        assert "error" not in result
        assert result["start_line"] == 5
        assert result["end_line"] == 7
        assert "line6" in result["content"]  # 0-indexed, so line 5 is "line6"
        assert "line8" in result["content"]
        assert "line5" not in result["content"]

    def test_read_file_snippet_handles_EOF(self, tmp_path):
        """Test that reading past EOF doesn't error."""
        # Setup
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        # Execute - try to read 10 lines from line 1
        result = read_file_snippet.invoke({
            "file_path": str(test_file),
            "start_line": 1,
            "num_lines": 10,
        })

        # Assert - should read to end of file
        assert "error" not in result
        assert result["start_line"] == 1
        assert result["end_line"] == 2  # Last line available
        assert result["total_lines"] == 3

    def test_read_file_snippet_validates_bounds(self, tmp_path):
        """Test validation of input parameters."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Test negative start_line
        result = read_file_snippet.invoke({
            "file_path": str(test_file),
            "start_line": -1,
            "num_lines": 10,
        })
        assert "error" in result
        assert "must be non-negative" in result["error"]

        # Test zero num_lines
        result = read_file_snippet.invoke({
            "file_path": str(test_file),
            "start_line": 0,
            "num_lines": 0,
        })
        assert "error" in result
        assert "must be positive" in result["error"]

        # Test num_lines too large
        result = read_file_snippet.invoke({
            "file_path": str(test_file),
            "start_line": 0,
            "num_lines": 600,
        })
        assert "error" in result
        assert "cannot exceed 500" in result["error"]

    def test_read_file_snippet_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        # Execute
        result = read_file_snippet.invoke({
            "file_path": "/nonexistent/file.txt",
            "start_line": 0,
            "num_lines": 10,
        })

        # Assert
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_read_file_snippet_start_beyond_file(self, tmp_path):
        """Test error when start_line is beyond file length."""
        # Setup
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\n")

        # Execute - start at line 10 when file only has 2 lines
        result = read_file_snippet.invoke({
            "file_path": str(test_file),
            "start_line": 10,
            "num_lines": 5,
        })

        # Assert
        assert "error" in result
        assert "exceeds file length" in result["error"]

    def test_read_file_snippet_default_params(self, tmp_path):
        """Test that default parameters work correctly."""
        # Setup
        test_file = tmp_path / "test.txt"
        lines = [f"line{i}\n" for i in range(1, 101)]  # 100 lines
        test_file.write_text("".join(lines))

        # Execute - use defaults (start_line=0, num_lines=50)
        result = read_file_snippet.invoke({"file_path": str(test_file)})

        # Assert
        assert "error" not in result
        assert result["start_line"] == 0
        assert result["end_line"] == 49  # First 50 lines (0-49)
        assert result["total_lines"] == 100
