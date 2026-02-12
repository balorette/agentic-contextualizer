"""Tests for repository analysis tools."""

import pytest
from unittest.mock import Mock, patch
from src.agents.tools.repository_tools import (
    scan_structure,
    extract_metadata,
    analyze_code,
    generate_context,
    refine_context,
)


class TestScanStructure:
    """Tests for scan_structure tool."""

    def test_scan_structure_success(self, tmp_path):
        """Test successful repository scan."""
        # Setup
        (tmp_path / "file1.py").write_text("# file1")
        (tmp_path / "file2.py").write_text("# file2")
        (tmp_path / "file3.py").write_text("# file3")

        # Execute
        result = scan_structure.invoke({"repo_path": str(tmp_path)})

        # Assert
        assert "error" not in result
        assert result["total_files"] == 3
        assert "file_list" in result
        assert len(result["file_list"]) == 3

    def test_scan_structure_returns_flat_list(self, tmp_path):
        """Test that scan returns a flat file list (not nested tree)."""
        (tmp_path / "main.py").write_text("# main")
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "lib.py").write_text("# lib")

        result = scan_structure.invoke({"repo_path": str(tmp_path)})

        # Should be a flat list of path strings
        assert isinstance(result["file_list"], list)
        assert all(isinstance(f, str) for f in result["file_list"])

    def test_scan_structure_handles_error(self, tmp_path):
        """Test error handling when scan fails."""
        # Execute with non-existent path
        result = scan_structure.invoke({"repo_path": str(tmp_path / "nonexistent")})

        # Assert
        assert "error" in result
        assert "Failed to scan repository" in result["error"]

    def test_scan_structure_idempotent(self, tmp_path):
        """Test that calling scan_structure multiple times returns same result."""
        # Create a simple test structure
        (tmp_path / "test.py").write_text("print('hello')")

        # Execute multiple times
        result1 = scan_structure.invoke({"repo_path": str(tmp_path)})
        result2 = scan_structure.invoke({"repo_path": str(tmp_path)})

        # Assert - results should be identical
        assert result1["total_files"] == result2["total_files"]
        assert result1["total_dirs"] == result2["total_dirs"]


class TestExtractMetadata:
    """Tests for extract_metadata tool."""

    @patch("src.agents.tools.repository_tools.MetadataExtractor")
    def test_extract_metadata_success(self, mock_extractor_class, tmp_path):
        """Test successful metadata extraction."""
        mock_metadata = Mock()
        mock_metadata.project_type = "python"
        mock_metadata.dependencies = {"requests": "2.0.0", "pytest": "7.0.0"}
        mock_metadata.entry_points = ["main.py"]
        mock_metadata.key_files = ["pyproject.toml", "README.md"]

        mock_extractor_class.return_value.extract.return_value = mock_metadata

        result = extract_metadata.invoke({"repo_path": str(tmp_path)})

        assert "error" not in result
        assert result["project_type"] == "python"
        assert "requests" in result["dependencies"]
        assert "main.py" in result["entry_points"]
        assert "pyproject.toml" in result["key_files"]

    @patch("src.agents.tools.repository_tools.MetadataExtractor")
    def test_extract_metadata_limits_dependencies(self, mock_extractor_class, tmp_path):
        """Test that dependencies are limited to 20."""
        many_deps = {f"dep{i}": "1.0.0" for i in range(30)}
        mock_metadata = Mock()
        mock_metadata.project_type = "python"
        mock_metadata.dependencies = many_deps
        mock_metadata.entry_points = []
        mock_metadata.key_files = []

        mock_extractor_class.return_value.extract.return_value = mock_metadata

        result = extract_metadata.invoke({"repo_path": str(tmp_path)})

        assert len(result["dependencies"]) == 20

    @patch("src.agents.tools.repository_tools.MetadataExtractor")
    def test_extract_metadata_handles_error(self, mock_extractor_class, tmp_path):
        """Test error handling when extraction fails."""
        mock_extractor_class.return_value.extract.side_effect = Exception("Extraction failed")

        result = extract_metadata.invoke({"repo_path": str(tmp_path)})

        assert "error" in result
        assert "Failed to extract metadata" in result["error"]


class TestAnalyzeCode:
    """Tests for analyze_code tool."""

    @patch("src.agents.tools.repository_tools._get_llm_provider")
    @patch("src.agents.tools.repository_tools.CodeAnalyzer")
    def test_analyze_code_success(self, mock_analyzer_class, mock_llm_provider, tmp_path):
        """Test successful code analysis."""
        # Setup
        mock_analyzer = Mock()
        mock_analysis = Mock()
        mock_analysis.architecture_patterns = ["MVC", "REST API"]
        mock_analysis.coding_conventions = {"style": "PEP 8"}
        mock_analysis.tech_stack = ["Python", "FastAPI"]
        mock_analysis.insights = "Well-structured codebase"

        mock_analyzer.analyze.return_value = mock_analysis
        mock_analyzer_class.return_value = mock_analyzer

        # Execute
        result = analyze_code.invoke({
            "repo_path": str(tmp_path),
            "user_summary": "A test project",
            "metadata_dict": {"project_type": "python", "dependencies": {}, "entry_points": [], "key_files": []},
            "file_list": [],
        })

        # Assert
        assert "error" not in result
        assert "MVC" in result["architecture_patterns"]
        assert result["coding_conventions"]["style"] == "PEP 8"
        assert "Python" in result["tech_stack"]
        assert result["insights"] == "Well-structured codebase"

    @patch("src.agents.tools.repository_tools._get_llm_provider")
    @patch("src.agents.tools.repository_tools.CodeAnalyzer")
    def test_analyze_code_handles_error(self, mock_analyzer_class, mock_llm_provider, tmp_path):
        """Test error handling when analysis fails."""
        # Setup
        mock_analyzer = Mock()
        mock_analyzer.analyze.side_effect = Exception("Analysis failed")
        mock_analyzer_class.return_value = mock_analyzer

        # Execute
        result = analyze_code.invoke({
            "repo_path": str(tmp_path),
            "user_summary": "A test project",
            "metadata_dict": {"project_type": "python", "dependencies": {}, "entry_points": [], "key_files": []},
            "file_list": [],
        })

        # Assert
        assert "error" in result
        assert "Failed to analyze code" in result["error"]


class TestGenerateContext:
    """Tests for generate_context tool."""

    @patch("src.agents.tools.repository_tools._get_context_generator")
    def test_generate_context_success(self, mock_generator_func, tmp_path):
        """Test successful context generation."""
        # Setup
        mock_generator = Mock()
        output_path = tmp_path / "context.md"
        output_path.write_text("# Context\n\nThis is a test context file.")
        mock_generator.generate.return_value = output_path

        mock_generator_func.return_value = mock_generator

        # Execute
        result = generate_context.invoke({
            "repo_path": str(tmp_path),
            "user_summary": "A test project",
            "metadata_dict": {"project_type": "python", "dependencies": {}, "entry_points": [], "key_files": []},
            "analysis_dict": {
                "architecture_patterns": ["MVC"],
                "coding_conventions": {},
                "tech_stack": ["Python"],
                "insights": "Test",
            },
        })

        # Assert
        assert "error" not in result
        assert "context_md" in result
        assert "output_path" in result
        assert "Context" in result["context_md"]

    @patch("src.agents.tools.repository_tools._get_context_generator")
    def test_generate_context_truncates_long_content(self, mock_generator_func, tmp_path):
        """Test that long context content is truncated in output."""
        # Setup
        mock_generator = Mock()
        output_path = tmp_path / "context.md"
        # Create content longer than 500 characters
        long_content = "x" * 1000
        output_path.write_text(long_content)
        mock_generator.generate.return_value = output_path

        mock_generator_func.return_value = mock_generator

        # Execute
        result = generate_context.invoke({
            "repo_path": str(tmp_path),
            "user_summary": "A test project",
            "metadata_dict": {"project_type": "python", "dependencies": {}, "entry_points": [], "key_files": []},
            "analysis_dict": {
                "architecture_patterns": [],
                "coding_conventions": {},
                "tech_stack": [],
                "insights": "",
            },
        })

        # Assert - should be truncated
        assert len(result["context_md"]) <= 504  # 500 + "..."

    @patch("src.agents.tools.repository_tools._get_context_generator")
    def test_generate_context_handles_error(self, mock_generator_func, tmp_path):
        """Test error handling when generation fails."""
        # Setup
        mock_generator = Mock()
        mock_generator.generate.side_effect = Exception("Generation failed")
        mock_generator_func.return_value = mock_generator

        # Execute
        result = generate_context.invoke({
            "repo_path": str(tmp_path),
            "user_summary": "A test project",
            "metadata_dict": {"project_type": "python", "dependencies": {}, "entry_points": [], "key_files": []},
            "analysis_dict": {
                "architecture_patterns": [],
                "coding_conventions": {},
                "tech_stack": [],
                "insights": "",
            },
        })

        # Assert
        assert "error" in result
        assert "Failed to generate context" in result["error"]


class TestRefineContext:
    """Tests for refine_context tool."""

    @patch("src.agents.tools.repository_tools._get_context_generator")
    def test_refine_context_success(self, mock_generator_func, tmp_path):
        """Test successful context refinement."""
        # Setup
        context_file = tmp_path / "context.md"
        context_file.write_text("# Original Context")

        mock_generator = Mock()
        updated_content = "# Updated Context\n\nWith new information."
        context_file.write_text(updated_content)
        mock_generator.refine.return_value = context_file

        mock_generator_func.return_value = mock_generator

        # Execute
        result = refine_context.invoke({
            "context_file_path": str(context_file),
            "refinement_request": "Add more details about the architecture",
        })

        # Assert
        assert "error" not in result
        assert "updated_context" in result
        assert "output_path" in result
        assert "Updated Context" in result["updated_context"]

    @patch("src.agents.tools.repository_tools._get_context_generator")
    def test_refine_context_handles_error(self, mock_generator_func, tmp_path):
        """Test error handling when refinement fails."""
        # Setup
        context_file = tmp_path / "context.md"
        context_file.write_text("# Original")

        mock_generator = Mock()
        mock_generator.refine.side_effect = Exception("Refinement failed")
        mock_generator_func.return_value = mock_generator

        # Execute
        result = refine_context.invoke({
            "context_file_path": str(context_file),
            "refinement_request": "Add details",
        })

        # Assert
        assert "error" in result
        assert "Failed to refine context" in result["error"]
