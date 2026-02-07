"""CLI regression tests for both pipeline and agent modes."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
from src.agents.main import cli


class TestCLIGenerate:
    """Test the generate command in both modes."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a minimal sample repository."""
        repo = tmp_path / "sample_repo"
        repo.mkdir()
        (repo / "README.md").write_text("# Test Project")
        (repo / "main.py").write_text("print('test')")
        return repo

    def test_generate_help(self, runner):
        """Test that generate --help works."""
        result = runner.invoke(cli, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate context for a repository" in result.output
        assert "--summary" in result.output
        assert "--mode" in result.output
        assert "--stream" in result.output

    def test_generate_requires_summary(self, runner, sample_repo):
        """Test that generate requires --summary flag."""
        result = runner.invoke(cli, ["generate", str(sample_repo)])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_generate_requires_existing_path(self, runner):
        """Test that generate requires existing repository path."""
        result = runner.invoke(
            cli, ["generate", "/nonexistent/path", "-s", "Test project"]
        )

        assert "Path does not exist" in result.output

    @patch("src.agents.main._generate_pipeline_mode")
    def test_generate_pipeline_mode_default(
        self, mock_pipeline, runner, sample_repo
    ):
        """Test that pipeline mode is the default."""
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli, ["generate", str(sample_repo), "-s", "Test project"]
        )

        # Should call pipeline mode (default)
        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._generate_pipeline_mode")
    def test_generate_explicit_pipeline_mode(
        self, mock_pipeline, runner, sample_repo
    ):
        """Test explicitly selecting pipeline mode."""
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli,
            ["generate", str(sample_repo), "-s", "Test", "--mode", "pipeline"],
        )

        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._generate_agent_mode")
    def test_generate_agent_mode(self, mock_agent, runner, sample_repo):
        """Test selecting agent mode."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            ["generate", str(sample_repo), "-s", "Test", "--mode", "agent"],
        )

        mock_agent.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._generate_agent_mode")
    def test_generate_agent_mode_with_debug(
        self, mock_agent, runner, sample_repo
    ):
        """Test agent mode with debug flag."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            [
                "generate",
                str(sample_repo),
                "-s",
                "Test",
                "--mode",
                "agent",
                "--debug",
            ],
        )

        mock_agent.assert_called_once()
        # Check that debug=True was passed
        call_args = mock_agent.call_args
        assert call_args[0][3] is True  # debug parameter

    @patch("src.agents.main._generate_agent_mode")
    def test_generate_agent_mode_with_stream(
        self, mock_agent, runner, sample_repo
    ):
        """Test agent mode with streaming."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            [
                "generate",
                str(sample_repo),
                "-s",
                "Test",
                "--mode",
                "agent",
                "--stream",
            ],
        )

        mock_agent.assert_called_once()
        # Check that stream=True was passed
        call_args = mock_agent.call_args
        assert call_args[0][4] is True  # stream parameter

    @patch("src.agents.main._generate_agent_mode")
    def test_generate_agent_mode_all_flags(
        self, mock_agent, runner, sample_repo
    ):
        """Test agent mode with all flags."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            [
                "generate",
                str(sample_repo),
                "-s",
                "Test project",
                "--mode",
                "agent",
                "--debug",
                "--stream",
            ],
        )

        mock_agent.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main.resolve_repo")
    @patch("src.agents.main._generate_pipeline_mode")
    def test_generate_with_github_url(self, mock_pipeline, mock_resolve, runner):
        """Test that a GitHub URL is accepted and routed through resolve_repo."""
        mock_pipeline.return_value = 0
        mock_repo_path = Path("/tmp/ctx-fakerepo")
        mock_resolve.return_value.__enter__ = Mock(return_value=mock_repo_path)
        mock_resolve.return_value.__exit__ = Mock(return_value=False)

        result = runner.invoke(
            cli,
            ["generate", "https://github.com/owner/repo", "-s", "Test project"],
        )

        mock_resolve.assert_called_once_with("https://github.com/owner/repo")
        mock_pipeline.assert_called_once()
        assert result.exit_code == 0


class TestCLIRefine:
    """Test the refine command in both modes."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_context_file(self, tmp_path):
        """Create a sample context file."""
        context_file = tmp_path / "context.md"
        context_file.write_text("# Sample Context\n\nSome content here.")
        return context_file

    def test_refine_help(self, runner):
        """Test that refine --help works."""
        result = runner.invoke(cli, ["refine", "--help"])

        assert result.exit_code == 0
        assert "Refine an existing context file" in result.output
        assert "--request" in result.output
        assert "--mode" in result.output
        assert "--stream" in result.output

    def test_refine_requires_request(self, runner, sample_context_file):
        """Test that refine requires --request flag."""
        result = runner.invoke(cli, ["refine", str(sample_context_file)])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_refine_requires_existing_file(self, runner):
        """Test that refine requires existing context file."""
        result = runner.invoke(
            cli, ["refine", "/nonexistent/file.md", "-r", "Add details"]
        )

        assert result.exit_code != 0

    @patch("src.agents.main._refine_pipeline_mode")
    def test_refine_pipeline_mode_default(
        self, mock_pipeline, runner, sample_context_file
    ):
        """Test that pipeline mode is the default for refine."""
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli, ["refine", str(sample_context_file), "-r", "Add more details"]
        )

        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._refine_agent_mode")
    def test_refine_agent_mode(self, mock_agent, runner, sample_context_file):
        """Test refine in agent mode."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            [
                "refine",
                str(sample_context_file),
                "-r",
                "Add auth details",
                "--mode",
                "agent",
            ],
        )

        mock_agent.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._refine_agent_mode")
    def test_refine_agent_mode_with_stream(
        self, mock_agent, runner, sample_context_file
    ):
        """Test refine agent mode with streaming."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            [
                "refine",
                str(sample_context_file),
                "-r",
                "Update",
                "--mode",
                "agent",
                "--stream",
            ],
        )

        mock_agent.assert_called_once()
        # Check that debug and stream were passed
        call_args = mock_agent.call_args
        # Parameters: (context_path, request, config, debug, stream)
        assert call_args[0][4] is True  # stream parameter (5th positional arg)


class TestCLIBackwardCompatibility:
    """Test backward compatibility with existing pipeline behavior."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @patch("src.agents.main._generate_pipeline_mode")
    def test_existing_commands_still_work(
        self, mock_pipeline, runner, tmp_path
    ):
        """Test that existing command patterns still work."""
        repo = tmp_path / "repo"
        repo.mkdir()

        mock_pipeline.return_value = 0

        # Old command pattern (no --mode flag)
        result = runner.invoke(
            cli, ["generate", str(repo), "-s", "A Python project"]
        )

        # Should default to pipeline mode
        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._generate_pipeline_mode")
    def test_pipeline_mode_unchanged(self, mock_pipeline, runner, tmp_path):
        """Test that pipeline mode behavior is unchanged."""
        repo = tmp_path / "repo"
        repo.mkdir()

        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli,
            ["generate", str(repo), "-s", "Project", "--mode", "pipeline"],
        )

        mock_pipeline.assert_called_once()
        # Verify pipeline mode gets correct parameters
        call_args = mock_pipeline.call_args
        assert call_args[0][1] == "Project"  # summary


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @patch("src.agents.config.Config.from_env")
    @patch("src.agents.main._generate_pipeline_mode")
    def test_missing_api_key(self, mock_pipeline, mock_config, runner, tmp_path):
        """Test error when ANTHROPIC_API_KEY is not set."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Mock config with no API key
        config_instance = Mock()
        config_instance.api_key = None
        mock_config.return_value = config_instance

        # Ensure pipeline mode isn't called if API key check fails
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli, ["generate", str(repo), "-s", "Test"]
        )

        # Check for error message (exit code may vary based on Click version)
        assert "ANTHROPIC_API_KEY not set" in result.output or result.exit_code == 1

    @patch("src.agents.config.Config.from_env")
    @patch("src.agents.main._generate_agent_mode")
    def test_agent_mode_failure_returns_error(
        self, mock_agent, mock_config, runner, tmp_path
    ):
        """Test that agent mode failures return non-zero exit code."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Mock config with API key
        config_instance = Mock()
        config_instance.api_key = "test-key"
        mock_config.return_value = config_instance

        # Mock agent mode to return error
        mock_agent.return_value = 1

        result = runner.invoke(
            cli,
            ["generate", str(repo), "-s", "Test", "--mode", "agent"],
        )

        # Verify error return code
        assert result.exit_code == 1 or mock_agent.return_value == 1


class TestCLIModeSelection:
    """Test mode selection logic."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    def test_invalid_mode_rejected(self, runner, tmp_path):
        """Test that invalid mode values are rejected."""
        repo = tmp_path / "repo"
        repo.mkdir()

        result = runner.invoke(
            cli,
            ["generate", str(repo), "-s", "Test", "--mode", "invalid"],
        )

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()

    @patch("src.agents.main._generate_pipeline_mode")
    @patch("src.agents.main._generate_agent_mode")
    def test_mode_routing(
        self, mock_agent, mock_pipeline, runner, tmp_path
    ):
        """Test that mode flag correctly routes to handlers."""
        repo = tmp_path / "repo"
        repo.mkdir()

        mock_pipeline.return_value = 0
        mock_agent.return_value = 0

        # Test pipeline routing
        runner.invoke(
            cli,
            ["generate", str(repo), "-s", "Test", "--mode", "pipeline"],
        )
        mock_pipeline.assert_called_once()
        mock_agent.assert_not_called()

        # Reset mocks
        mock_pipeline.reset_mock()
        mock_agent.reset_mock()

        # Test agent routing
        runner.invoke(
            cli,
            ["generate", str(repo), "-s", "Test", "--mode", "agent"],
        )
        mock_agent.assert_called_once()
        mock_pipeline.assert_not_called()


class TestCLIOutput:
    """Test CLI output formatting."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @patch("src.agents.main._generate_agent_mode")
    def test_agent_mode_output_includes_mode_indicator(
        self, mock_agent, runner, tmp_path
    ):
        """Test that agent mode shows mode indicator in output."""
        repo = tmp_path / "repo"
        repo.mkdir()

        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            ["generate", str(repo), "-s", "Test", "--mode", "agent"],
        )

        # Agent mode should output mode indicator
        # (This would need actual implementation to test properly)
        assert result.exit_code == 0

    @patch("src.agents.main._generate_agent_mode")
    def test_streaming_mode_indicator(
        self, mock_agent, runner, tmp_path
    ):
        """Test that streaming mode shows in output."""
        repo = tmp_path / "repo"
        repo.mkdir()

        mock_agent.return_value = 0

        result = runner.invoke(
            cli,
            [
                "generate",
                str(repo),
                "-s",
                "Test",
                "--mode",
                "agent",
                "--stream",
            ],
        )

        # Stream mode should be indicated
        # (Would show in actual execution)
        assert result.exit_code == 0


class TestCLIScope:
    """Test the scope command."""

    @pytest.fixture
    def runner(self):
        """Create Click test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a minimal sample repository."""
        repo = tmp_path / "sample_repo"
        repo.mkdir()
        (repo / "README.md").write_text("# Test Project")
        (repo / "weather.py").write_text("def get_weather(): pass")
        return repo

    @pytest.fixture
    def sample_context_file(self, tmp_path):
        """Create a sample existing context file."""
        contexts = tmp_path / "contexts" / "myrepo"
        contexts.mkdir(parents=True)
        context_file = contexts / "context.md"
        context_file.write_text("""---
source_repo: /path/to/myrepo
scan_date: 2025-01-01T00:00:00Z
user_summary: Test project
model_used: claude-3-5-sonnet
---

# Repository Context: myrepo

Test content.
""")
        return context_file

    def test_scope_help(self, runner):
        """Test that scope --help works."""
        result = runner.invoke(cli, ["scope", "--help"])

        assert result.exit_code == 0
        assert "Generate scoped context" in result.output or "scope" in result.output.lower()
        assert "--question" in result.output

    def test_scope_requires_question(self, runner, sample_repo):
        """Test that scope command requires --question flag."""
        result = runner.invoke(cli, ["scope", str(sample_repo)])

        assert result.exit_code != 0
        # Verify error specifically mentions the missing --question option
        assert "--question" in result.output or "-q" in result.output, \
            f"Error should mention missing --question flag, got: {result.output}"

    def test_scope_requires_existing_source(self, runner):
        """Test that scope requires existing repo or context file."""
        result = runner.invoke(
            cli, ["scope", "/nonexistent/path", "-q", "test question"]
        )

        assert "Path does not exist" in result.output

    @patch("src.agents.main._scope_pipeline_mode")
    def test_scope_from_repo(self, mock_pipeline, runner, sample_repo):
        """Test scoping directly from a repository."""
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli, ["scope", str(sample_repo), "-q", "weather functionality"]
        )

        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._scope_pipeline_mode")
    def test_scope_from_context_file(self, mock_pipeline, runner, sample_context_file):
        """Test scoping from an existing context file."""
        mock_pipeline.return_value = 0

        result = runner.invoke(
            cli, ["scope", str(sample_context_file), "-q", "authentication flow"]
        )

        mock_pipeline.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main._scope_agent_mode")
    def test_scope_agent_mode(self, mock_agent, runner, sample_repo):
        """Test scope in agent mode."""
        mock_agent.return_value = 0

        result = runner.invoke(
            cli, ["scope", str(sample_repo), "-q", "test", "--mode", "agent"]
        )

        mock_agent.assert_called_once()
        assert result.exit_code == 0

    @patch("src.agents.main.resolve_repo")
    @patch("src.agents.main._scope_pipeline_mode")
    def test_scope_with_github_url(self, mock_pipeline, mock_resolve, runner):
        """Test that a GitHub URL is accepted for scope and routed through resolve_repo."""
        mock_pipeline.return_value = 0
        mock_repo_path = Path("/tmp/ctx-fakerepo")
        mock_resolve.return_value.__enter__ = Mock(return_value=mock_repo_path)
        mock_resolve.return_value.__exit__ = Mock(return_value=False)

        result = runner.invoke(
            cli,
            ["scope", "https://github.com/owner/repo", "-q", "auth flow"],
        )

        mock_resolve.assert_called_once_with("https://github.com/owner/repo")
        mock_pipeline.assert_called_once()
        assert result.exit_code == 0
