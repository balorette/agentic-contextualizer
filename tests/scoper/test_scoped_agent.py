"""Tests for scoped context agent factory."""

import pytest
from unittest.mock import patch, MagicMock


class TestCreateScopedAgent:
    """Tests for create_scoped_agent factory."""

    @pytest.fixture
    def sample_repo(self, tmp_path):
        """Create a sample repository."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "main.py").write_text("def main(): pass")
        (repo / "README.md").write_text("# Test Project")
        return repo

    @pytest.fixture
    def mock_config(self):
        """Mock Config.from_env to avoid needing real API key."""
        with patch("src.agents.scoper.agent.Config") as mock:
            config_instance = MagicMock()
            config_instance.model_name = "claude-sonnet-4-5-20250929"
            config_instance.api_key = "test-key"
            config_instance.output_dir = "contexts"
            mock.from_env.return_value = config_instance
            yield mock

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM provider."""
        with patch("src.agents.scoper.agent.create_llm_provider") as mock:
            yield mock

    @pytest.fixture
    def mock_create_agent(self):
        """Mock langchain create_agent."""
        with patch("src.agents.scoper.agent.create_agent") as mock:
            mock_agent = MagicMock()
            mock.return_value = mock_agent
            yield mock

    @pytest.fixture
    def mock_init_chat_model(self):
        """Mock langchain init_chat_model."""
        with patch("src.agents.scoper.agent.init_chat_model") as mock:
            yield mock

    def test_creates_agent_with_correct_tools(
        self,
        sample_repo,
        mock_config,
        mock_llm,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """Test that factory creates agent with expected tools."""
        from src.agents.scoper.agent import create_scoped_agent

        agent = create_scoped_agent(sample_repo)

        # Verify create_agent was called
        mock_create_agent.assert_called_once()

        # Check tools were passed
        call_kwargs = mock_create_agent.call_args[1]
        tools = call_kwargs["tools"]

        # Should have: file tools, analysis tools, code search tools, and generate_scoped_context
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "search_for_files" in tool_names
        assert "extract_file_imports" in tool_names
        assert "grep_in_files" in tool_names
        assert "find_code_definitions" in tool_names
        assert "generate_scoped_context" in tool_names

    def test_uses_custom_backend(
        self,
        sample_repo,
        mock_config,
        mock_llm,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """Test that factory uses provided backend."""
        from src.agents.scoper.agent import create_scoped_agent
        from src.agents.tools import InMemoryFileBackend

        custom_backend = InMemoryFileBackend(files={"test.py": "content"})

        agent = create_scoped_agent(
            sample_repo,
            file_backend=custom_backend,
        )

        # Agent should be created (backend is used internally)
        mock_create_agent.assert_called_once()

    def test_passes_system_prompt(
        self,
        sample_repo,
        mock_config,
        mock_llm,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """Test that factory passes system prompt to agent."""
        from src.agents.scoper.agent import create_scoped_agent, SCOPED_AGENT_SYSTEM_PROMPT

        agent = create_scoped_agent(sample_repo)

        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["system_prompt"] == SCOPED_AGENT_SYSTEM_PROMPT

    def test_passes_checkpointer(
        self,
        sample_repo,
        mock_config,
        mock_llm,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """Test that factory passes checkpointer to agent."""
        from src.agents.scoper.agent import create_scoped_agent

        mock_checkpointer = MagicMock()

        agent = create_scoped_agent(
            sample_repo,
            checkpointer=mock_checkpointer,
        )

        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["checkpointer"] == mock_checkpointer

    def test_debug_flag_passed(
        self,
        sample_repo,
        mock_config,
        mock_llm,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """Test that debug flag is passed to agent."""
        from src.agents.scoper.agent import create_scoped_agent

        agent = create_scoped_agent(sample_repo, debug=True)

        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["debug"] is True

    def test_config_from_env_called_once(
        self,
        sample_repo,
        mock_config,
        mock_llm,
        mock_create_agent,
        mock_init_chat_model,
    ):
        """Generator LLM provider should use the same config as the agent model."""
        from src.agents.scoper.agent import create_scoped_agent

        create_scoped_agent(sample_repo)

        calls = mock_config.from_env.call_count
        assert calls == 1, f"Config.from_env() called {calls} times, expected 1"


class TestCreateScopedAgentWithBudget:
    """Tests for create_scoped_agent_with_budget factory."""

    @pytest.fixture
    def mock_create_scoped_agent(self):
        """Mock create_scoped_agent."""
        with patch("src.agents.scoper.agent.create_scoped_agent") as mock:
            mock.return_value = MagicMock()
            yield mock

    @pytest.fixture
    def mock_budget_tracker(self):
        """Mock BudgetTracker - imported from middleware module."""
        with patch("src.agents.middleware.BudgetTracker") as mock:
            yield mock

    def test_returns_agent_and_tracker(
        self,
        tmp_path,
        mock_create_scoped_agent,
        mock_budget_tracker,
    ):
        """Test that factory returns both agent and tracker."""
        from src.agents.scoper.agent import create_scoped_agent_with_budget

        repo = tmp_path / "repo"
        repo.mkdir()

        agent, tracker = create_scoped_agent_with_budget(repo)

        assert agent is not None
        assert tracker is not None
        mock_budget_tracker.assert_called_once()

    def test_passes_budget_limits(
        self,
        tmp_path,
        mock_create_scoped_agent,
        mock_budget_tracker,
    ):
        """Test that budget limits are passed to tracker."""
        from src.agents.scoper.agent import create_scoped_agent_with_budget

        repo = tmp_path / "repo"
        repo.mkdir()

        agent, tracker = create_scoped_agent_with_budget(
            repo,
            max_tokens=10000,
            max_cost_usd=1.0,
        )

        mock_budget_tracker.assert_called_once_with(
            max_tokens=10000,
            max_cost_usd=1.0,
        )


class TestScopedAgentSystemPrompt:
    """Tests for the scoped agent system prompt."""

    def test_prompt_includes_tools(self):
        """Test that system prompt documents available tools."""
        from src.agents.scoper.agent import SCOPED_AGENT_SYSTEM_PROMPT

        assert "read_file" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "search_for_files" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "extract_file_imports" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "grep_in_files" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "find_code_definitions" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "generate_scoped_context" in SCOPED_AGENT_SYSTEM_PROMPT

    def test_prompt_includes_workflow(self):
        """Test that system prompt includes workflow guidance."""
        from src.agents.scoper.agent import SCOPED_AGENT_SYSTEM_PROMPT

        assert "Search" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "Read" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "Generate" in SCOPED_AGENT_SYSTEM_PROMPT

    def test_prompt_includes_budget_guidance(self):
        """Test that system prompt includes budget guidance."""
        from src.agents.scoper.agent import SCOPED_AGENT_SYSTEM_PROMPT

        assert "Budget" in SCOPED_AGENT_SYSTEM_PROMPT
        assert "10-20" in SCOPED_AGENT_SYSTEM_PROMPT  # file read budget
