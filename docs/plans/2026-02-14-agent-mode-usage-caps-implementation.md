# Agent-Mode Usage Caps Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix agent-mode usage caps so TPM throttle records actual token usage and BudgetTracker is automatically enforced during agent LLM calls via LiteLLM.

**Architecture:** Add `after_model()` hook to `TokenBudgetMiddleware` that extracts `usage_metadata` from the last AIMessage after each model call, records it to the shared `TPMThrottle`, and feeds it to an optional `BudgetTracker`. Wire `BudgetTracker` into middleware via factory functions.

**Tech Stack:** Python, LangChain AgentMiddleware API, pytest

---

### Task 1: Add `after_model` to TokenBudgetMiddleware — Tests

**Files:**
- Modify: `tests/agents/test_token_budget_middleware.py`

**Step 1: Write failing tests for `after_model`**

Add a new test class `TestTokenBudgetMiddlewareAfterModel` to the existing test file. These tests cover: recording actual usage to throttle, recording to budget tracker, fallback to estimate, and budget enforcement.

```python
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from agents.middleware.token_budget import TokenBudgetMiddleware
from agents.middleware.budget import BudgetTracker, BudgetExceededError
from agents.llm.rate_limiting import TPMThrottle


class TestTokenBudgetMiddlewareAfterModel:
    """Tests for after_model hook — records actual usage post-call."""

    def _make_middleware(self, throttle=None, estimator=None, budget_tracker=None):
        return TokenBudgetMiddleware(
            throttle=throttle or TPMThrottle(max_tpm=100000, safety_factor=1.0),
            estimator=estimator or FakeEstimator(value=500),
            model_name="test-model",
            budget_tracker=budget_tracker,
        )

    def test_after_model_records_actual_usage_to_throttle(self):
        """after_model extracts usage_metadata from AIMessage and records to throttle."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        mw = self._make_middleware(throttle=throttle)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        mw.after_model(state, runtime=None)

        assert throttle.current_usage == 280

    def test_after_model_records_to_budget_tracker(self):
        """after_model feeds usage into BudgetTracker when present."""
        tracker = BudgetTracker(max_tokens=50000)
        mw = self._make_middleware(budget_tracker=tracker)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        mw.after_model(state, runtime=None)

        assert tracker.total_tokens == 400
        assert tracker.total_prompt_tokens == 300
        assert tracker.total_completion_tokens == 100

    def test_after_model_raises_on_budget_exceeded(self):
        """after_model raises BudgetExceededError when budget is blown."""
        tracker = BudgetTracker(max_tokens=100)
        mw = self._make_middleware(budget_tracker=tracker)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 80, "output_tokens": 50, "total_tokens": 130},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        with pytest.raises(BudgetExceededError):
            mw.after_model(state, runtime=None)

    def test_after_model_falls_back_to_estimate(self):
        """When AIMessage has no usage_metadata, fall back to last estimate."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        estimator = FakeEstimator(value=750)
        mw = self._make_middleware(throttle=throttle, estimator=estimator)

        # Simulate before_model storing the estimate
        state_before = {"messages": [{"role": "user", "content": "hi"}]}
        mw.before_model(state_before, runtime=None)

        # AIMessage without usage_metadata
        ai_msg = AIMessage(content="response")
        state_after = {"messages": [HumanMessage(content="hi"), ai_msg]}

        mw.after_model(state_after, runtime=None)

        # Should have recorded the estimate (750)
        assert throttle.current_usage == 750

    def test_after_model_no_messages_is_noop(self):
        """after_model with empty messages does nothing."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        mw = self._make_middleware(throttle=throttle)

        state = {"messages": []}
        mw.after_model(state, runtime=None)

        assert throttle.current_usage == 0

    def test_after_model_no_ai_message_is_noop(self):
        """after_model with no AIMessage at end does nothing."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        mw = self._make_middleware(throttle=throttle)

        state = {"messages": [HumanMessage(content="hi")]}
        mw.after_model(state, runtime=None)

        assert throttle.current_usage == 0

    def test_after_model_without_throttle_is_noop(self):
        """after_model with no throttle configured does nothing."""
        mw = TokenBudgetMiddleware(max_input_tokens=5000)

        ai_msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}

        # Should not raise
        mw.after_model(state, runtime=None)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_token_budget_middleware.py::TestTokenBudgetMiddlewareAfterModel -v`
Expected: FAIL — `after_model` not implemented yet, `budget_tracker` param doesn't exist

---

### Task 2: Implement `after_model` in TokenBudgetMiddleware

**Files:**
- Modify: `src/agents/middleware/token_budget.py`

**Step 1: Add `budget_tracker` param and `_last_estimate` to `__init__`**

In `TokenBudgetMiddleware.__init__`, add:
```python
budget_tracker: Optional["BudgetTracker"] = None,
```
And store:
```python
self.budget_tracker = budget_tracker
self._last_estimate: int = 0
```

Add to imports (TYPE_CHECKING block):
```python
from ..middleware.budget import BudgetTracker
```

**Step 2: Store estimate in `before_model`**

At the end of the TPM throttling block in `before_model`, save the estimate:
```python
self._last_estimate = estimated
```

**Step 3: Implement `after_model`**

```python
def after_model(self, state, runtime) -> dict[str, Any] | None:
    """Record actual token usage after the model call."""
    if not self.throttle:
        return None

    messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
    if not messages:
        return None

    # Find the last AIMessage
    last_msg = messages[-1]
    is_ai = (
        (isinstance(last_msg, dict) and last_msg.get("role") == "assistant")
        or getattr(last_msg, "type", None) == "ai"
    )
    if not is_ai:
        return None

    # Extract usage_metadata (LangChain standard)
    usage = getattr(last_msg, "usage_metadata", None)
    if usage:
        total = usage.get("total_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    else:
        logger.warning(
            "No usage_metadata on AIMessage — falling back to estimate (%d tokens)",
            self._last_estimate,
        )
        total = self._last_estimate
        input_tokens = self._last_estimate
        output_tokens = 0

    if total > 0:
        self.throttle.record_usage(total)

    if self.budget_tracker:
        self.budget_tracker.add_usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            operation="agent_model_call",
        )
        self.budget_tracker.check_budget()

    return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_token_budget_middleware.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/agents/middleware/token_budget.py tests/agents/test_token_budget_middleware.py
git commit -m "feat: add after_model hook to record actual token usage in agent mode"
```

---

### Task 3: Wire BudgetTracker into `build_token_middleware`

**Files:**
- Modify: `src/agents/llm/chat_model_factory.py`

**Step 1: Add `budget_tracker` parameter to `build_token_middleware()`**

```python
def build_token_middleware(
    config: Config,
    model_name: str,
    throttle: "TPMThrottle | None" = None,
    budget_tracker: "BudgetTracker | None" = None,
) -> "TokenBudgetMiddleware":
```

Add to the TYPE_CHECKING block:
```python
from ..middleware.budget import BudgetTracker
```

Pass it through to `TokenBudgetMiddleware`:
```python
return TokenBudgetMiddleware(
    max_input_tokens=config.max_input_tokens,
    max_tool_output_chars=config.max_tool_output_chars,
    throttle=throttle,
    estimator=estimator,
    model_name=_strip_provider_prefix(model_name),
    budget_tracker=budget_tracker,
)
```

**Step 2: Run existing tests to verify no regression**

Run: `uv run pytest tests/agents/test_token_budget_middleware.py tests/agents/test_budget.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/agents/llm/chat_model_factory.py
git commit -m "feat: build_token_middleware accepts optional BudgetTracker"
```

---

### Task 4: Wire BudgetTracker into factory functions

**Files:**
- Modify: `src/agents/factory.py`
- Modify: `src/agents/scoper/agent.py`

**Step 1: Update `create_contextualizer_agent_with_budget` in `factory.py`**

Change the function to create the `BudgetTracker` first, then pass it into the agent creation so the middleware gets it:

```python
def create_contextualizer_agent_with_budget(
    max_tokens: int = 50000,
    max_cost_usd: float = 5.0,
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    debug: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_litellm: bool = False,
    config: Optional["Config"] = None,
):
    from .config import Config
    from .middleware import BudgetTracker
    from .llm.chat_model_factory import build_chat_model, build_token_middleware

    if config is None:
        config = Config.from_env()

    if not use_litellm and config.llm_provider == "litellm":
        use_litellm = True

    model = build_chat_model(
        config=config,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        use_litellm=use_litellm,
        debug=debug,
    )

    tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)

    budget_mw = build_token_middleware(config, model_name, budget_tracker=tracker)

    tools = [
        scan_structure,
        extract_metadata,
        analyze_code,
        generate_context,
        refine_context,
        list_key_files,
        read_file_snippet,
    ]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[budget_mw],
        checkpointer=checkpointer,
        debug=debug,
    )

    return agent, tracker
```

**Step 2: Update `create_scoped_agent_with_budget` in `scoper/agent.py`**

Pass `BudgetTracker` through to `build_token_middleware`:

```python
def create_scoped_agent_with_budget(
    repo_path: str | Path,
    max_tokens: int = 30000,
    max_cost_usd: float = 2.0,
    model_name: str = "anthropic:claude-sonnet-4-5-20250929",
    checkpointer: Optional[object] = None,
    output_dir: str = "contexts",
    debug: bool = False,
    base_url: Optional[str] = None,
    config: Optional["Config"] = None,
):
    from ..middleware import BudgetTracker

    tracker = BudgetTracker(max_tokens=max_tokens, max_cost_usd=max_cost_usd)

    # Create the agent inline (similar to create_scoped_agent) but with tracker wired in
    repo_path = Path(repo_path).resolve()
    backend = LocalFileBackend(repo_path)

    if config is None:
        config = Config.from_env()

    use_litellm = config.llm_provider == "litellm"

    from ..llm.chat_model_factory import build_chat_model, build_token_middleware
    from ..llm.rate_limiting import TPMThrottle

    model = build_chat_model(
        config=config,
        model_name=model_name,
        base_url=base_url,
        use_litellm=use_litellm,
        debug=debug,
    )

    throttle = TPMThrottle(config.max_tpm, config.tpm_safety_factor)

    file_tools = create_file_tools(backend, max_chars=8000, max_search_results=15)
    analysis_tools = create_analysis_tools(backend)
    code_search_tools = create_search_tools(
        backend, max_grep_results=15, max_def_results=15, context_lines=1
    )

    llm_provider = create_llm_provider(config, throttle=throttle)
    generator = ScopedGenerator(llm_provider, output_dir)

    @tool
    def generate_scoped_context(
        question: str,
        relevant_file_paths: list[str],
        insights: str,
        code_references: list[dict] | None = None,
    ) -> dict:
        """Generate the final scoped context markdown file.

        Call this when you have gathered sufficient context to answer the question.
        Pass the PATHS of relevant files — the tool reads their contents automatically.

        Args:
            question: The original scope question being answered
            relevant_file_paths: List of file paths the agent determined are relevant
            insights: Your analysis and insights about the code
            code_references: Optional list of code reference dicts with keys:
                - path: File path
                - line_start: Starting line number
                - line_end: Optional ending line number
                - description: Brief description of what this code does

        Returns:
            Dictionary with:
            - output_path: Path to generated markdown file
            - error: Error message if generation failed
        """
        try:
            if not relevant_file_paths:
                return {"output_path": None, "error": "relevant_file_paths must not be empty."}

            relevant_files = {}
            for file_path in relevant_file_paths:
                content = backend.read_file(file_path)
                if content is not None:
                    relevant_files[file_path] = content

            if not relevant_files:
                return {
                    "output_path": None,
                    "error": f"Could not read any of the {len(relevant_file_paths)} provided file paths.",
                }

            refs = None
            if code_references:
                refs = []
                for ref in code_references:
                    try:
                        refs.append(CodeReference(
                            path=ref["path"],
                            line_start=ref["line_start"],
                            line_end=ref.get("line_end"),
                            description=ref.get("description", ""),
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue

            output_path = generator.generate(
                repo_name=repo_path.name,
                question=question,
                relevant_files=relevant_files,
                insights=insights,
                model_name=config.model_name,
                source_repo=str(repo_path),
                code_references=refs,
            )
            return {"output_path": str(output_path), "error": None}
        except Exception as e:
            logger.exception("generate_scoped_context failed")
            return {"output_path": None, "error": str(e)}

    tools = file_tools + analysis_tools + code_search_tools + [generate_scoped_context]
    budget_mw = build_token_middleware(config, model_name, throttle=throttle, budget_tracker=tracker)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SCOPED_AGENT_SYSTEM_PROMPT,
        middleware=[budget_mw],
        checkpointer=checkpointer,
        debug=debug,
    )

    return agent, tracker
```

**Step 3: Run all tests**

Run: `uv run pytest tests/agents/test_budget.py tests/agents/test_token_budget_middleware.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/agents/factory.py src/agents/scoper/agent.py
git commit -m "feat: wire BudgetTracker into agent middleware via factory functions"
```

---

### Task 5: Integration test — full round-trip with mocked model

**Files:**
- Modify: `tests/agents/test_token_budget_middleware.py`

**Step 1: Write integration test**

Add a test that simulates the full `before_model` -> model call -> `after_model` cycle to verify the throttle and tracker accumulate across multiple turns.

```python
class TestTokenBudgetMiddlewareRoundTrip:
    """Integration test: before_model + after_model across multiple turns."""

    def test_multiple_turns_accumulate_in_throttle_and_tracker(self):
        """Simulate 3 agent turns and verify cumulative tracking."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        tracker = BudgetTracker(max_tokens=50000)
        estimator = FakeEstimator(value=500)
        mw = TokenBudgetMiddleware(
            throttle=throttle,
            estimator=estimator,
            model_name="test-model",
            budget_tracker=tracker,
        )

        # Turn 1: 200 tokens
        state1 = {"messages": [{"role": "user", "content": "turn 1"}]}
        mw.before_model(state1, runtime=None)
        ai1 = AIMessage(
            content="resp 1",
            usage_metadata={"input_tokens": 150, "output_tokens": 50, "total_tokens": 200},
        )
        mw.after_model({"messages": [HumanMessage(content="turn 1"), ai1]}, runtime=None)

        # Turn 2: 400 tokens
        state2 = {"messages": [
            HumanMessage(content="turn 1"), ai1,
            HumanMessage(content="turn 2"),
        ]}
        mw.before_model(state2, runtime=None)
        ai2 = AIMessage(
            content="resp 2",
            usage_metadata={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400},
        )
        mw.after_model(
            {"messages": [HumanMessage(content="turn 1"), ai1, HumanMessage(content="turn 2"), ai2]},
            runtime=None,
        )

        # Turn 3: 600 tokens
        state3 = {"messages": [
            HumanMessage(content="turn 1"), ai1,
            HumanMessage(content="turn 2"), ai2,
            HumanMessage(content="turn 3"),
        ]}
        mw.before_model(state3, runtime=None)
        ai3 = AIMessage(
            content="resp 3",
            usage_metadata={"input_tokens": 450, "output_tokens": 150, "total_tokens": 600},
        )
        mw.after_model(
            {"messages": [
                HumanMessage(content="turn 1"), ai1,
                HumanMessage(content="turn 2"), ai2,
                HumanMessage(content="turn 3"), ai3,
            ]},
            runtime=None,
        )

        # Throttle: 200 + 400 + 600 = 1200
        assert throttle.current_usage == 1200

        # Tracker: same total
        assert tracker.total_tokens == 1200
        assert tracker.total_prompt_tokens == 900  # 150 + 300 + 450
        assert tracker.total_completion_tokens == 300  # 50 + 100 + 150
```

**Step 2: Run all middleware tests**

Run: `uv run pytest tests/agents/test_token_budget_middleware.py -v`
Expected: ALL PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/agents/test_token_budget_middleware.py
git commit -m "test: add round-trip integration test for agent-mode usage tracking"
```
