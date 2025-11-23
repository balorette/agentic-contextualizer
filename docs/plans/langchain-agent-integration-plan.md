# LangChain Agent Integration Plan

## Goal
Transition the current deterministic pipeline into a LangChain-powered agent workflow so the system can reason about repository state, invoke scanning/analyzer/generation tools adaptively, and support richer refinement loops while preserving cost controls.

## Current State
- CLI orchestrates a strictly linear flow: scan ➜ metadata ➜ analyze ➜ generate/refine.
- LangChain is only used as a thin Anthropic client (`ChatAnthropic`), with no agent executors, tool abstractions, or memory.
- Each stage exchanges rich Python objects directly; prompts assume a single pass with two LLM calls.
- Observability, retry logic, and guardrails rely on bespoke code; no centralized tracing.

## Target Architecture (LangChain v1.0 + LangGraph)

### 1. Tool Layer (LangChain v1.0 `@tool` decorator)
   - Wrap existing pipeline stages as LangChain tools using `@tool` decorator:
     - `scan_structure`: Wraps `StructureScanner` - returns file tree and key metadata files
     - `extract_metadata`: Wraps `MetadataExtractor` - returns project type, dependencies, commands
     - `analyze_code`: Wraps `CodeAnalyzer` - returns architectural patterns and conventions
     - `generate_context`: Wraps `ContextGenerator` - produces final markdown context file
     - `refine_context`: Wraps `RefineContext` - updates existing context based on feedback
   - Add quick utility tools for exploration:
     - `list_key_files`: Returns list of entry points and config files
     - `read_file_snippet`: Reads specific file sections for targeted analysis
   - All tools use Pydantic models for input/output schemas (type-safe, auto-validated)

### 2. Agent Layer (LangChain v1.0 `create_agent`)
   - **Primary approach**: Use `langchain.agents.create_agent()` (v1.0 standard)
     - Simpler than building StateGraph manually for basic agent workflows
     - Built-in ReAct loop with tool calling
     - Easy to add middleware for guardrails, PII handling, summarization
   - **Alternative (future migration)**: LangGraph `StateGraph` for complex multi-step workflows
     - More control over state transitions and conditional routing
     - Better for parallel tool execution and human-in-the-loop
     - Migrate to this when needing advanced orchestration patterns
   - System prompt describes:
     - Available tools and their purpose
     - Budget constraints (aim for ≤2 expensive LLM calls: analyze + generate)
     - Failure modes (oversized repos, missing dependencies)
     - Expected output format (markdown with YAML frontmatter)

### 3. Execution & Memory (LangGraph Checkpointer)
   - **Checkpointing**: Use `MemorySaver` (dev) or `RedisSaver` (prod) for state persistence
     - Stores conversation history between refinement iterations
     - Caches intermediate artifacts (scan results, metadata) by thread_id
     - Enables resume-from-failure and human-in-the-loop refinement
   - **State management**:
     - Thread-based sessions: each repo analysis gets unique `thread_id`
     - Store scan artifacts in checkpointer state to avoid re-scanning on refinement
     - Track token usage per thread for budget enforcement

### 4. Observability & Guardrails (LangSmith)
   - **Tracing**: Use `@traceable` decorator on all tools and agent entrypoint
     - Automatic capture of inputs/outputs, token usage, latency
     - Track cost per operation (prompt_tokens, completion_tokens, total_cost)
     - Export traces to LangSmith for debugging and optimization
   - **Middleware** (LangChain v1.0 middleware system):
     - `SummarizationMiddleware`: Condense conversation history before hitting token limits
     - `HumanInTheLoopMiddleware`: Require approval for `generate_context` and `refine_context`
     - Custom `BudgetMiddleware`: Track and limit token spend per session
   - **Retry policies**: Add `RetryPolicy(max_attempts=3)` to tools that call external APIs or LLMs
   - **Error handling**: Surface tool errors to agent for recovery (e.g., "repo too large, try focusing on /src only")

## Implementation Tasks (For `superpowers:execute-plan`)

### Phase 1: Tool Layer Foundation

#### Task 1.1: Define Pydantic Schemas for Tool I/O
**File**: `src/agents/tools/schemas.py` (new)
- Create Pydantic models for all tool inputs/outputs:
  - `ScanStructureInput(repo_path: str, ignore_patterns: list[str])`
  - `ScanStructureOutput(file_tree: dict, key_files: dict)`
  - `ExtractMetadataInput(file_tree: dict, key_files: dict)`
  - `ExtractMetadataOutput(project_type: str, dependencies: list, commands: dict)`
  - `AnalyzeCodeInput(metadata: dict, user_summary: str)`
  - `AnalyzeCodeOutput(architecture: str, patterns: dict, conventions: list)`
  - `GenerateContextInput(scan_data: dict, metadata: dict, analysis: dict, user_summary: str)`
  - `GenerateContextOutput(context_md: str, output_path: str)`
  - `RefineContextInput(existing_context: str, refinement_request: str)`
  - `RefineContextOutput(updated_context: str)`
- Add validation logic (e.g., `repo_path` must exist, `file_tree` must be valid JSON)

#### Task 1.2: Create LangChain Tool Wrappers
**File**: `src/agents/tools/repository_tools.py` (new)
- Implement `@tool` decorated functions:
  ```python
  from langchain.tools import tool
  from src.agents.scanner.structure import StructureScanner
  from .schemas import ScanStructureInput, ScanStructureOutput

  @tool
  def scan_structure(repo_path: str, ignore_patterns: list[str] = None) -> dict:
      """Scan repository structure and identify key files."""
      scanner = StructureScanner()
      result = scanner.scan(repo_path, ignore_patterns or [])
      return ScanStructureOutput(**result).model_dump()
  ```
- Similar wrappers for: `extract_metadata`, `analyze_code`, `generate_context`, `refine_context`
- Ensure tools are **idempotent** (safe to call multiple times)
- Add error handling that returns descriptive error messages to the agent

#### Task 1.3: Create Utility Tools for Exploration
**File**: `src/agents/tools/exploration_tools.py` (new)
- Implement quick-access tools:
  ```python
  @tool
  def list_key_files(file_tree: dict) -> list[str]:
      """List entry points and configuration files."""
      # Extract from file_tree: package.json, main.py, etc.

  @tool
  def read_file_snippet(file_path: str, start_line: int = 0, num_lines: int = 50) -> str:
      """Read a snippet from a specific file."""
      # Safely read file with bounds checking
  ```

#### Task 1.4: Tool Integration Tests
**File**: `tests/agents/tools/test_repository_tools.py` (new)
- Test each tool wrapper in isolation
- Verify Pydantic schema validation (valid inputs pass, invalid inputs raise errors)
- Test idempotency (calling tools multiple times produces same result)
- Mock underlying pipeline components (StructureScanner, etc.)

### Phase 2: Agent Setup with LangChain v1.0

#### Task 2.1: Install Dependencies
**File**: `pyproject.toml`
- Add to dependencies:
  ```toml
  langchain = "^1.0.0"
  langgraph = "^0.6.0"
  langsmith = "^0.2.0"
  langchain-anthropic = "^0.3.0"
  ```
- Run: `uv pip install -e ".[dev]"`

#### Task 2.2: Implement Agent Factory
**File**: `src/agents/factory.py` (new)
- Create `create_contextualizer_agent()` function:
  ```python
  from langchain.agents import create_agent
  from langchain.chat_models import init_chat_model
  from .tools.repository_tools import scan_structure, extract_metadata, analyze_code, generate_context, refine_context

  def create_contextualizer_agent(model_name: str = "claude-sonnet-4-5-20250929"):
      model = init_chat_model(model_name)
      tools = [scan_structure, extract_metadata, analyze_code, generate_context, refine_context]

      system_prompt = """You are a repository context generator...
      [Include budget constraints, tool usage guidelines, output format]
      """

      agent = create_agent(
          model=model,
          tools=tools,
          system_prompt=system_prompt
      )
      return agent
  ```

#### Task 2.3: Add LangSmith Tracing Configuration
**File**: `src/agents/observability.py` (new)
- Setup LangSmith tracing:
  ```python
  from langsmith import Client, traceable
  import os

  def configure_tracing(project_name: str = "agentic-contextualizer"):
      os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
      os.environ["LANGSMITH_TRACING"] = "true"
      os.environ["LANGSMITH_PROJECT"] = project_name
  ```
- Wrap agent invocation with `@traceable`

#### Task 2.4: Implement Checkpointing
**File**: `src/agents/memory.py` (new)
- Setup MemorySaver for development:
  ```python
  from langgraph.checkpoint.memory import MemorySaver

  def create_checkpointer(backend: str = "memory"):
      if backend == "memory":
          return MemorySaver()
      elif backend == "redis":
          # Future: RedisSaver.from_conn_string(...)
          raise NotImplementedError("Redis backend not yet implemented")
  ```

### Phase 3: CLI Integration

#### Task 3.1: Update CLI to Support Agent Mode
**File**: `src/main.py`
- Add `--mode` flag: `pipeline` (default) or `agent`
- Implement agent-based generation:
  ```python
  if args.mode == "agent":
      from src.agents.factory import create_contextualizer_agent
      from src.agents.memory import create_checkpointer

      agent = create_contextualizer_agent()
      checkpointer = create_checkpointer()

      config = {"configurable": {"thread_id": generate_thread_id(args.repo_path)}}

      result = agent.invoke({
          "messages": [{"role": "user", "content": f"Generate context for {args.repo_path}. {args.summary}"}]
      }, config=config)
  ```

#### Task 3.2: Add Refinement Command for Agent Mode
**File**: `src/main.py`
- Update `--refine` to use agent with checkpointer:
  ```python
  if args.refine and args.mode == "agent":
      # Load existing context, resume thread
      result = agent.invoke({
          "messages": [{"role": "user", "content": f"Refine context: {args.refine}"}]
      }, config=config)  # Same thread_id resumes conversation
  ```

#### Task 3.3: Add Streaming Output
**File**: `src/agents/streaming.py` (new)
- Implement streaming for real-time feedback:
  ```python
  for chunk in agent.stream(messages, config=config, stream_mode="updates"):
      print(chunk)  # Show tool calls and intermediate results
  ```

### Phase 4: Middleware and Guardrails

#### Task 4.1: Implement Budget Middleware
**File**: `src/agents/middleware/budget.py` (new)
- Track token usage per session:
  ```python
  from langchain.agents.middleware import AgentMiddleware

  class BudgetMiddleware(AgentMiddleware):
      def __init__(self, max_tokens: int = 50000):
          self.max_tokens = max_tokens
          self.usage = {}

      def wrap_model_call(self, request, handler):
          # Track tokens, raise error if budget exceeded
  ```

#### Task 4.2: Add Summarization Middleware
**File**: `src/agents/factory.py` (update)
- Add `SummarizationMiddleware` to agent creation:
  ```python
  from langchain.agents.middleware import SummarizationMiddleware

  middleware = [
      SummarizationMiddleware(
          model=model_name,
          max_tokens_before_summary=10000
      )
  ]
  agent = create_agent(model, tools, system_prompt, middleware=middleware)
  ```

#### Task 4.3: Add Human-in-the-Loop Middleware
**File**: `src/agents/factory.py` (update)
- Add approval requirement for generation:
  ```python
  from langchain.agents.middleware import HumanInTheLoopMiddleware

  middleware.append(
      HumanInTheLoopMiddleware(
          interrupt_on={"generate_context": {"allowed_decisions": ["approve", "reject", "edit"]}}
      )
  )
  ```

### Phase 5: Testing and Validation

#### Task 5.1: Agent Integration Tests
**File**: `tests/agents/test_agent_integration.py` (new)
- Test end-to-end agent execution on sample repos
- Verify tool calling sequence (scan → metadata → analyze → generate)
- Test refinement loop (initial generation + refinement request)
- Test error recovery (malformed inputs, oversized repos)

#### Task 5.2: Budget and Cost Tests
**File**: `tests/agents/test_budget.py` (new)
- Verify token tracking accuracy
- Test budget enforcement (agent stops before exceeding limit)
- Test cost calculation (matches expected token usage × model pricing)

#### Task 5.3: CLI Regression Tests
**File**: `tests/test_cli.py` (update)
- Test both `--mode pipeline` and `--mode agent` produce valid context files
- Test `--refine` in both modes
- Verify backward compatibility (existing pipeline mode unchanged)

#### Task 5.4: LangSmith Trace Validation
- Manually verify traces in LangSmith dashboard
- Check token usage, latency, error rates
- Ensure all tool calls are captured with correct metadata

## Execution Timeline

### Phase 1: Tool Layer (Days 1-3)
- Day 1: Task 1.1 - Define all Pydantic schemas
- Day 2: Task 1.2 & 1.3 - Implement all tool wrappers
- Day 3: Task 1.4 - Write and validate tool integration tests

### Phase 2: Agent Setup (Days 4-6)
- Day 4: Task 2.1 & 2.2 - Install dependencies, implement agent factory
- Day 5: Task 2.3 & 2.4 - Setup tracing and checkpointing
- Day 6: Integration testing of agent with tools

### Phase 3: CLI Integration (Days 7-8)
- Day 7: Task 3.1 & 3.2 - Update CLI with agent mode and refinement
- Day 8: Task 3.3 - Add streaming output, test end-to-end

### Phase 4: Middleware & Guardrails (Days 9-10)
- Day 9: Task 4.1 & 4.2 - Implement budget and summarization middleware
- Day 10: Task 4.3 - Add human-in-the-loop, integrate all middleware

### Phase 5: Testing & Validation (Days 11-12)
- Day 11: Task 5.1 & 5.2 - Agent integration and budget tests
- Day 12: Task 5.3 & 5.4 - CLI regression tests, LangSmith validation

## Dependencies and Prerequisites

### Required Before Starting
1. **LangSmith Account**: Create account at smith.langchain.com, obtain API key
2. **Environment Setup**:
   - Add `LANGSMITH_API_KEY` to `.env`
   - Add `LANGSMITH_TRACING=true` to `.env`
   - Add `LANGSMITH_PROJECT=agentic-contextualizer` to `.env`
3. **Current Pipeline**: Ensure existing pipeline (StructureScanner, CodeAnalyzer, etc.) is stable and tested

### External Documentation References
- LangChain v1.0: `/websites/langchain_oss_python_releases_langchain-v1`
- LangGraph: `/langchain-ai/langgraph` (Context7)
- LangSmith: `/websites/smith_langchain` (Context7)

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Cost overruns** | High | Implement `BudgetMiddleware` from day 1; set conservative token limits (50k per session); monitor via LangSmith |
| **Complexity creep** | Medium | Start with `create_agent` (simpler); defer LangGraph StateGraph to Phase 2 after validating core workflow |
| **LLM reliability** | Medium | Keep `--mode pipeline` as default; agent mode opt-in only; implement retry policies on all tools |
| **Tool schema mismatches** | High | Use Pydantic strict validation; comprehensive tool tests before agent integration |
| **Middleware compatibility** | Low | Use only official LangChain v1.0 middleware; test each middleware in isolation first |
| **Memory/checkpointer failures** | Medium | Start with `MemorySaver` (in-memory); add error handling for checkpoint serialization |

## Success Criteria

### Functional Requirements
- ✅ Agent mode generates valid context files matching existing pipeline output format
- ✅ Refinement loop works: user can request changes, agent updates context without re-scanning
- ✅ All tool calls are correctly sequenced (scan → metadata → analyze → generate)
- ✅ Error recovery: agent handles oversized repos, missing files, invalid inputs gracefully
- ✅ Backward compatibility: `--mode pipeline` still works identically to current behavior

### Non-Functional Requirements
- ✅ Token usage stays within budget (≤50k tokens per session for typical repos)
- ✅ LangSmith traces capture all tool calls with accurate token counts
- ✅ Agent completes typical repo analysis in <2 minutes (excluding LLM latency)
- ✅ Test coverage ≥80% for new agent code (tools, factory, middleware)
- ✅ CLI experience remains simple: `python -m src.main /path/to/repo --mode agent`

### Validation Checklist (Before Merge)
- [ ] All Phase 1-5 tasks completed and tested
- [ ] At least 3 real repos tested end-to-end (small, medium, large)
- [ ] LangSmith dashboard shows accurate traces with no errors
- [ ] Budget enforcement verified (agent stops when hitting token limit)
- [ ] Documentation updated (README.md, CLAUDE.md with agent mode examples)
- [ ] Backward compatibility tested (existing pipeline users unaffected)
- [ ] Performance benchmarks collected (time, tokens, cost per repo size)

## Quick Reference

### Key LangChain v1.0 APIs Used
- `langchain.agents.create_agent()` - Main agent creation (simpler than StateGraph)
- `langchain.tools.tool` - Decorator for tool creation
- `langchain.chat_models.init_chat_model()` - Model initialization
- `langchain.agents.middleware.*` - Middleware for guardrails (Budget, Summarization, HITL)

### Key LangGraph APIs Used
- `langgraph.checkpoint.memory.MemorySaver` - In-memory checkpointing
- `langgraph.checkpoint.redis.RedisSaver` - Production checkpointing (future)
- `@traceable` - LangSmith tracing decorator

### Command Examples
```bash
# Generate context using agent mode
python -m src.main /path/to/repo --summary "Brief description" --mode agent

# Refine existing context
python -m src.main --refine "add more auth details" contexts/repo/context.md --mode agent

# Use legacy pipeline mode (default)
python -m src.main /path/to/repo --summary "Brief description" --mode pipeline
```

### File Structure After Implementation
```
src/
├── agents/
│   ├── __init__.py
│   ├── factory.py              # Agent creation (create_agent)
│   ├── memory.py               # Checkpointer setup
│   ├── observability.py        # LangSmith tracing config
│   ├── streaming.py            # Streaming output handler
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic I/O models
│   │   ├── repository_tools.py # Main pipeline tools
│   │   └── exploration_tools.py # Utility tools
│   └── middleware/
│       ├── __init__.py
│       └── budget.py           # Token budget tracking
├── scanner/                    # Existing
├── llm/                        # Existing
└── main.py                     # Updated with --mode flag

tests/
├── agents/
│   ├── tools/
│   │   └── test_repository_tools.py
│   ├── test_agent_integration.py
│   └── test_budget.py
└── test_cli.py                 # Updated with agent mode tests
```

---

**Plan Status**: Ready for execution with `superpowers:execute-plan`

**Last Updated**: 2025-11-23
**Author**: Claude Code (using LangChain v1.0 & LangGraph docs via Context7 MCP)
