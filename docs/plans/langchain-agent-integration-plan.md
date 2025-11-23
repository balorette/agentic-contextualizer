# LangChain Agent Integration Plan

## Goal
Transition the current deterministic pipeline into a LangChain-powered agent workflow so the system can reason about repository state, invoke scanning/analyzer/generation tools adaptively, and support richer refinement loops while preserving cost controls.

## Current State
- CLI orchestrates a strictly linear flow: scan ➜ metadata ➜ analyze ➜ generate/refine.
- LangChain is only used as a thin Anthropic client (`ChatAnthropic`), with no agent executors, tool abstractions, or memory.
- Each stage exchanges rich Python objects directly; prompts assume a single pass with two LLM calls.
- Observability, retry logic, and guardrails rely on bespoke code; no centralized tracing.

## Target Architecture
1. **Tool Layer**
   - Wrap `StructureScanner`, `MetadataExtractor`, `CodeAnalyzer`, `ContextGenerator`, and `RefineContext` into LangChain tools with well-defined input/output schemas (pydantic models).
   - Expose quick summary tools (e.g., `ListKeyFiles`, `ReadFileSnippet`) for incremental exploration.

2. **Agent Layer**
   - Use a LangChain ReAct or OpenAI Functions-style agent (future: LangGraph workflow) that reasons over the repo summary, plans tool invocations, and maintains conversation history.
   - Provide system prompt instructions describing available tools, their contracts, and budget constraints (e.g., aim for ≤2 expensive LLM calls per run, fail-fast on oversized repos).

3. **Execution & Memory**
   - Employ an `AgentExecutor` with `ConversationBufferMemory` (or LangGraph state) to persist context between tool calls and refinements.
   - Persist intermediate artifacts (scan results, metadata) so repeated runs can reference cached outputs when nothing changed.

4. **Observability & Guardrails**
   - Instrument with LangSmith / LCEL callbacks for tracing tool usage, token spend, and latency.
   - Enforce max-depth and rate limits via tool wrappers; surface clear errors back to the agent for recovery.

## Workstreams & Tasks
1. **Planning & Enablement**
   - Finalize tool boundaries and I/O schemas.
   - Decide on primary agent type (ReAct vs. Functions vs. LangGraph) and target model (e.g., Anthropic Claude, GPT-5.1-Codex).

2. **Tool Abstraction**
   - Convert existing pipeline stages into LangChain tools (use `@tool` or `StructuredTool`).
   - Ensure tools are idempotent and side-effect-aware (e.g., generation writes files only when agent commits to final output).

3. **Agent Orchestration**
   - Implement the agent executor with selected model, memory, and toolset.
   - Update CLI entry points to initialize the agent, feed user summary/goals, and stream decisions/logs.

4. **Cost & Policy Controls**
   - Add budgeting module (token counters, max tool calls) to keep the “two LLM calls” promise when desired.
   - Provide configuration flags for deterministic vs. agentic mode.

5. **Refinement Workflow**
   - Enable multi-turn conversations where the agent reviews existing context, asks clarifying questions, and issues targeted tool calls for updates.

6. **Testing & Validation**
   - Expand unit/integration tests to cover tool wrappers and agent plans (use LangChain’s testing utilities or snapshot traces).
   - Add CLI regression tests for both pipeline modes.

## Milestones
1. **Week 1** – Tool interface design, schema definitions, doc updates.
2. **Week 2** – Tool implementation + initial agent executor (happy-path generation).
3. **Week 3** – Refinement loop, caching, cost controls, and observability hooks.
4. **Week 4** – Comprehensive testing, documentation, and rollout toggle (`--mode agentic|pipeline`).

## Risks & Mitigations
- **Cost overruns**: strictly track token usage per tool call; implement budget guardrails in agent prompt and executor.
- **Complexity creep**: keep initial agent simple (single-agent ReAct) before considering multi-agent LangGraph flows.
- **LLM reliability**: maintain fallback deterministic pipeline or allow agent to defer to canned flow when confidence is low.
- **Secret management**: ensure new tooling does not log sensitive paths; continue to rely on `.env` + dotenv for provider keys.

## Success Criteria
- Agent mode can generate/refine context files end-to-end without manual sequencing.
- Tool invocations are observable (traces) and stay within configured token budgets.
- Users can switch between legacy pipeline and agentic mode via CLI flag for backward compatibility.
