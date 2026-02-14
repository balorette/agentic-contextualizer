# Agent Token Consumption Fixes — Design

## Problem

Debug log analysis of a scoped agent run revealed 214k input tokens across 9 API calls.
Root causes:

1. **Tool output defaults too generous for agent mode** — `grep_in_files` returns up to 50
   results, `search_for_files` returns up to 30, `read_file` allows 13,500 chars. These
   accumulate in the LangChain message history and compound across calls.
2. **Middleware truncation limit too high** — `max_tool_output_chars` defaults to 12,000,
   which still allows large tool results into the conversation.
3. **No system prompt guidance** — The agent has no guidance to request smaller result sets.
4. **Grep context lines add bulk** — `context_lines=2` in grep doubles output size vs 1.

## Approach: Agent-Aware Tool Defaults

Pass tighter limits from the scoped agent factory to the tool factory functions. No new
config fields, no wrapper layers. The tool factories already accept a `backend` argument;
we add optional limit parameters with backward-compatible defaults so non-agent callers
are unaffected.

## Fixes

### Fix 1: Tool factory functions accept optional limits

**`create_file_tools(backend, max_chars=13_500)`**
- Scoped agent passes `max_chars=8_000`

**`create_search_tools(backend, max_grep_results=50, max_search_results=30, context_lines=2)`**
- Scoped agent passes `max_grep_results=15, max_search_results=15, context_lines=1`

Existing callers continue to get the current defaults.

### Fix 2: Lower `max_tool_output_chars` default

In `config.py`, change `max_tool_output_chars` default from 12,000 to 6,000. This
tightens the middleware safety net so even if a tool produces large output, it gets
truncated before entering the message history.

### Fix 3: System prompt guidance

Add a short section to `SCOPED_AGENT_SYSTEM_PROMPT` telling the agent to use small
`max_results` values (5-10) and avoid reading files larger than necessary.

### Fix 4: Reduce grep context_lines for agent mode

Pass `context_lines=1` (down from 2) when the scoped agent creates search tools.

## Files to Modify

- `src/agents/tools/file.py` — `create_file_tools()` signature
- `src/agents/tools/search.py` — `create_search_tools()` signature + `grep_pattern()`
- `src/agents/scoper/agent.py` — pass tighter defaults + system prompt update
- `src/agents/config.py` — `max_tool_output_chars` default 12000 -> 6000

## Expected Impact

- Tool outputs shrink ~50-60% per call
- Middleware truncation catches remaining outliers at 6k instead of 12k
- Cumulative input tokens across a 9-call run should drop from ~214k to ~100-120k
- No behavioral change for non-agent callers (backward compatible defaults)
