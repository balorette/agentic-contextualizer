# LangSmith Trace Validation Guide

This guide explains how to validate agent execution using LangSmith tracing.

## Prerequisites

1. **LangSmith Account**: Create account at https://smith.langchain.com
2. **API Key**: Get API key from Settings > API Keys
3. **Environment Setup**:
   ```bash
   export LANGSMITH_API_KEY="your-api-key-here"
   export LANGSMITH_TRACING="true"
   export LANGSMITH_PROJECT="agentic-contextualizer"
   ```

Alternatively, add to `.env` file:
```bash
LANGSMITH_API_KEY=your-api-key-here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agentic-contextualizer
```

## Running Agent with Tracing

### Method 1: Using Agent Mode CLI

```bash
# Generate context with tracing
python -m src.agents.main generate /path/to/repo \
  -s "Your project description" \
  --mode agent

# With streaming for real-time feedback
python -m src.agents.main generate /path/to/repo \
  -s "Your project description" \
  --mode agent \
  --stream
```

### Method 2: Programmatic Usage

```python
from src.agents.factory import create_contextualizer_agent
from src.agents.memory import create_checkpointer, create_agent_config
from src.agents.observability import configure_tracing

# Configure tracing
configure_tracing(project_name="agentic-contextualizer")

# Create agent
checkpointer = create_checkpointer()
agent = create_contextualizer_agent(checkpointer=checkpointer)

# Run agent
config = create_agent_config("/path/to/repo")
result = agent.invoke({
    "messages": [{"role": "user", "content": "Generate context for /path/to/repo"}]
}, config=config)
```

## What to Validate in LangSmith Dashboard

### 1. Trace Completeness

**Navigate to**: LangSmith Dashboard > Projects > agentic-contextualizer > Traces

**Check**:
- ✅ All tool calls are captured
- ✅ Tool call sequence is logical (scan → metadata → analyze → generate)
- ✅ No missing steps in the execution flow

**Expected Tool Sequence**:
1. `scan_structure` - File tree scanning
2. `extract_metadata` - Project metadata extraction
3. `analyze_code` - LLM-based code analysis
4. `generate_context` - Final context file generation

**Screenshot Example**:
```
Trace View:
├─ Agent (root)
│  ├─ scan_structure
│  │  └─ Success (0.2s)
│  ├─ extract_metadata
│  │  └─ Success (0.1s)
│  ├─ analyze_code
│  │  └─ LLM Call (2.3s)
│  └─ generate_context
│     └─ LLM Call (3.1s)
```

### 2. Token Usage Validation

**Navigate to**: Trace Details > Usage tab

**Check**:
- ✅ Input tokens match expected range
- ✅ Output tokens match expected range
- ✅ Total tokens within budget
- ✅ Cost estimate is reasonable

**Expected Token Ranges** (typical repository):

| Operation        | Input Tokens | Output Tokens | Total   |
|------------------|--------------|---------------|---------|
| analyze_code     | 2,000-5,000  | 800-2,000     | ~5,000  |
| generate_context | 3,000-8,000  | 1,500-3,000   | ~8,000  |
| **Total Session**| **5,000-13k**| **2,300-5k**  | **~13k**|

**Cost Estimate** (Claude Sonnet 4.5):
- Input: $3.00 per 1M tokens
- Output: $15.00 per 1M tokens
- **Expected session cost**: $0.05 - $0.15

**Validation Steps**:
1. Open trace in LangSmith
2. Click on "Usage" tab
3. Verify token counts match expected ranges
4. Check that cost is reasonable
5. Compare with budget limits (default: 50k tokens, $5.00)

### 3. Latency Analysis

**Navigate to**: Trace Details > Timeline view

**Check**:
- ✅ Tool calls complete in reasonable time
- ✅ No unexplained delays
- ✅ LLM calls show expected latency

**Expected Latencies**:

| Operation        | Expected Time | Notes                    |
|------------------|---------------|--------------------------|
| scan_structure   | 0.1-0.5s      | Depends on repo size     |
| extract_metadata | 0.05-0.2s     | Fast file reading        |
| analyze_code     | 2-5s          | LLM call + processing    |
| generate_context | 3-8s          | LLM call + file write    |
| **Total**        | **5-15s**     | Excluding LLM API time   |

**Red Flags**:
- ❌ Tool calls taking >10s (investigate)
- ❌ Unexplained gaps in timeline
- ❌ Timeout errors

### 4. Error Rate Monitoring

**Navigate to**: Projects > agentic-contextualizer > Analytics

**Check**:
- ✅ Success rate > 95%
- ✅ No recurring error patterns
- ✅ Tool failures are handled gracefully

**Common Errors to Monitor**:
1. **Repository not found**: Invalid path provided
2. **LLM API errors**: Rate limiting, network issues
3. **Token limit exceeded**: Repository too large
4. **Permission errors**: File access issues

**Validation Steps**:
1. Review error logs in trace details
2. Check error distribution (should be <5%)
3. Verify error messages are descriptive
4. Confirm graceful degradation (tools return error dicts, not exceptions)

### 5. Metadata Accuracy

**Navigate to**: Trace Details > Metadata tab

**Check**:
- ✅ Thread ID is correct and consistent
- ✅ Model name matches configuration
- ✅ Tags are properly set
- ✅ User ID / session info is captured

**Expected Metadata**:
```json
{
  "thread_id": "repo-[16-char-hash]",
  "model": "anthropic:claude-sonnet-4-5-20250929",
  "project": "agentic-contextualizer",
  "tags": ["agent-mode"],
  "session_id": "optional-session-id"
}
```

## Trace Comparison

### Compare Pipeline vs Agent Mode

**Test Both Modes**:
```bash
# Pipeline mode (baseline)
python -m src.agents.main generate /path/to/repo \
  -s "Test project" \
  --mode pipeline

# Agent mode (with tracing)
python -m src.agents.main generate /path/to/repo \
  -s "Test project" \
  --mode agent
```

**Compare**:
1. **Output Quality**: Both should produce similar context files
2. **Token Usage**: Agent mode may use slightly more tokens
3. **Latency**: Agent mode has overhead from agent executor
4. **Traceability**: Agent mode provides detailed traces

## Automated Trace Validation

### Using LangSmith SDK

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# Get recent traces
traces = client.list_runs(
    project_name="agentic-contextualizer",
    start_time=datetime.now() - timedelta(hours=1),
)

for trace in traces:
    print(f"Trace ID: {trace.id}")
    print(f"  Status: {trace.status}")
    print(f"  Tokens: {trace.total_tokens}")
    print(f"  Cost: ${trace.total_cost:.4f}")
    print(f"  Duration: {trace.duration:.2f}s")

    # Validate token usage
    if trace.total_tokens > 50000:
        print(f"  ⚠️  WARNING: Exceeded token budget!")

    # Validate cost
    if trace.total_cost > 5.0:
        print(f"  ⚠️  WARNING: Exceeded cost budget!")
```

## Troubleshooting

### Tracing Not Working

**Symptoms**: No traces appear in LangSmith dashboard

**Solutions**:
1. Verify API key is set: `echo $LANGSMITH_API_KEY`
2. Check tracing is enabled: `echo $LANGSMITH_TRACING`
3. Verify project name matches dashboard
4. Check network connectivity to LangSmith API
5. Review application logs for connection errors

### Incomplete Traces

**Symptoms**: Some tool calls missing from traces

**Solutions**:
1. Ensure all tools are properly decorated
2. Check that checkpointer is enabled
3. Verify agent is created with tracing configured
4. Review for exceptions that might interrupt tracing

### Token Count Mismatches

**Symptoms**: Dashboard shows different tokens than logs

**Solutions**:
1. Check model being used (different models count differently)
2. Verify token extraction from response metadata
3. Compare with budget tracker logs
4. Check for streaming vs non-streaming differences

## Best Practices

1. **Run Test Traces First**: Test with small repositories before large ones
2. **Monitor Budget**: Set alerts for token/cost thresholds
3. **Review Regularly**: Check traces weekly for patterns
4. **Tag Traces**: Use tags to organize different test scenarios
5. **Archive Old Traces**: Clean up dashboard periodically
6. **Document Anomalies**: Keep notes on unusual traces

## Validation Checklist

Use this checklist for manual validation:

- [ ] Tracing is enabled (API key set, `LANGSMITH_TRACING=true`)
- [ ] Traces appear in LangSmith dashboard
- [ ] All tool calls are captured in traces
- [ ] Tool sequence is logical (scan → metadata → analyze → generate)
- [ ] Token usage is within expected ranges
- [ ] Cost estimates are reasonable
- [ ] Latency is acceptable (total <30s for typical repos)
- [ ] Error rate is <5%
- [ ] Metadata is accurate (thread_id, model, etc.)
- [ ] Both pipeline and agent modes work
- [ ] Refinement traces show conversation continuity

## Example Trace Analysis

### Successful Generation

```
Trace: Generate Context for sample-fastapi-app
Status: ✅ Success
Duration: 8.3s
Tokens: 11,234 (input: 7,842 / output: 3,392)
Cost: $0.074

Tool Calls:
├─ scan_structure (0.2s)
│  └─ Result: 42 files, 8 directories
├─ extract_metadata (0.1s)
│  └─ Result: Python project, FastAPI dependencies
├─ analyze_code (3.8s)
│  └─ LLM Call: 4,123 tokens
│  └─ Result: REST API, async patterns, pytest tests
└─ generate_context (4.2s)
   └─ LLM Call: 7,111 tokens
   └─ Result: context.md created (2.4 KB)
```

**Validation**: ✅ All metrics within expected ranges

### Failed Generation

```
Trace: Generate Context for invalid-repo
Status: ❌ Error
Duration: 0.5s
Error: Repository path does not exist

Tool Calls:
└─ scan_structure (0.1s)
   └─ Error: Path /invalid/repo does not exist
```

**Validation**: ✅ Error handled gracefully, descriptive message

## Summary

LangSmith tracing provides comprehensive visibility into agent execution. Use this guide to:
1. Verify all traces are captured correctly
2. Monitor token usage and costs
3. Analyze performance and latency
4. Detect and debug errors
5. Compare pipeline vs agent modes

For more information, see:
- LangSmith Documentation: https://docs.smith.langchain.com
- LangChain Tracing Guide: https://python.langchain.com/docs/langsmith/
