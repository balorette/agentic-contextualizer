# Token Rate Limiting Design

**Date:** 2026-02-13
**Status:** Approved
**Problem:** Both agent modes hit 30K TPM (tokens per minute) input limits on Anthropic and LiteLLM gateway. The existing `InMemoryRateLimiter` only throttles requests per second, not token volume.

## Context

- Users have variable TPM limits depending on their provider tier
- Affects both full context agent mode and scoped context agent mode
- Must support Anthropic (Claude), OpenAI, and Gemini via LiteLLM
- Must follow DRY/SOLID principles and be easy to maintain

## Approach: Hybrid Three-Layer Defense

Three independent layers, each reducing the chance of hitting limits:

1. **Token Reduction** (pre-call): Estimate tokens, reject oversized requests
2. **Proactive Throttling** (pre-call): Sliding window tracks TPM, delays when approaching limit
3. **Reactive Safety Net** (post-call): Handle 429s with backoff, extract response headers

## Architecture

### Decorator Pattern

```
RateLimitedProvider (implements LLMProvider)
  ├── TokenEstimator    → pre-call token counting
  ├── TPMThrottle       → sliding window enforcement
  ├── RetryHandler      → 429 backoff + header extraction
  └── inner LLMProvider → AnthropicProvider or LiteLLMProvider
```

`RateLimitedProvider` wraps any `LLMProvider` without modifying it (Open/Closed). It implements `LLMProvider`, so it's a drop-in replacement (Liskov Substitution).

### File Layout

```
src/agents/
├── llm/
│   ├── provider.py              # Existing ABC (unchanged)
│   ├── litellm_provider.py      # Existing (unchanged)
│   ├── rate_limiting.py          # NEW: RateLimitedProvider, TPMThrottle, RetryHandler
│   └── token_estimator.py        # NEW: TokenEstimator protocol + implementations
├── middleware/
│   ├── budget.py                 # Existing (unchanged)
│   └── token_budget.py           # Updated: add optional throttle for agent mode
└── config.py                     # Updated: add TPM config fields
```

## Component Details

### 1. Configuration (config.py)

New fields replacing `rate_limit_rps` / `rate_limit_burst`:

```python
max_tpm: int = Field(default=30000)
tpm_safety_factor: float = Field(default=0.85)
max_tokens_per_call: Optional[int] = Field(default=None)
retry_max_attempts: int = Field(default=3)
retry_initial_wait: float = Field(default=2.0)
```

Environment variables:

```
MAX_TPM=30000
TPM_SAFETY_FACTOR=0.85
MAX_TOKENS_PER_CALL=8000
RETRY_MAX_ATTEMPTS=3
RETRY_INITIAL_WAIT=2.0
```

### 2. TokenEstimator (llm/token_estimator.py)

Protocol + one implementation using `litellm.token_counter()`.

```python
class TokenEstimator(Protocol):
    def estimate(self, messages: list[dict], model: str) -> int: ...

class LiteLLMTokenEstimator:
    def estimate(self, messages: list[dict], model: str) -> int:
        return litellm.token_counter(model=model, messages=messages)
```

Uses litellm which delegates to tiktoken for OpenAI, approximates for Claude/Gemini (~85-90% accurate). Sufficient for pre-call gating; exact counts come from post-call response metadata.

### 3. TPMThrottle (llm/rate_limiting.py)

Sliding window tracking tokens consumed in a 60-second window.

```python
class TPMThrottle:
    def __init__(self, max_tpm: int, safety_factor: float = 0.85): ...
    def wait_if_needed(self, estimated_tokens: int) -> float: ...
    def record_usage(self, actual_tokens: int) -> None: ...
    def current_usage(self) -> int: ...
    def remaining_budget(self) -> int: ...
```

Key behaviors:
- Uses `time.monotonic()` for clock-skew safety
- Thread-safe via `threading.Lock`
- `wait_if_needed()` calculates when enough old entries expire, sleeps until safe
- `record_usage()` tracks actual tokens (not estimates) for window accuracy
- Safety factor (0.85) reserves 15% buffer for estimation errors

### 4. RetryHandler (llm/rate_limiting.py)

Handles 429 errors with exponential backoff + jitter, extracts response headers.

```python
class RetryHandler:
    def __init__(self, max_attempts: int = 3, initial_wait: float = 2.0): ...
    def execute_with_retry(self, fn: Callable, *args, **kwargs) -> Any: ...
    def extract_rate_limit_info(self, response) -> Optional[RateLimitInfo]: ...

@dataclass
class RateLimitInfo:
    input_tokens_remaining: Optional[int] = None
    output_tokens_remaining: Optional[int] = None
    reset_at: Optional[datetime] = None
```

Backoff schedule: `initial_wait * 2^attempt` with +/-25% jitter.
Respects `retry-after` header when available.
Extracts provider-specific headers (Anthropic: `anthropic-ratelimit-*`, OpenAI: `x-ratelimit-*`).

### 5. RateLimitedProvider (llm/rate_limiting.py)

Decorator composing all three layers.

```python
class RateLimitedProvider(LLMProvider):
    def __init__(self, provider, throttle, estimator, retry_handler, max_tokens_per_call=None): ...
    def generate(self, prompt, system=None) -> LLMResponse: ...
    def generate_structured(self, prompt, system=None, schema=None) -> BaseModel: ...
```

Flow for each call:
1. Build messages, estimate tokens
2. Reject if over `max_tokens_per_call`
3. `throttle.wait_if_needed(estimated)`
4. `retry_handler.execute_with_retry(provider.generate, ...)`
5. `throttle.record_usage(response.tokens_used)`
6. Extract headers, update throttle if authoritative data available

Internal `_rate_limited_call(fn, messages)` method handles the shared 5-step flow for both `generate()` and `generate_structured()` (DRY).

### 6. Agent Mode Integration (middleware/token_budget.py)

Extend existing `TokenBudgetMiddleware` with optional throttle:

```python
class TokenBudgetMiddleware(AgentMiddleware):
    def __init__(
        self,
        max_input_tokens=None,
        max_tool_output_chars=12000,
        throttle: TPMThrottle | None = None,      # NEW
        estimator: TokenEstimator | None = None,   # NEW
    ): ...
```

In `before_model()`: trim messages first, then estimate trimmed messages, then `throttle.wait_if_needed()`. One middleware guarantees correct ordering.

### 7. Shared Throttle

The `TPMThrottle` instance is shared between pipeline and agent modes within a session:

```
Session
  └── TPMThrottle (one instance)
       ├── Pipeline mode: via RateLimitedProvider
       └── Agent mode: via TokenBudgetMiddleware
```

## Error Handling

Two custom exceptions:

- `TokenBudgetExceededError`: Single call exceeds `max_tokens_per_call`. User action: reduce input size.
- `TPMExhaustedError`: All retry attempts exhausted on 429s. Includes `retry_after` value. User action: wait or upgrade tier.

## Logging

| Event | Level | Example |
|-------|-------|---------|
| Throttle waiting | INFO | `TPM throttle: waiting 12.3s (23400/25500 tokens used)` |
| Retry on 429 | WARNING | `Rate limited by provider. Retry 1/3 in 4.0s` |
| Header update | DEBUG | `Provider reports 8000 input tokens remaining` |
| Per-call rejected | ERROR | `Estimated 12000 tokens exceeds 8000 per-call limit` |
| Retries exhausted | ERROR | `Rate limit: 3/3 retries exhausted` |

All logging via `logging` module. No `print()` in rate limiting code.

## Changes Summary

| File | Action | Scope |
|------|--------|-------|
| `llm/token_estimator.py` | CREATE | Protocol + LiteLLMTokenEstimator |
| `llm/rate_limiting.py` | CREATE | TPMThrottle, RetryHandler, RateLimitedProvider, exceptions |
| `config.py` | UPDATE | Add TPM fields, remove rate_limit_rps/rate_limit_burst |
| `llm/provider.py` | UPDATE | Wrap inner provider in create_llm_provider() |
| `middleware/token_budget.py` | UPDATE | Add optional throttle/estimator params |
| `factory.py` | UPDATE | Pass throttle/estimator to middleware |

## Design Decisions

1. **Decorator over modification**: Wrapping providers preserves existing code and tests.
2. **Protocol over ABC for TokenEstimator**: Pythonic, no inheritance required for future implementations.
3. **Extend middleware over new middleware**: Trimming and throttling are order-dependent; one middleware guarantees correctness.
4. **litellm.token_counter() over tiktoken**: Already a dependency, handles all providers. tiktoken only accurate for OpenAI.
5. **Safety factor 0.85**: At 30K TPM, reserves 4,500 tokens buffer for estimation errors on non-OpenAI models.
6. **Shared TPMThrottle instance**: Prevents pipeline + agent mode from independently consuming the full TPM budget.
