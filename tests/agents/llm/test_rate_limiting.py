"""Tests for TPM rate limiting."""

import time
import threading
import pytest
from agents.llm.rate_limiting import TPMThrottle


class TestTPMThrottle:
    """Tests for sliding window TPM throttle."""

    def test_init_computes_effective_limit(self):
        """Effective limit should be max_tpm * safety_factor."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=0.85)
        assert throttle.effective_limit == 25500

    def test_record_usage_updates_current(self):
        """Recording usage should increase current_usage."""
        throttle = TPMThrottle(max_tpm=30000)
        assert throttle.current_usage == 0
        throttle.record_usage(5000)
        assert throttle.current_usage == 5000

    def test_remaining_budget_decreases(self):
        """Remaining budget should decrease as tokens are used."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=1.0)
        assert throttle.remaining_budget == 30000
        throttle.record_usage(10000)
        assert throttle.remaining_budget == 20000

    def test_wait_if_needed_returns_zero_when_under_budget(self):
        """Should return 0 wait time when under budget."""
        throttle = TPMThrottle(max_tpm=30000, safety_factor=1.0)
        waited = throttle.wait_if_needed(5000)
        assert waited == 0.0

    def test_wait_if_needed_blocks_when_over_budget(self):
        """Should wait when estimated tokens would exceed effective limit."""
        throttle = TPMThrottle(max_tpm=1000, safety_factor=1.0)
        throttle.record_usage(900)
        # This should need to wait — 900 + 200 > 1000
        start = time.monotonic()
        # Use a very short window for testing by monkey-patching
        throttle._window_seconds = 0.3
        waited = throttle.wait_if_needed(200)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2  # Should have waited near window expiry

    def test_old_entries_expire(self):
        """Entries older than 60s should not count toward usage."""
        throttle = TPMThrottle(max_tpm=30000)
        throttle._window_seconds = 0.2  # Short window for testing
        throttle.record_usage(20000)
        assert throttle.current_usage == 20000
        time.sleep(0.3)  # Wait for entries to expire
        assert throttle.current_usage == 0

    def test_thread_safety(self):
        """Concurrent record_usage calls should not lose data."""
        throttle = TPMThrottle(max_tpm=100000, safety_factor=1.0)
        num_threads = 10
        tokens_per_thread = 100

        def record():
            for _ in range(tokens_per_thread):
                throttle.record_usage(1)

        threads = [threading.Thread(target=record) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert throttle.current_usage == num_threads * tokens_per_thread
