"""Memory and checkpointing for agent state persistence."""

import hashlib
from pathlib import Path
from typing import Optional, Literal
from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer(backend: Literal["memory", "redis"] = "memory") -> MemorySaver:
    """Create a checkpointer for agent state persistence.

    Checkpointers allow agents to maintain conversation history and state
    across multiple invocations. This enables:
    - Multi-turn conversations with context retention
    - Resume-from-failure recovery
    - Refinement workflows without re-scanning

    Args:
        backend: Checkpointer backend ("memory" or "redis")
            - "memory": In-memory storage (default, good for development)
            - "redis": Redis-based storage (future, for production)

    Returns:
        Configured checkpointer instance

    Raises:
        NotImplementedError: If redis backend is requested (not yet implemented)

    Example:
        ```python
        from src.agents.memory import create_checkpointer
        from src.agents.factory import create_contextualizer_agent

        # Create agent with checkpointing
        checkpointer = create_checkpointer()
        agent = create_contextualizer_agent(checkpointer=checkpointer)

        # First invocation - generate context
        config = {"configurable": {"thread_id": "repo-123"}}
        result1 = agent.invoke({
            "messages": [{"role": "user", "content": "Generate context for /path/to/repo"}]
        }, config=config)

        # Second invocation - refine context (uses same thread_id)
        result2 = agent.invoke({
            "messages": [{"role": "user", "content": "Add more details about auth"}]
        }, config=config)
        # Agent remembers the previous conversation!
        ```

    Note:
        MemorySaver stores state in-process RAM. State is lost when
        process exits. For production, use Redis backend (future enhancement).
    """
    if backend == "memory":
        return MemorySaver()
    elif backend == "redis":
        # TODO: Implement Redis checkpointer for production
        # from langgraph.checkpoint.redis import RedisSaver
        # return RedisSaver.from_conn_string(redis_url)
        raise NotImplementedError(
            "Redis backend not yet implemented. Use backend='memory' for now."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'memory' or 'redis'.")


def generate_thread_id(repo_path: str, session_id: Optional[str] = None) -> str:
    """Generate a deterministic thread ID for a repository.

    Thread IDs are used to group related agent invocations. Using
    the same thread_id allows the agent to maintain conversation
    history and state.

    Args:
        repo_path: Path to the repository
        session_id: Optional session identifier for multiple concurrent analyses

    Returns:
        Deterministic thread ID string

    Example:
        ```python
        from src.agents.memory import generate_thread_id

        # Same repo always gets same thread_id
        thread1 = generate_thread_id("/path/to/myrepo")
        thread2 = generate_thread_id("/path/to/myrepo")
        assert thread1 == thread2

        # Different sessions get different thread_ids
        thread3 = generate_thread_id("/path/to/myrepo", session_id="session1")
        thread4 = generate_thread_id("/path/to/myrepo", session_id="session2")
        assert thread3 != thread4
        ```

    Note:
        Thread IDs are SHA-256 hashes of the normalized repo path,
        truncated to 16 characters for readability.
    """
    # Normalize path
    normalized_path = str(Path(repo_path).resolve())

    # Include session_id if provided
    if session_id:
        key = f"{normalized_path}:{session_id}"
    else:
        key = normalized_path

    # Generate deterministic hash
    hash_obj = hashlib.sha256(key.encode("utf-8"))
    thread_id = hash_obj.hexdigest()[:16]

    return f"repo-{thread_id}"


def create_agent_config(repo_path: str, session_id: Optional[str] = None, **extra_config) -> dict:
    """Create agent configuration with thread ID and optional extras.

    This is a convenience function that creates the config dict
    needed for agent.invoke() with checkpointing.

    Args:
        repo_path: Path to the repository being analyzed
        session_id: Optional session identifier
        **extra_config: Additional configuration options

    Returns:
        Configuration dictionary for agent.invoke()

    Example:
        ```python
        from src.agents.memory import create_agent_config, create_checkpointer
        from src.agents.factory import create_contextualizer_agent

        checkpointer = create_checkpointer()
        agent = create_contextualizer_agent(checkpointer=checkpointer)

        # Create config for this repo
        config = create_agent_config("/path/to/repo")

        # Use in agent invocation
        result = agent.invoke({"messages": [...]}, config=config)

        # Later, same repo uses same thread (maintains context)
        config2 = create_agent_config("/path/to/repo")
        result2 = agent.invoke({"messages": [...]}, config=config2)
        ```
    """
    thread_id = generate_thread_id(repo_path, session_id)

    config = {
        "configurable": {
            "thread_id": thread_id,
            **extra_config,
        }
    }

    return config


class CheckpointNotFoundError(Exception):
    """Raised when trying to resume from a non-existent checkpoint."""

    pass


def clear_checkpoint(checkpointer: MemorySaver, thread_id: str) -> None:
    """Clear checkpoint data for a specific thread.

    Useful for starting fresh with a repository that was previously analyzed.

    Args:
        checkpointer: The checkpointer instance
        thread_id: Thread ID to clear

    Example:
        ```python
        from src.agents.memory import create_checkpointer, generate_thread_id, clear_checkpoint

        checkpointer = create_checkpointer()
        thread_id = generate_thread_id("/path/to/repo")

        # Clear any existing state for this repo
        clear_checkpoint(checkpointer, thread_id)

        # Now agent will start fresh
        ```

    Note:
        This uses the checkpointer's public delete_thread API.
        For MemorySaver, this deletes in-memory state.
        For other backends (e.g., Redis), this would delete from the backend.
    """
    # Use the documented delete_thread API instead of accessing private storage
    checkpointer.delete_thread(thread_id)
