"""Configuration management for the Agentic Contextualizer."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "egg-info", ".egg-info", ".tox", ".nox",
})


class Config(BaseModel):
    """Application configuration."""

    # LLM Settings
    llm_provider: str = Field(default="anthropic")
    model_name: str = Field(default="claude-3-5-sonnet-20241022")
    api_key: Optional[str] = Field(default=None)
    api_base_url: Optional[str] = Field(default=None)

    # Provider-specific API keys
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)

    max_retries: int = Field(default=3)
    timeout: int = Field(default=60)

    # Rate Limiting — TPM-aware
    max_tpm: int = Field(default=30000)
    tpm_safety_factor: float = Field(default=0.85)
    max_tokens_per_call: Optional[int] = Field(default=None)
    retry_max_attempts: int = Field(default=3)
    retry_initial_wait: float = Field(default=2.0)

    # Token Budget
    max_output_tokens: Optional[int] = Field(default=16384)
    max_input_tokens: Optional[int] = Field(default=128_000)
    max_tool_output_chars: int = Field(default=12000)
    max_scan_files: int = Field(default=200)

    # Scanner Settings
    max_file_size: int = Field(default=1_000_000)  # 1MB
    ignored_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_IGNORED_DIRS))

    # Output Settings
    output_dir: Path = Field(default=Path("contexts"))

    @classmethod
    def from_env(cls, cli_overrides: Optional[dict] = None) -> "Config":
        """Load configuration from environment variables with optional CLI overrides.

        Args:
            cli_overrides: Dict of config fields to override (from CLI flags)

        Returns:
            Config instance with merged settings
        """
        def _parse_int(value: Optional[str], fallback: int) -> int:
            try:
                return int(value) if value is not None else fallback
            except ValueError:
                return fallback

        def _parse_float(value: Optional[str], fallback: float) -> float:
            try:
                return float(value) if value is not None else fallback
            except ValueError:
                return fallback

        ignored_dirs = list(DEFAULT_IGNORED_DIRS)
        extra_ignored = os.getenv("IGNORED_DIRS")
        if extra_ignored:
            ignored_dirs.extend(
                [entry.strip() for entry in extra_ignored.split(",") if entry.strip()]
            )

        output_dir_env = os.getenv("OUTPUT_DIR")

        # Read base URL with fallback for backward compatibility
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
        if base_url and not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid LLM_BASE_URL: '{base_url}'. "
                "Must start with http:// or https://"
            )

        # Auto-switch to litellm when a custom base URL is set (as documented in .env.example)
        llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
        if base_url and llm_provider == "anthropic" and not os.getenv("LLM_PROVIDER"):
            llm_provider = "litellm"

        config_dict = {
            "llm_provider": llm_provider,
            "model_name": os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022"),
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "api_base_url": base_url,

            # Provider-specific keys
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),

            "max_retries": _parse_int(os.getenv("LLM_MAX_RETRIES"), 3),
            "timeout": _parse_int(os.getenv("LLM_TIMEOUT"), 60),
            "max_tpm": _parse_int(os.getenv("MAX_TPM"), 30000),
            "tpm_safety_factor": _parse_float(os.getenv("TPM_SAFETY_FACTOR"), 0.85),
            "max_tokens_per_call": _parse_int(os.getenv("MAX_TOKENS_PER_CALL"), None) if os.getenv("MAX_TOKENS_PER_CALL") else None,
            "retry_max_attempts": _parse_int(os.getenv("RETRY_MAX_ATTEMPTS"), 3),
            "retry_initial_wait": _parse_float(os.getenv("RETRY_INITIAL_WAIT"), 2.0),
            "max_output_tokens": _parse_int(os.getenv("LLM_MAX_OUTPUT_TOKENS"), 16384),
            "max_input_tokens": _parse_int(os.getenv("LLM_MAX_INPUT_TOKENS"), 128_000) if os.getenv("LLM_MAX_INPUT_TOKENS") else 128_000,
            "max_tool_output_chars": _parse_int(os.getenv("MAX_TOOL_OUTPUT_CHARS"), 12000),
            "max_scan_files": _parse_int(os.getenv("MAX_SCAN_FILES"), 200),
            "max_file_size": _parse_int(os.getenv("MAX_FILE_SIZE"), 1_000_000),
            "ignored_dirs": ignored_dirs,
            "output_dir": Path(output_dir_env) if output_dir_env else Path("contexts"),
        }

        # Apply CLI overrides
        if cli_overrides:
            config_dict.update({k: v for k, v in cli_overrides.items() if v is not None})

        return cls(**config_dict)
