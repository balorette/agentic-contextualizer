"""Configuration management for the Agentic Contextualizer."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


DEFAULT_IGNORED_DIRS = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
]


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

    # Scanner Settings
    max_file_size: int = Field(default=1_000_000)  # 1MB
    ignored_dirs: list[str] = Field(default_factory=lambda: DEFAULT_IGNORED_DIRS.copy())

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

        ignored_dirs = DEFAULT_IGNORED_DIRS.copy()
        extra_ignored = os.getenv("IGNORED_DIRS")
        if extra_ignored:
            ignored_dirs.extend(
                [entry.strip() for entry in extra_ignored.split(",") if entry.strip()]
            )

        output_dir_env = os.getenv("OUTPUT_DIR")

        config_dict = {
            "llm_provider": os.getenv("LLM_PROVIDER", "anthropic"),
            "model_name": os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022"),
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "api_base_url": os.getenv("ANTHROPIC_BASE_URL"),

            # Provider-specific keys
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),

            "max_retries": _parse_int(os.getenv("LLM_MAX_RETRIES"), 3),
            "timeout": _parse_int(os.getenv("LLM_TIMEOUT"), 60),
            "max_file_size": _parse_int(os.getenv("MAX_FILE_SIZE"), 1_000_000),
            "ignored_dirs": ignored_dirs,
            "output_dir": Path(output_dir_env) if output_dir_env else Path("contexts"),
        }

        # Apply CLI overrides
        if cli_overrides:
            config_dict.update({k: v for k, v in cli_overrides.items() if v is not None})

        return cls(**config_dict)
