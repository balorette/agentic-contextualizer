#!/usr/bin/env python
"""Test script to debug agent mode configuration."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.config import Config
from agents.llm.provider import _resolve_api_key_for_model

# Load config
config = Config.from_env()

print("=" * 60)
print("Configuration Debug")
print("=" * 60)
print(f"Provider: {config.llm_provider}")
print(f"Model: {config.model_name}")
print(f"Base URL: {config.api_base_url}")
print(f"OpenAI Key: {config.openai_api_key[:10]}..." if config.openai_api_key else "OpenAI Key: None")
print(f"Anthropic Key: {config.anthropic_api_key[:10]}..." if config.anthropic_api_key else "Anthropic Key: None")
print()

# Test key resolution
resolved_key = _resolve_api_key_for_model(config.model_name, config)
print(f"Resolved API Key for {config.model_name}: {resolved_key[:10]}..." if resolved_key else "None")
print()

# Check environment variables
print("=" * 60)
print("Environment Variables")
print("=" * 60)
print(f"LLM_BASE_URL: {os.getenv('LLM_BASE_URL', 'Not set')}")
print(f"ANTHROPIC_BASE_URL: {os.getenv('ANTHROPIC_BASE_URL', 'Not set')}")
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'Not set')}")
print(f"MODEL_NAME: {os.getenv('MODEL_NAME', 'Not set')}")
print()

# Test LiteLLM directly
print("=" * 60)
print("Testing LiteLLM Direct Call")
print("=" * 60)

try:
    import litellm

    # Enable debug
    litellm.set_verbose = True
    os.environ["LITELLM_LOG"] = "DEBUG"

    print(f"Calling litellm.completion with:")
    print(f"  model: {config.model_name}")
    print(f"  api_base: {config.api_base_url}")
    print(f"  api_key: {resolved_key[:10]}..." if resolved_key else "None")
    print()

    response = litellm.completion(
        model=config.model_name,
        messages=[{"role": "user", "content": "Say 'hello' in one word"}],
        api_base=config.api_base_url,
        api_key=resolved_key,
        timeout=30,
    )

    print("✅ Success!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"❌ Failed: {e}")
    print()
    print("Debug info:")
    print(f"  Error type: {type(e).__name__}")
    print(f"  Error message: {str(e)}")

    # Try to extract more details
    if hasattr(e, '__dict__'):
        print("  Error details:")
        for key, value in e.__dict__.items():
            print(f"    {key}: {value}")
