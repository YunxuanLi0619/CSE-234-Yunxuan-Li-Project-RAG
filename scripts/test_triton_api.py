#!/usr/bin/env python3
"""Tiny TritonAI connectivity test for the Project 1 solution.

Usage:
    export OPENAI_API_KEY="$(cat ~/api-key.txt)"
    export GENERATOR_BASE_URL="https://tritonai-api.ucsd.edu"
    export GENERATOR_MODEL="api-llama-4-scout"
    python3 scripts/test_triton_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import LLMConfig, call_openai_compatible, discover_api_key, discover_base_url, get_config_value


def main() -> int:
    config = LLMConfig(
        model=get_config_value("GENERATOR_MODEL", "TRITONAI_MODEL", "OPENAI_MODEL") or "api-llama-4-scout",
        api_key=discover_api_key(),
        base_url=discover_base_url(),
        max_tokens=80,
        timeout=60,
    )
    if not config.api_key:
        print("No API key found. Set OPENAI_API_KEY or put the key in ~/api-key.txt.", file=sys.stderr)
        return 1
    print(f"Calling model={config.model} base_url={config.base_url or 'OpenAI default'}")
    answer = call_openai_compatible(
        [{"role": "user", "content": "Reply with exactly: TritonAI API is working."}],
        config,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
