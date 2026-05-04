"""TritonAI/OpenAI-compatible generation plus a deterministic fallback.

The primary path intentionally mirrors the TritonAI sample:

    client = openai.OpenAI(api_key=..., base_url="https://tritonai-api.ucsd.edu")
    client.chat.completions.create(model="api-gpt-oss-120b", messages=[...])

If the `openai` package is unavailable, the code falls back to a standard-library
HTTP request against the same chat-completions endpoint.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .text_utils import split_sentences, tokenize


@dataclass
class LLMConfig:
    model: str = "api-gpt-oss-120b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 420
    timeout: int = 60
    retries: int = 5
    disabled: bool = False
    fail_on_error: bool = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dotenv() -> Dict[str, str]:
    """Load local `.env` values without requiring python-dotenv.

    This lets the project run the way the user wants on DataHub with a local
    `.env` file, while `.gitignore` prevents accidentally submitting secrets.
    """
    values: Dict[str, str] = {}
    for path in (Path.cwd() / ".env", _project_root() / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_config_value(*names: str) -> Optional[str]:
    env_file = _load_dotenv()
    for name in names:
        value = os.environ.get(name) or env_file.get(name)
        if value:
            return value.strip()
    return None


def discover_api_key(explicit: str | None = None) -> Optional[str]:
    if explicit:
        return explicit
    value = get_config_value("GENERATOR_API_KEY", "OPENAI_API_KEY", "TRITONAI_API_KEY")
    if value:
        return value
    key_file = Path.home() / "api-key.txt"
    if key_file.exists():
        text = key_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    return None


def discover_base_url(explicit: str | None = None) -> Optional[str]:
    if explicit:
        return explicit.rstrip("/")
    value = get_config_value("GENERATOR_BASE_URL", "OPENAI_BASE_URL", "JUDGE_BASE_URL")
    if value:
        return value.rstrip("/")
    # UCSD project environment commonly provides ~/api-key.txt for TritonAI.
    if (Path.home() / "api-key.txt").exists():
        return "https://tritonai-api.ucsd.edu"
    return None


def _chat_endpoint(base_url: Optional[str]) -> str:
    if not base_url:
        return "https://api.openai.com/v1/chat/completions"
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def call_with_openai_package(messages: List[Dict[str, str]], config: LLMConfig) -> str:
    """Call the API using the same OpenAI client style as TritonAI's example."""
    import openai

    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.base_url or "https://tritonai-api.ucsd.edu",
        timeout=config.timeout,
        max_retries=config.retries,
    )
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("API returned message.content=None")
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()


def call_with_urllib(messages: List[Dict[str, str]], config: LLMConfig) -> str:
    """Standard-library fallback for OpenAI-compatible chat completions."""
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _chat_endpoint(config.base_url),
        data=data,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=config.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    if content is None:
        raise RuntimeError("API returned message.content=None")
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()


def call_openai_compatible(messages: List[Dict[str, str]], config: LLMConfig) -> str:
    if config.disabled:
        raise RuntimeError("LLM calls disabled")
    if not config.api_key:
        raise RuntimeError("No API key configured")

    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        try:
            try:
                return call_with_openai_package(messages, config)
            except ImportError:
                return call_with_urllib(messages, config)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            last_error = RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}")
            if attempt < config.retries:
                time.sleep(min(20.0, 2.0 * (attempt + 1)))
        except (urllib.error.URLError, KeyError, IndexError, AttributeError, json.JSONDecodeError, RuntimeError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt < config.retries:
                time.sleep(min(20.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"LLM request failed: {last_error}")


def make_messages(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "You answer technical documentation questions using only the retrieved context. "
        "Be precise, concise, and complete. If the context does not support a claim, do not include it. "
        "Do not mention hidden tests or evaluation rubrics."
    )
    user = (
        "Retrieved context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer from the context. Include important conditions, parameter names, and distinctions."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extractive_fallback_answer(question: str, context: str, *, max_sentences: int = 6) -> str:
    """A deterministic answer used when no API key/model is available.

    It is deliberately conservative: it quotes/summarizes the most relevant
    context lines instead of inventing unsupported claims.
    """
    q_terms = set(tokenize(question))
    if not context.strip():
        return "The retrieved context is empty, so the answer cannot be determined from the documentation."

    # Remove line labels for more natural fallback wording.
    cleaned_lines = []
    for line in context.splitlines():
        line = re.sub(r"^\[Source:[^\]]+\]\s*$", "", line).strip()
        line = re.sub(r"^L\d+:\s*", "", line).strip()
        if line:
            cleaned_lines.append(line)
    sentences = []
    for line in cleaned_lines:
        sentences.extend(split_sentences(line))

    scored = []
    for idx, sent in enumerate(sentences):
        terms = set(tokenize(sent))
        overlap = len(q_terms & terms)
        api_bonus = 1 if re.search(r"`|:func:|:class:|[A-Z]{2,}[A-Za-z0-9_]*", sent) else 0
        if overlap or api_bonus:
            scored.append((overlap * 3 + api_bonus - idx * 0.01, sent))

    if not scored:
        selected = sentences[: max_sentences]
    else:
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [sent for _, sent in scored[:max_sentences]]

    # Restore original-ish order.
    order = {sent: i for i, sent in enumerate(sentences)}
    selected.sort(key=lambda s: order.get(s, 0))
    answer = " ".join(selected).strip()
    if not answer:
        answer = "The retrieved context does not contain enough information to answer confidently."
    return answer


def generate_answer(question: str, context: str, config: LLMConfig) -> str:
    if not config.disabled and config.api_key:
        try:
            answer = call_openai_compatible(make_messages(question, context), config)
            if answer:
                return answer
        except Exception as exc:
            if config.fail_on_error:
                raise
            print(f"[warn] generation API failed; using extractive fallback: {exc}", file=sys.stderr)
            return (
                f"{extractive_fallback_answer(question, context)} "
                f"(Generation API was unavailable, so this answer used the local extractive fallback.)"
            )
    return extractive_fallback_answer(question, context)
