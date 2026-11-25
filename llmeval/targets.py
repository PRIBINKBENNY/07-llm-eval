"""What gets evaluated. Every target returns (output_text, usage) for a case."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .suite import Case


@dataclass
class Usage:
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


class OpenAITarget:
    """Any OpenAI-compatible chat model (OpenRouter, OpenAI, vLLM from 03-deploy) plus a system prompt."""

    def __init__(self, model: str, system: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 512, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
                            api_key=api_key or os.environ.get("LLM_API_KEY", "not-needed"))
        self.client, self.model, self.system = client, model, system
        self.temperature, self.max_tokens = temperature, max_tokens

    def messages(self, case: Case) -> list[dict]:
        if isinstance(case.input, list):
            msgs = list(case.input)
        else:
            content = case.input if not case.context else f"Context:\n{case.context}\n\nQuestion: {case.input}"
            msgs = [{"role": "user", "content": content}]
        if self.system and not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": self.system}] + msgs
        return msgs

    def __call__(self, case: Case) -> tuple[str, Usage]:
        r = self.client.chat.completions.create(model=self.model, messages=self.messages(case),
                                                temperature=self.temperature, max_tokens=self.max_tokens)
        u = r.usage
        return r.choices[0].message.content or "", Usage(self.model, getattr(u, "prompt_tokens", 0) or 0,
                                                          getattr(u, "completion_tokens", 0) or 0)


class HTTPTarget:
    """Your own app behind HTTP, e.g. POST {"question": ...} -> {"answer": ...}."""

    def __init__(self, url: str, input_key: str = "question", output_key: str = "answer", timeout: float = 60):
        self.url, self.input_key, self.output_key, self.timeout = url, input_key, output_key, timeout

    def __call__(self, case: Case) -> tuple[str, Usage]:
        import httpx

        r = httpx.post(self.url, json={self.input_key: case.input}, timeout=self.timeout)
        r.raise_for_status()
        return str(r.json()[self.output_key]), Usage()


class FunctionTarget:
    def __init__(self, fn: Callable[[Case], str]):
        self.fn = fn

    def __call__(self, case: Case) -> tuple[str, Usage]:
        return self.fn(case), Usage()


def make_target(spec: dict):
    kind = spec.get("type", "openai")
    if kind == "openai":
        return OpenAITarget(spec.get("model", os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")), spec.get("system"),
                            spec.get("base_url"), temperature=spec.get("temperature", 0.0))
    if kind == "http":
        return HTTPTarget(spec["url"], spec.get("input_key", "question"), spec.get("output_key", "answer"))
    raise ValueError(f"unknown target type {kind!r}")
