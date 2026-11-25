"""Observability for any LLM app: wrap calls, record latency, tokens, cost and errors to SQLite.

    from llmeval.tracing import Tracer
    tracer = Tracer("traces.db")
    with tracer.span("answer_question", user="u1") as s:
        resp = client.chat.completions.create(...)
        s.record_usage(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

# $ per 1M tokens (input, output). Edit for your models; unknown models cost 0 and are flagged.
PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return (tokens_in * p[0] + tokens_out * p[1]) / 1_000_000


SCHEMA = """CREATE TABLE IF NOT EXISTS spans (
  id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, name TEXT, start REAL, end REAL, latency_ms REAL,
  model TEXT, tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, status TEXT, error TEXT, attrs TEXT)"""


class Span:
    def __init__(self, tracer: "Tracer", name: str, trace_id: str, parent_id: str | None, attrs: dict):
        self.tracer, self.name, self.trace_id, self.parent_id = tracer, name, trace_id, parent_id
        self.id = uuid.uuid4().hex[:16]
        self.attrs = dict(attrs)
        self.model, self.tokens_in, self.tokens_out, self.cost = None, 0, 0, None
        self.status, self.error = "ok", None
        self.start = time.time()

    def record_usage(self, model: str, tokens_in: int, tokens_out: int) -> None:
        self.model = model
        self.tokens_in += tokens_in or 0
        self.tokens_out += tokens_out or 0
        c = cost_usd(model, self.tokens_in, self.tokens_out)
        self.cost = c
        if c is None:
            self.attrs["unpriced_model"] = True

    def set(self, **attrs) -> None:
        self.attrs.update(attrs)


class Tracer:
    def __init__(self, path: str = "traces.db"):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    @contextmanager
    def span(self, name: str, **attrs):
        stack = getattr(self._local, "stack", [])
        parent = stack[-1] if stack else None
        s = Span(self, name, parent.trace_id if parent else uuid.uuid4().hex[:16], parent.id if parent else None, attrs)
        stack.append(s)
        self._local.stack = stack
        try:
            yield s
        except Exception as e:
            s.status, s.error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            stack.pop()
            end = time.time()
            with self._lock, self._conn() as c:
                c.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (s.id, s.trace_id, s.parent_id, s.name, s.start, end, (end - s.start) * 1000, s.model,
                           s.tokens_in, s.tokens_out, s.cost, s.status, s.error, json.dumps(s.attrs, default=str)))

    def rows(self, where: str = "1=1", params: tuple = ()) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(f"SELECT * FROM spans WHERE {where} ORDER BY start", params)]

    def summary(self) -> dict:
        with self._conn() as c:
            n, err, cost, tin, tout = c.execute(
                "SELECT COUNT(*), SUM(status='error'), COALESCE(SUM(cost_usd),0), COALESCE(SUM(tokens_in),0), "
                "COALESCE(SUM(tokens_out),0) FROM spans").fetchone()
            lat = [r[0] for r in c.execute("SELECT latency_ms FROM spans ORDER BY latency_ms")]
        q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None  # noqa: E731
        return {"spans": n, "errors": err or 0, "cost_usd": round(cost, 6), "tokens_in": tin, "tokens_out": tout,
                "latency_ms": {"p50": q(0.5), "p95": q(0.95)}}
