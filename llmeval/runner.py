"""Run a suite against a target: concurrent, cached, retried, traced. Then compare runs."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .scorers import JudgeFn, Score, judge_score, rule_score
from .suite import Case, Suite
from .targets import Usage
from .tracing import Tracer, cost_usd


class Cache:
    """SQLite cache so re-running a suite doesn't re-pay for identical calls."""

    def __init__(self, path: str | None):
        self.path = path
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as c:
                c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, k: str):
        if not self.path:
            return None
        with sqlite3.connect(self.path) as c:
            row = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v) -> None:
        if self.path:
            with sqlite3.connect(self.path) as c:
                c.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, json.dumps(v)))


@dataclass
class CaseResult:
    id: str
    tags: list[str]
    input: object
    expected: object
    output: str
    scores: list[dict]
    passed: bool
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    error: str | None = None
    cached: bool = False


@dataclass
class RunResult:
    suite: str
    target: str
    started: str
    results: list[CaseResult] = field(default_factory=list)
    threshold: float = 0.8

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def summary(self) -> dict:
        lat = sorted(r.latency_ms for r in self.results if not r.cached) or [0.0]
        by_scorer: dict[str, list[bool]] = {}
        by_tag: dict[str, list[bool]] = {}
        for r in self.results:
            for s in r.scores:
                by_scorer.setdefault(s["name"], []).append(s["passed"])
            for t in r.tags or ["untagged"]:
                by_tag.setdefault(t, []).append(r.passed)
        rate = lambda xs: round(100 * sum(xs) / len(xs), 1)  # noqa: E731
        costs = [r.cost_usd for r in self.results if r.cost_usd is not None]
        return {
            "suite": self.suite, "target": self.target, "cases": len(self.results),
            "pass_rate": round(100 * self.pass_rate, 1), "passed_threshold": self.pass_rate >= self.threshold,
            "errors": sum(r.error is not None for r in self.results),
            "by_scorer": {k: rate(v) for k, v in by_scorer.items()},
            "by_tag": {k: rate(v) for k, v in by_tag.items()},
            "latency_ms": {"p50": round(lat[len(lat) // 2], 1), "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1)},
            "tokens": {"in": sum(r.tokens_in for r in self.results), "out": sum(r.tokens_out for r in self.results)},
            "cost_usd": round(sum(costs), 6) if costs else None,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"suite": self.suite, "target": self.target, "started": self.started,
                                          "threshold": self.threshold, "summary": self.summary(),
                                          "results": [asdict(r) for r in self.results]}, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        d = json.loads(Path(path).read_text())
        return cls(d["suite"], d["target"], d["started"], [CaseResult(**r) for r in d["results"]], d.get("threshold", 0.8))


def run_suite(suite: Suite, target, judge: JudgeFn | None = None, target_name: str = "target",
              concurrency: int = 8, retries: int = 2, cache: Cache | None = None, tracer: Tracer | None = None
              ) -> RunResult:
    cache = cache or Cache(None)

    def one(case: Case) -> CaseResult:
        key = Cache.key(target_name, case.input, case.context)
        hit = cache.get(key)
        t0 = time.perf_counter()
        output, usage, error, cached = "", Usage(), None, False
        if hit is not None:
            output, usage, cached = hit["output"], Usage(**hit["usage"]), True
        else:
            for attempt in range(retries + 1):
                try:
                    if tracer:
                        with tracer.span("eval_case", case=case.id, suite=suite.name) as sp:
                            output, usage = target(case)
                            if usage.model:
                                sp.record_usage(usage.model, usage.tokens_in, usage.tokens_out)
                    else:
                        output, usage = target(case)
                    error = None
                    cache.put(key, {"output": output, "usage": asdict(usage)})
                    break
                except Exception as e:  # rate limits, timeouts: retry with backoff, then record
                    error = f"{type(e).__name__}: {e}"
                    time.sleep(min(8, 0.5 * 2 ** attempt))
        latency = (time.perf_counter() - t0) * 1000

        scores: list[Score] = []
        if error:
            scores = [Score("error", False, 0.0, error)]
        else:
            for spec in case.scorers:
                if spec["type"] == "judge":
                    if judge is None:
                        scores.append(Score(spec.get("name", "judge"), False, 0.0, "no judge configured"))
                    else:
                        try:
                            scores.append(judge_score(spec, case, output, judge))
                        except Exception as e:
                            scores.append(Score(spec.get("name", "judge"), False, 0.0, f"judge error: {e}"))
                else:
                    scores.append(rule_score(spec, case, output))
        return CaseResult(case.id, case.tags, case.input, case.expected, output, [asdict(s) for s in scores],
                          all(s.passed for s in scores), round(latency, 1), usage.tokens_in, usage.tokens_out,
                          cost_usd(usage.model, usage.tokens_in, usage.tokens_out) if usage.model else None,
                          error, cached)

    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(one, suite.cases))
    return RunResult(suite.name, target_name, time.strftime("%Y-%m-%d %H:%M:%S"), results, suite.threshold)


def compare(base: RunResult, new: RunResult, max_drop_pts: float = 2.0) -> dict:
    """Regression gate: fails if the pass rate drops by more than `max_drop_pts` percentage points,
    or if any case that passed in the baseline now fails."""
    b = {r.id: r for r in base.results}
    n = {r.id: r for r in new.results}
    regressions = sorted(i for i in b if i in n and b[i].passed and not n[i].passed)
    fixed = sorted(i for i in b if i in n and not b[i].passed and n[i].passed)
    drop = 100 * (base.pass_rate - new.pass_rate)
    ok = drop <= max_drop_pts and not regressions
    return {"ok": ok, "base_pass_rate": round(100 * base.pass_rate, 1), "new_pass_rate": round(100 * new.pass_rate, 1),
            "delta_pts": round(-drop, 1), "regressions": regressions, "fixed": fixed,
            "new_cases": sorted(set(n) - set(b)), "removed_cases": sorted(set(b) - set(n))}
