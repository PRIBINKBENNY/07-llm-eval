"""llmeval run datasets/support_bot.yaml --model openai/gpt-4o-mini --out runs/gpt4o-mini.json --html
llmeval compare runs/baseline.json runs/new.json        # exit 1 on regression (use in CI)
llmeval report runs/new.json --baseline runs/baseline.json
llmeval kappa labels.csv                                 # judge vs human agreement
llmeval traces traces.db                                 # latency / tokens / cost summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _load_dotenv():
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.split(" #")[0].strip())


def make_judge(model: str | None):
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL", os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")),
                    api_key=os.environ.get("JUDGE_API_KEY", os.environ.get("LLM_API_KEY", "not-needed")))
    model = model or os.environ.get("JUDGE_MODEL", "openai/gpt-4o")

    def judge(messages):
        r = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=300)
        return r.choices[0].message.content or ""

    return judge


def main(argv=None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="llmeval")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("suite")
    r.add_argument("--model", help="override the suite's target model")
    r.add_argument("--judge-model")
    r.add_argument("--out")
    r.add_argument("--html", action="store_true")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--cache", default=".llmeval_cache.db")
    r.add_argument("--traces", default="traces.db")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--max-drop", type=float, default=2.0)
    rp = sub.add_parser("report")
    rp.add_argument("run")
    rp.add_argument("--baseline")
    rp.add_argument("--out")
    k = sub.add_parser("kappa", help="CSV with columns judge,human")
    k.add_argument("csv")
    t = sub.add_parser("traces")
    t.add_argument("db", nargs="?", default="traces.db")
    a = p.parse_args(argv)

    from .report import write
    from .runner import Cache, RunResult, compare, run_suite

    if a.cmd == "run":
        from .suite import Suite
        from .targets import make_target
        from .tracing import Tracer

        suite = Suite.load(a.suite)
        spec = dict(suite.target)
        if a.model:
            spec["model"] = a.model
        target = make_target(spec)
        name = spec.get("model") or spec.get("url", "target")
        needs_judge = any(s["type"] == "judge" for cs in suite.cases for s in cs.scorers)
        judge = make_judge(a.judge_model or suite.judge.get("model")) if needs_judge else None
        res = run_suite(suite, target, judge, name, a.concurrency, cache=Cache(a.cache), tracer=Tracer(a.traces))
        out = a.out or f"runs/{suite.name}-{name.replace('/', '_')}.json"
        res.save(out)
        print(json.dumps(res.summary(), indent=2))
        if a.html:
            print("report:", write(res, Path(out).with_suffix(".html")))
        return 0 if res.summary()["passed_threshold"] else 1
    if a.cmd == "compare":
        result = compare(RunResult.load(a.base), RunResult.load(a.new), a.max_drop)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    if a.cmd == "report":
        run = RunResult.load(a.run)
        base = RunResult.load(a.baseline) if a.baseline else None
        print(write(run, a.out or Path(a.run).with_suffix(".html"), base))
        return 0
    if a.cmd == "kappa":
        from .scorers import cohens_kappa

        rows = list(csv.DictReader(open(a.csv)))
        kv = cohens_kappa([r["judge"] for r in rows], [r["human"] for r in rows])
        print(f"Cohen's kappa = {kv:.3f} over {len(rows)} labels "
              f"({'strong' if kv > 0.8 else 'good' if kv > 0.6 else 'weak: fix the rubric or the judge model'})")
        return 0
    if a.cmd == "traces":
        from .tracing import Tracer

        print(json.dumps(Tracer(a.db).summary(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
