"""Eval suites: a YAML file of cases, each with an input, optional expected output, tags and scorers.

name: support-bot
target: {type: openai, model: openai/gpt-4o-mini, system: "You are a helpful support agent."}
defaults: {scorers: [{type: judge, rubric: "Correct, polite and concise."}]}
cases:
  - id: refund-window
    input: "How long do I have to return an item?"
    expected: "30 days"
    tags: [policy]
    scorers: [{type: contains, value: "30 days"}]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Case:
    id: str
    input: str | list[dict]
    expected: object = None
    tags: list[str] = field(default_factory=list)
    scorers: list[dict] = field(default_factory=list)
    context: str | None = None  # e.g. retrieved docs, for groundedness rubrics
    meta: dict = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    cases: list[Case]
    target: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)
    threshold: float = 0.8  # pass rate needed for the suite to pass

    @classmethod
    def load(cls, path: str | Path) -> "Suite":
        raw = yaml.safe_load(Path(path).read_text())
        default_scorers = (raw.get("defaults") or {}).get("scorers", [])
        cases, seen = [], set()
        for i, c in enumerate(raw["cases"]):
            cid = str(c.get("id", f"case-{i + 1}"))
            if cid in seen:
                raise ValueError(f"duplicate case id {cid!r}")
            seen.add(cid)
            scorers = c.get("scorers", []) + ([] if c.get("no_defaults") else default_scorers)
            if not scorers:
                raise ValueError(f"case {cid!r} has no scorers")
            cases.append(Case(cid, c["input"], c.get("expected"), c.get("tags", []), scorers,
                              c.get("context"), c.get("meta", {})))
        return cls(raw.get("name", Path(path).stem), cases, raw.get("target", {}), raw.get("judge", {}),
                   raw.get("threshold", 0.8))
