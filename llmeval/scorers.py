"""Scorers. Deterministic ones first (cheap, exact), LLM-as-judge for what rules can't check.

Every scorer returns Score(passed, value in [0,1], reason).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .suite import Case


@dataclass
class Score:
    name: str
    passed: bool
    value: float
    reason: str = ""


def _json(text: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _type_ok(v, t: str) -> bool:
    return {"string": isinstance(v, str), "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": isinstance(v, int) and not isinstance(v, bool), "boolean": isinstance(v, bool),
            "array": isinstance(v, list), "object": isinstance(v, dict), "null": v is None}.get(t, True)


def check_schema(v, schema: dict, path: str = "$") -> list[str]:
    """Small JSON-Schema subset: type (or list of types), required, properties, items, enum, minLength."""
    errs = []
    t = schema.get("type")
    if t and not any(_type_ok(v, x) for x in ([t] if isinstance(t, str) else t)):
        return [f"{path}: expected {t}, got {type(v).__name__}"]
    if "enum" in schema and v not in schema["enum"]:
        errs.append(f"{path}: {v!r} not in {schema['enum']}")
    if isinstance(v, str) and len(v) < schema.get("minLength", 0):
        errs.append(f"{path}: shorter than {schema['minLength']}")
    if isinstance(v, dict):
        for k in schema.get("required", []):
            if k not in v:
                errs.append(f"{path}.{k}: missing")
        for k, sub in schema.get("properties", {}).items():
            if k in v:
                errs += check_schema(v[k], sub, f"{path}.{k}")
    if isinstance(v, list) and "items" in schema:
        for i, item in enumerate(v):
            errs += check_schema(item, schema["items"], f"{path}[{i}]")
    return errs


def rule_score(spec: dict, case: Case, output: str) -> Score:
    kind = spec["type"]
    name = spec.get("name", kind)
    out = output.strip()
    expected = spec.get("value", case.expected)
    if kind == "exact":
        ok = out.lower() == str(expected).strip().lower()
        return Score(name, ok, float(ok), "" if ok else f"expected exactly {expected!r}")
    if kind == "contains":
        vals = expected if isinstance(expected, list) else [expected]
        missing = [v for v in vals if str(v).lower() not in out.lower()]
        return Score(name, not missing, 1 - len(missing) / len(vals), f"missing {missing}" if missing else "")
    if kind == "not_contains":
        vals = spec["value"] if isinstance(spec["value"], list) else [spec["value"]]
        found = [v for v in vals if str(v).lower() in out.lower()]
        return Score(name, not found, float(not found), f"must not mention {found}" if found else "")
    if kind == "regex":
        ok = re.search(spec["pattern"], out, re.I | re.S) is not None
        return Score(name, ok, float(ok), "" if ok else f"no match for /{spec['pattern']}/")
    if kind == "json":
        v = _json(out)
        if v is None:
            return Score(name, False, 0.0, "not valid JSON")
        errs = check_schema(v, spec["schema"]) if "schema" in spec else []
        return Score(name, not errs, float(not errs), "; ".join(errs[:5]))
    if kind == "length":
        n = len(out.split())
        lo, hi = spec.get("min_words", 0), spec.get("max_words", 10**9)
        ok = lo <= n <= hi
        return Score(name, ok, float(ok), "" if ok else f"{n} words, want {lo}-{hi}")
    if kind == "number":
        nums = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", out)]
        target, tol = float(expected), float(spec.get("tolerance", 0))
        ok = any(abs(x - target) <= tol for x in nums)
        return Score(name, ok, float(ok), "" if ok else f"no number within {tol} of {target} in {nums[:5]}")
    if kind == "refusal":
        refused = bool(re.search(r"\b(can't|cannot|won't|unable to|not able to|I'm sorry)\b", out, re.I))
        want = spec.get("expect", True)
        ok = refused == want
        return Score(name, ok, float(ok), "" if ok else ("should refuse" if want else "refused wrongly"))
    raise ValueError(f"unknown scorer {kind!r}")


# ---------------------------------------------------------------- LLM-as-judge
JUDGE_SYSTEM = """You are a strict evaluator. Grade the RESPONSE against the RUBRIC.
Think about each rubric point, then reply with ONLY JSON: {"reasoning": "<2-3 sentences>", "score": <1-5>}
5 = fully meets the rubric, 3 = partially, 1 = fails. Judge only what the rubric asks; ignore style otherwise.
Do not reward length."""

PAIRWISE_SYSTEM = """Compare two responses to the same input against the RUBRIC.
Reply with ONLY JSON: {"reasoning": "<2-3 sentences>", "winner": "A" | "B" | "tie"}. Do not reward length or position."""

JudgeFn = Callable[[list[dict]], str]  # messages -> raw model text


def judge_score(spec: dict, case: Case, output: str, judge: JudgeFn) -> Score:
    name = spec.get("name", "judge")
    parts = [f"RUBRIC:\n{spec['rubric']}", f"INPUT:\n{case.input if isinstance(case.input, str) else json.dumps(case.input)}"]
    if case.context:
        parts.append(f"CONTEXT (the response must be grounded in this):\n{case.context}")
    if case.expected is not None:
        parts.append(f"REFERENCE ANSWER:\n{case.expected}")
    parts.append(f"RESPONSE:\n{output}")
    raw = judge([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}])
    v = _json(raw)
    if not isinstance(v, dict) or "score" not in v:
        return Score(name, False, 0.0, f"judge returned unparseable output: {raw[:120]!r}")
    s = max(1, min(5, int(round(float(v["score"])))))
    min_pass = spec.get("min_score", 4)
    return Score(name, s >= min_pass, (s - 1) / 4, f"{s}/5: {v.get('reasoning', '')}")


def pairwise(case: Case, a: str, b: str, rubric: str, judge: JudgeFn) -> dict:
    """Judge A vs B twice with positions swapped; only a consistent verdict counts (cuts position bias)."""
    def ask(first, second):
        msg = f"RUBRIC:\n{rubric}\n\nINPUT:\n{case.input}\n\nRESPONSE A:\n{first}\n\nRESPONSE B:\n{second}"
        v = _json(judge([{"role": "system", "content": PAIRWISE_SYSTEM}, {"role": "user", "content": msg}])) or {}
        return v.get("winner", "tie")

    w1 = ask(a, b)
    w2 = {"A": "B", "B": "A"}.get(ask(b, a), "tie")  # map back to original labels
    winner = w1 if w1 == w2 else "tie"
    return {"winner": winner, "consistent": w1 == w2, "raw": [w1, w2]}


def cohens_kappa(a: list, b: list) -> float:
    """Agreement between judge labels and human labels beyond chance. >0.6 is good, >0.8 strong."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
