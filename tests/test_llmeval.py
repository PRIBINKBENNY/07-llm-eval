import json
from pathlib import Path

import pytest

from llmeval import Cache, RunResult, Suite, Tracer, cohens_kappa, compare, pairwise, rule_score, run_suite
from llmeval.scorers import check_schema, judge_score
from llmeval.suite import Case
from llmeval.targets import FunctionTarget, OpenAITarget, Usage

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "datasets" / "support_bot.yaml"

GOOD = {
    "return-window": "You can return it within 30 days with your receipt.",
    "warranty": "Yes. The warranty lasts 18 months, so you're still covered.",
    "hours": "Sorry, support isn't open on Sunday. We're available Mon-Fri 9:00-18:00 IST.",
    "safety-escalation": "Stop using the arm and cut the power now. Then email safety@orion.example.",
    "privacy": "Sorry, I can't share other customers' information.",
    "json-ticket": '{"customer": "Asha", "issue": "gripper error after firmware update", "error_code": "E22", "priority": "high"}',
    "maths": "That's 4,350 rupees in total.",
    "grounded": "E14 means the encoder lost its zero position. Re-home the arm.",
}


def fake_judge(messages):
    """Deterministic stand-in: 5 if the response shares key words with the rubric, else 2."""
    text = messages[-1]["content"]
    rubric = text.split("RUBRIC:\n")[1].split("\n\n")[0].lower()
    resp = text.split("RESPONSE:\n")[-1].lower()
    keys = [w for w in ["18 months", "sunday", "safety@orion.example", "encoder", "re-home"] if w in rubric]
    return json.dumps({"reasoning": "checked", "score": 5 if all(k in resp for k in keys) else 2})


def test_suite_loads_with_defaults():
    s = Suite.load(SUITE)
    assert s.name == "support-bot" and len(s.cases) == 8 and s.threshold == 0.85
    by = {c.id: c for c in s.cases}
    assert any(sc["type"] == "length" for sc in by["warranty"].scorers)        # default appended
    assert not any(sc["type"] == "length" for sc in by["json-ticket"].scorers)  # opted out


def test_suite_rejects_duplicates_and_empty_scorers(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("cases:\n - {id: a, input: x, scorers: [{type: exact, value: x}]}\n - {id: a, input: y, scorers: [{type: exact, value: y}]}\n")
    with pytest.raises(ValueError, match="duplicate"):
        Suite.load(p)
    p.write_text("cases:\n - {id: a, input: x}\n")
    with pytest.raises(ValueError, match="no scorers"):
        Suite.load(p)


def test_rule_scorers():
    c = Case("x", "q", expected="30 days")
    assert rule_score({"type": "contains"}, c, "Within 30 Days.").passed
    assert not rule_score({"type": "contains", "value": ["30 days", "receipt"]}, c, "30 days").passed
    assert rule_score({"type": "regex", "pattern": r"\b\d{2} days\b"}, c, "in 30 days").passed
    assert rule_score({"type": "number", "value": 4350}, c, "total: 4,350.00").passed
    assert not rule_score({"type": "number", "value": 4350, "tolerance": 1}, c, "4352").passed
    assert rule_score({"type": "length", "max_words": 3}, c, "one two three").passed
    assert rule_score({"type": "refusal"}, c, "Sorry, I can't do that.").passed
    assert not rule_score({"type": "refusal"}, c, "Sure! It's #4411").passed
    assert rule_score({"type": "json"}, c, '```json\n{"a": 1}\n```').passed
    assert not rule_score({"type": "json"}, c, "no json here").passed


def test_json_schema_subset():
    schema = {"type": "object", "required": ["a", "b"], "properties": {
        "a": {"type": "string", "enum": ["x", "y"]}, "b": {"type": "array", "items": {"type": "integer"}}}}
    assert check_schema({"a": "x", "b": [1, 2]}, schema) == []
    errs = check_schema({"a": "z", "b": [1, "2"]}, schema)
    assert any("not in" in e for e in errs) and any("$.b[1]" in e for e in errs)
    assert check_schema({"a": "x"}, schema) == ["$.b: missing"]


def test_judge_parsing_and_failure_modes():
    c = Case("x", "q", context="ctx")
    spec = {"type": "judge", "rubric": "be right", "min_score": 4}
    s = judge_score(spec, c, "answer", lambda m: 'Sure: {"reasoning": "fine", "score": 4}')
    assert s.passed and s.value == 0.75 and "4/5" in s.reason
    assert not judge_score(spec, c, "answer", lambda m: '{"score": 2}').passed
    bad = judge_score(spec, c, "answer", lambda m: "I refuse to grade")
    assert not bad.passed and "unparseable" in bad.reason
    # the judge sees context and the response
    seen = {}
    judge_score(spec, c, "the answer", lambda m: seen.setdefault("m", m) and '{"score": 5}')
    assert "CONTEXT" in seen["m"][-1]["content"] and "the answer" in seen["m"][-1]["content"]


def test_pairwise_swaps_positions_to_cancel_bias():
    c = Case("x", "q")
    always_a = lambda m: '{"winner": "A"}'  # noqa: E731 - a judge with pure position bias
    r = pairwise(c, "resp1", "resp2", "rubric", always_a)
    assert r["winner"] == "tie" and not r["consistent"]

    def prefers_long(m):
        t = m[-1]["content"]
        a = t.split("RESPONSE A:\n")[1].split("\n\nRESPONSE B:")[0]
        b = t.split("RESPONSE B:\n")[1]
        return json.dumps({"winner": "A" if len(a) > len(b) else "B"})

    r = pairwise(c, "short", "a much longer answer", "rubric", prefers_long)
    assert r["winner"] == "B" and r["consistent"]


def test_kappa():
    assert cohens_kappa(["p", "f", "p", "f"], ["p", "f", "p", "f"]) == 1.0
    assert cohens_kappa(["p", "p", "f", "f"], ["p", "f", "p", "f"]) == 0.0
    assert 0 < cohens_kappa(list("ppppffff"), list("pppfffff")) < 1


def run_with(answers, tmp_path, **kw):
    suite = Suite.load(SUITE)
    target = FunctionTarget(lambda case: answers.get(case.id, "I don't know."))
    return run_suite(suite, target, fake_judge, "fake", cache=Cache(str(tmp_path / "c.db")), **kw)


def test_full_run_good_answers_pass(tmp_path):
    res = run_with(GOOD, tmp_path)
    failed = [(r.id, r.scores) for r in res.results if not r.passed]
    assert not failed, failed
    s = res.summary()
    assert s["pass_rate"] == 100.0 and s["passed_threshold"] and s["by_tag"]["safety"] == 100.0


def test_regression_gate_and_report(tmp_path):
    base = run_with(GOOD, tmp_path / "a")
    worse = dict(GOOD, privacy="Sure, his order number is #4411.", maths="About 4,000.")
    new = run_with(worse, tmp_path / "b")
    cmp = compare(base, new)
    assert not cmp["ok"] and cmp["regressions"] == ["maths", "privacy"] and cmp["delta_pts"] == -25.0
    assert compare(base, base)["ok"]
    base.save(tmp_path / "base.json")
    again = RunResult.load(tmp_path / "base.json")
    assert again.pass_rate == base.pass_rate
    from llmeval.report import render

    page = render(new, base)
    assert "Regression gate: FAILED" in page and "⚠ regression" in page and "#4411" in page
    assert "<script" not in page  # outputs are escaped, never executed
    evil = run_with(dict(GOOD, maths="<script>alert(1)</script>"), tmp_path / "c")
    assert "<script>alert" not in render(evil)


def test_cache_avoids_repeat_calls_and_errors_are_retried(tmp_path):
    calls = {"n": 0}

    def flaky(case):
        calls["n"] += 1
        if calls["n"] % 3 == 1:
            raise TimeoutError("slow")
        return GOOD[case.id]

    suite = Suite.load(SUITE)
    cache = Cache(str(tmp_path / "c.db"))
    r1 = run_suite(suite, FunctionTarget(flaky), fake_judge, "flaky", concurrency=1, retries=2, cache=cache)
    assert all(r.error is None for r in r1.results)
    n = calls["n"]
    r2 = run_suite(suite, FunctionTarget(flaky), fake_judge, "flaky", cache=cache)
    assert calls["n"] == n and all(r.cached for r in r2.results)


def test_tracing_records_tokens_cost_nesting_and_errors(tmp_path):
    t = Tracer(str(tmp_path / "t.db"))
    with t.span("request", user="u1") as outer:
        with t.span("llm_call") as s:
            s.record_usage("openai/gpt-4o-mini", 1000, 200)
        outer.set(route="faq")
    with pytest.raises(ValueError):
        with t.span("broken"):
            raise ValueError("boom")
    rows = {r["name"]: r for r in t.rows()}
    assert rows["llm_call"]["parent_id"] == rows["request"]["id"]
    assert rows["llm_call"]["trace_id"] == rows["request"]["trace_id"]
    assert rows["llm_call"]["cost_usd"] == pytest.approx((1000 * 0.15 + 200 * 0.60) / 1e6)
    assert rows["broken"]["status"] == "error" and "boom" in rows["broken"]["error"]
    s = t.summary()
    assert s["spans"] == 3 and s["errors"] == 1 and s["tokens_in"] == 1000


def test_openai_target_builds_messages_and_reads_usage():
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    FakeClient.kw = kw

                    class R:
                        choices = [type("C", (), {"message": type("M", (), {"content": "hi"})()})()]
                        usage = type("U", (), {"prompt_tokens": 12, "completion_tokens": 3})()
                    return R()

    tgt = OpenAITarget("m", system="be brief", client=FakeClient)
    out, usage = tgt(Case("x", "What is E14?", context="E14: encoder"))
    assert out == "hi" and usage == Usage("m", 12, 3)
    msgs = FakeClient.kw["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"} and "Context:\nE14: encoder" in msgs[1]["content"]


def test_cli_compare_exit_codes(tmp_path):
    from llmeval.cli import main

    base = run_with(GOOD, tmp_path / "a")
    new = run_with(dict(GOOD, maths="no idea"), tmp_path / "b")
    base.save(tmp_path / "base.json")
    new.save(tmp_path / "new.json")
    assert main(["compare", str(tmp_path / "base.json"), str(tmp_path / "base.json")]) == 0
    assert main(["compare", str(tmp_path / "base.json"), str(tmp_path / "new.json")]) == 1
    assert main(["report", str(tmp_path / "new.json"), "--baseline", str(tmp_path / "base.json"),
                 "--out", str(tmp_path / "r.html")]) == 0
    assert (tmp_path / "r.html").read_text().startswith("<!DOCTYPE html>")
