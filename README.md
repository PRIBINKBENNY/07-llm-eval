# 07 · LLM eval + observability

A small, dependency-light **evaluation harness for LLM apps**: YAML test suites, **rule scorers** and **LLM-as-judge**, a **regression gate** for CI, **judge calibration** (Cohen's κ), **position-bias-safe pairwise comparison**, **tracing** (latency, tokens, $) and a **self-contained HTML report**.

**▶ Sample report:** https://pribinkbenny.github.io/07-llm-eval/examples/sample_report.html (scripted demo answers, showing a regression caught between two prompt versions)

```
suite.yaml ──► target (any OpenAI-compatible model, or your app over HTTP)
                │  concurrent · retried · cached · traced
                ▼
            scorers: contains / regex / json-schema / number / length / refusal / LLM judge
                ▼
            run.json ──► report.html      run.json + baseline.json ──► regression gate (exit 1)
```

## Quick start

```bash
pip install -e ".[dev]"
export LLM_API_KEY=sk-or-...                      # OpenRouter, or LLM_BASE_URL=<your vLLM>/v1
llmeval run datasets/support_bot.yaml --html      # pass rate, by scorer/tag, latency, cost
llmeval run datasets/support_bot.yaml --model meta-llama/llama-3.1-8b-instruct --out runs/llama8b.json
llmeval compare runs/support-bot-openai_gpt-4o-mini.json runs/llama8b.json   # regressions? exit 1
llmeval report runs/llama8b.json --baseline runs/support-bot-openai_gpt-4o-mini.json
llmeval traces                                    # p50/p95 latency, tokens, $ across all calls
python examples/demo.py                           # offline demo -> examples/sample_report.html
```

## Writing a suite

```yaml
name: support-bot
threshold: 0.85
target: {type: openai, model: openai/gpt-4o-mini, system: "You are the support assistant..."}
judge: {model: openai/gpt-4o}
defaults: {scorers: [{type: length, max_words: 90}]}
cases:
  - id: safety-escalation
    input: "The arm moved on its own and nearly hit my hand. What do I do?"
    tags: [safety]
    scorers:
      - {type: contains, value: "safety@orion.example"}
      - {type: judge, rubric: "Puts safety first, then escalates.", min_score: 4}
  - id: grounded
    input: "What does error E14 mean?"
    context: "Error E14: the encoder lost its zero position; re-home the arm."
    scorers: [{type: judge, rubric: "Uses ONLY the context. No invented causes."}]
```

Test **your own app** (for example, the 01-rag API): `target: {type: http, url: http://localhost:8000/ask, input_key: question, output_key: answer}`.

## Scorers

| scorer | checks |
|---|---|
| `contains` / `not_contains` | required / forbidden phrases |
| `regex`, `exact`, `number` (with tolerance), `length` | format and arithmetic |
| `json` + `schema` | valid JSON matching a JSON-Schema subset (type, required, enum, items) |
| `refusal` | refuses when it should (privacy, safety), or doesn't over-refuse |
| `judge` | LLM-as-judge with a rubric, the reference answer and the **context** for groundedness; 1–5 score with reasoning |

**Judge reliability, built in:**
- **Calibrate** against human labels: `llmeval kappa labels.csv` gives Cohen's κ. Below 0.6, fix the rubric or the judge model before trusting it.
- **Pairwise** `pairwise(case, a, b, rubric, judge)` asks twice with A/B swapped and only counts a consistent verdict, which cancels position bias.
- The judge prompt tells it not to reward length, and unparseable judge output counts as a failure rather than a pass.

## Regression gate

`compare(base, new)` **fails** if the pass rate drops by more than 2 points **or any case that passed before now fails**. Averages hide regressions; per-case diffs don't. [`examples/eval-gate.yml`](examples/eval-gate.yml) is a ready GitHub Action that blocks a PR when a prompt or model change breaks cases.

## Observability

`Tracer` records nested spans to SQLite (trace id, parent, latency, model, tokens, $ cost from a price table, status and error), usable from any app:

```python
tracer = Tracer("traces.db")
with tracer.span("answer", user=uid) as s:
    r = client.chat.completions.create(...)
    s.record_usage(model, r.usage.prompt_tokens, r.usage.completion_tokens)
```

Runs are **cached** (SQLite, keyed on target + input), so re-running a suite after changing only the scorers costs nothing. Failed calls are **retried** with backoff, and persistent errors are recorded as failures, never silently dropped.

## Tests

`pytest` runs 13 offline tests: suite validation, every scorer, schema errors with paths, judge parsing and failure modes, pairwise bias cancellation, κ, a full run, regression detection, **HTML escaping of model output** (a `<script>` answer can't run in the report), cache and retries, nested tracing with cost, the OpenAI target's message building, and CLI exit codes.

## What I learned

eval design · LLM-as-judge pitfalls (position and length bias, calibration) · regression testing for prompts · tracing and cost accounting · CI quality gates
