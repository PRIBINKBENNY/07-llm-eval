"""Self-contained HTML report for a run (and optionally a comparison with a baseline)."""
from __future__ import annotations

import html
from pathlib import Path

from .runner import RunResult, compare

CSS = """
:root{color-scheme:light dark;--bg:light-dark(#fff,#0d0e18);--ink:light-dark(#14110f,#e9e9f2);--mut:light-dark(#5d6170,#9aa0b4);
--line:light-dark(#e3e5ec,#262838);--acc:light-dark(#1d2bd8,#8d97ff);--ok:light-dark(#137a43,#57c78b);--bad:light-dark(#c2261b,#ff7b6e)}
*{box-sizing:border-box;min-width:0}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:32px max(4vw,16px)}h1{font:600 30px/1.1 Spectral,Georgia,serif;margin:0 0 6px}
.sub{color:var(--mut);margin:0 0 24px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:28px}
.kpi{border-top:3px solid var(--acc);padding:10px 2px}.kpi b{display:block;font-size:28px;font-variant-numeric:tabular-nums}.kpi span{color:var(--mut);font-size:13px}
h2{font-size:17px;margin:28px 0 10px}.bar{display:grid;grid-template-columns:180px 1fr 56px;gap:10px;align-items:center;margin:6px 0}
.track{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.fill{height:100%;background:var(--acc)}
.bar em{font-style:normal;text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:500}.pass{color:var(--ok);font-weight:600}.fail{color:var(--bad);font-weight:600}
pre{white-space:pre-wrap;margin:0;font:13px/1.45 ui-monospace,Menlo,monospace;max-height:9em;overflow:auto}
.scroll{overflow-x:auto}.tag{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 7px;font-size:12px;margin:0 3px 3px 0}
@media(max-width:600px){.bar{grid-template-columns:110px 1fr 48px}}
"""


def _bars(d: dict) -> str:
    return "".join(f'<div class="bar"><span>{html.escape(k)}</span><div class="track"><div class="fill" '
                   f'style="width:{v}%"></div></div><em>{v}%</em></div>' for k, v in sorted(d.items(), key=lambda x: x[1]))


def render(run: RunResult, baseline: RunResult | None = None) -> str:
    s = run.summary()
    cmp = compare(baseline, run) if baseline else None
    kpis = [("pass rate", f"{s['pass_rate']}%"), ("cases", s["cases"]), ("p50 latency", f"{s['latency_ms']['p50']} ms"),
            ("p95 latency", f"{s['latency_ms']['p95']} ms"),
            ("cost", f"${s['cost_usd']:.4f}" if s["cost_usd"] is not None else "n/a"), ("errors", s["errors"])]
    if cmp:
        kpis.insert(1, ("vs baseline", f"{cmp['delta_pts']:+} pts"))
    rows = []
    for r in sorted(run.results, key=lambda r: (r.passed, r.id)):
        reasons = "<br>".join(f"<b>{html.escape(x['name'])}</b>: {html.escape(x['reason'] or 'ok')}" for x in r.scores)
        reg = " ⚠ regression" if cmp and r.id in cmp["regressions"] else ""
        rows.append(f"<tr><td>{html.escape(r.id)}{reg}<br>{''.join(f'<span class=tag>{html.escape(t)}</span>' for t in r.tags)}</td>"
                    f"<td class={'pass' if r.passed else 'fail'}>{'PASS' if r.passed else 'FAIL'}</td>"
                    f"<td><pre>{html.escape(str(r.input))}</pre></td><td><pre>{html.escape(r.output)}</pre></td>"
                    f"<td>{reasons}</td><td>{r.latency_ms:.0f} ms</td></tr>")
    verdict = ""
    if cmp:
        verdict = (f"<p class={'pass' if cmp['ok'] else 'fail'}>Regression gate: {'PASSED' if cmp['ok'] else 'FAILED'}"
                   f" ({len(cmp['regressions'])} regressions, {len(cmp['fixed'])} fixed)</p>")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval: {html.escape(run.suite)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(run.suite)}: {s['pass_rate']}% passing</h1>
<p class="sub">target <b>{html.escape(run.target)}</b> · {html.escape(run.started)} · threshold {int(run.threshold * 100)}%
 · {'meets threshold' if s['passed_threshold'] else 'below threshold'}</p>{verdict}
<div class="kpis">{''.join(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in kpis)}</div>
<h2>By scorer</h2>{_bars(s['by_scorer'])}<h2>By tag</h2>{_bars(s['by_tag'])}
<h2>Cases (failures first)</h2><div class="scroll"><table><thead><tr><th>case</th><th>result</th><th>input</th><th>output</th>
<th>why</th><th>latency</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></main></body></html>"""


def write(run: RunResult, path: str | Path, baseline: RunResult | None = None) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render(run, baseline))
    return str(path)
