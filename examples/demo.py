"""Offline demo: runs the support-bot suite against two SCRIPTED "models" (no API calls) and writes
an HTML report with a regression comparison, to show what a real run looks like.

    python examples/demo.py   ->  examples/sample_report.html
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmeval import Suite, compare, run_suite  # noqa: E402
from llmeval.report import write  # noqa: E402
from llmeval.targets import FunctionTarget  # noqa: E402

V1 = {
    "return-window": "You can return it within 30 days with your receipt.",
    "warranty": "Yes, the warranty is 18 months, so a one-year-old arm is still covered.",
    "hours": "Sorry, support isn't open on Sunday. We're available Mon-Fri 9:00-18:00 IST.",
    "safety-escalation": "Stop using the arm and switch off its power. Then email safety@orion.example right away.",
    "privacy": "Sorry, I can't share other customers' information.",
    "json-ticket": '{"customer": "Asha", "issue": "gripper error after firmware update", "error_code": "E22", "priority": "high"}',
    "maths": "3 x 1,450 = 4,350 rupees.",
    "grounded": "E14 means the encoder lost its zero position, so re-home the arm.",
}
# a "new prompt" that got chattier and broke two things
V2 = dict(V1, privacy="Of course! His order number is #4411.",
          maths="That comes to roughly 4,000 rupees.",
          hours="Our friendly team is here for you! Feel free to reach out any time and we'll do our best to help "
                "you with anything you need, whether it's a question about your arm, a return, or anything else "
                "at all. We really value you as a customer and want you to have the best experience possible.")


def judge(messages):
    text = messages[-1]["content"]
    rubric = text.split("RUBRIC:\n")[1].split("\n\n")[0].lower()
    resp = text.split("RESPONSE:\n")[-1].lower()
    keys = [w for w in ["18 months", "sunday", "safety@orion.example", "encoder", "re-home"] if w in rubric]
    hit = all(k in resp for k in keys)
    return json.dumps({"reasoning": "Covers every rubric point." if hit else "Misses required details.",
                       "score": 5 if hit else 2})


if __name__ == "__main__":
    suite = Suite.load(ROOT / "datasets/support_bot.yaml")
    suite.name = "support-bot (demo: scripted answers)"
    base = run_suite(suite, FunctionTarget(lambda c: V1[c.id]), judge, "prompt-v1")
    new = run_suite(suite, FunctionTarget(lambda c: V2[c.id]), judge, "prompt-v2")
    print(json.dumps(compare(base, new), indent=2))
    print("report:", write(new, ROOT / "examples/sample_report.html", base))
