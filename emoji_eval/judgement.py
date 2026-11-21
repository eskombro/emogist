"""Judgement rules: does a model use ⏪/⏩ and 💬 where they belong?

Uses the `tags` column of data/test_sentences.csv:
  time   - a time marker really matters (future plans, "after ten years"...)
  speech - speech or a message (the boss's "ok", a phone call, a text...)
These are the rules prompting alone did not teach the students, so a teacher must
apply them consistently, or fine-tuning won't teach them either.
"""

import csv
from pathlib import Path

from emoji_eval.columns import model_columns
from emoji_eval.paths import TEST_FILE


def has_time_marker(output: str) -> bool:
    return "⏪" in output or "⏩" in output


def rate(outputs: list[str], test) -> str:
    if not outputs:
        return "  n/a"
    return f"{sum(map(test, outputs)) / len(outputs):5.0%}"


def judgement_report(results_file: Path) -> None:
    with open(TEST_FILE, newline="", encoding="utf-8") as f:
        tags = {r["id"]: set(r["tags"].split()) for r in csv.DictReader(f)}
    with open(results_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = model_columns(rows[0])

    def outputs(model: str, tag: str, present: bool) -> list[str]:
        return [r[model] for r in rows if (tag in tags[r["id"]]) == present]

    n_time = sum("time" in t for t in tags.values())
    n_speech = sum("speech" in t for t in tags.values())
    print(f"\nJudgement rules (higher is better for 'where needed', lower for 'elsewhere'):")
    print(f"  {'model':<22} ⏪/⏩ where needed  ⏪/⏩ elsewhere   💬 on speech  💬 elsewhere")
    total = len(tags)
    print(f"  {'':<22} ({n_time} sentences)      ({total - n_time})          ({n_speech})         ({total - n_speech})")
    for m in models:
        print(f"  {m:<22} {rate(outputs(m, 'time', True), has_time_marker):>8}"
              f"         {rate(outputs(m, 'time', False), has_time_marker):>8}"
              f"        {rate(outputs(m, 'speech', True), lambda o: '💬' in o):>8}"
              f"     {rate(outputs(m, 'speech', False), lambda o: '💬' in o):>8}")
