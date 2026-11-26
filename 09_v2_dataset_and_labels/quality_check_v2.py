"""Quality check: are the v2 prompt's labels better than the v1 prompt's? All 300 v2 dev sentences.

    python 09_v2_dataset_and_labels/quality_check_v2.py

Labels come from style_check_v2.py's run (results/style_check_v2.csv, run it first):
  - v2: the v2 pipeline's label (2 tries, then up to 2 repairs);
  - v1: the v1 prompt's first label that passes the v2 *format* (1 to 10 emojis). Style warnings are
    ignored for v1, since that prompt never saw the v2 signs; the judge scores meaning only.
Both are measured like the student: Haiku decodes without the sentence, the Claude v2.1 judge
scores the guess. No label = 0, as in the baseline. Claude only measures; dev is never training data.
Uses the Anthropic API (cached) and, for a missing v1 second seed, Ollama.
Output: results/quality_check_v2.csv.
"""

import csv
import random
from pathlib import Path

import anthropic

from emoji_eval.roundtrip import JUDGE_V21_CACHE_FILE, decode, judge_v2, load_cache, load_judge_v2_cache
from emoji_eval.runner import TEMPERATURE
from emoji_eval.validate import check_validity_v2
from labels import Labeller

HERE = Path(__file__).resolve().parent
LABELS = HERE / "results" / "style_check_v2.csv"
OUT = HERE / "results" / "quality_check_v2.csv"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def interval(values: list[float], n: int = 2000) -> str:
    rng = random.Random(7)
    means = sorted(mean([rng.choice(values) for _ in values]) for _ in range(n))
    return f"{mean(values):+.2f} [{means[int(0.025 * n)]:+.2f}, {means[int(0.975 * n) - 1]:+.2f}]"


def main() -> None:
    with open(LABELS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    v1 = Labeller("v1")
    for r in rows:
        r["v1_label"] = next((c for c in (v1.candidate(r["sentence"], TEMPERATURE, seed) for seed in (1, 2))
                              if check_validity_v2(c) == "ok"), "")
        r["v2_label"] = r["v2"]

    client = anthropic.Anthropic(max_retries=5)
    decodes, judgements = load_cache(), load_judge_v2_cache(JUDGE_V21_CACHE_FILE)
    for i, r in enumerate(rows, 1):
        for p in ("v1", "v2"):
            emojis = r[f"{p}_label"]
            if not emojis:
                r[f"{p}_guess"], r[f"{p}_score"] = "", "0"
                continue
            guess = decode(client, decodes, emojis, r["lang"])
            r[f"{p}_guess"] = guess
            r[f"{p}_score"] = str(judge_v2(client, judgements, r["sentence"], guess, version="v2.1")["score"])
        if i % 50 == 0:
            print(f"{i}/{len(rows)}", flush=True)

    fields = ["id", "lang", "source", "type", "tags", "sentence",
              "v1_label", "v1_guess", "v1_score", "v2_label", "v2_guess", "v2_score"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def report(name: str, group: list[dict]) -> None:
        a = [float(r["v1_score"]) for r in group]
        b = [float(r["v2_score"]) for r in group]
        print(f"{name:<22}{len(group):>5}{mean(a):>8.2f}{mean(b):>8.2f}   {interval([y - x for x, y in zip(a, b)])}")

    print(f"\n{'':<22}{'n':>5}{'v1':>8}{'v2':>8}   v2 - v1 [95% interval]")
    report("all", rows)
    for lang in ("en", "fr", "es"):
        report(lang, [r for r in rows if r["lang"] == lang])
    for source in ("real", "ai"):
        report(source, [r for r in rows if r["source"] == source])
    for tag in ("question", "conditional", "long", "speech", "number", "negation"):
        group = [r for r in rows if tag in r["tags"].split()]
        if group:
            report(f"tag {tag}", group)
    print(f"\nno label: v1 {sum(not r['v1_label'] for r in rows)}, v2 {sum(not r['v2_label'] for r in rows)}")
    better = sum(float(r["v2_score"]) > float(r["v1_score"]) for r in rows)
    worse = sum(float(r["v2_score"]) < float(r["v1_score"]) for r in rows)
    print(f"v2 better on {better}, worse on {worse}, equal on {len(rows) - better - worse}")


if __name__ == "__main__":
    main()
