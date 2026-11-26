"""Score the v1 model's outputs on the whole v2 dev set with the v2.1 judge: the baseline v2 training has to beat.

    python 07_v2_rules_and_scoring/score_v1_dev.py

Reads results/v1_dev_outputs.csv (run_v1_on_dev.py) and the Haiku decodes in the cache
(rubric_check.py --decode). An output that breaks the v2 format rules scores 0.
Uses the API for judgements not in cache/judge_v21_cache.csv. Writes results/v1_dev_scores.csv.
"""

import csv
import random
from pathlib import Path

import anthropic

from emoji_eval.roundtrip import JUDGE_V21_CACHE_FILE, judge_v2, load_cache, load_judge_v2_cache

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "results" / "v1_dev_outputs.csv"
SCORES = HERE / "results" / "v1_dev_scores.csv"


def mean_ci(values: list[float], n: int = 2000) -> str:
    rng = random.Random(7)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return f"{sum(values) / len(values):.2f} [{means[int(0.025 * n)]:.2f}, {means[int(0.975 * n) - 1]:.2f}]"


def main() -> None:
    with open(OUTPUTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    decodes = load_cache()
    cache = load_judge_v2_cache(JUDGE_V21_CACHE_FILE)
    client = anthropic.Anthropic(max_retries=5)
    out = []
    for i, r in enumerate(rows, 1):
        row = {k: r[k] for k in ("id", "lang", "source", "type", "sentence", "output", "valid_v2")}
        if r["valid_v2"] == "ok":
            guess = decodes[(r["output"], r["lang"])]
            result = judge_v2(client, cache, r["sentence"], guess, version="v2.1")
            row.update(guess=guess, **{c: result[c] for c in ("topic", "event", "outcome")},
                       score=result["score"], reason=result["reason"])
        else:
            row.update(guess="", topic="", event="", outcome="", score=0.0, reason="invalid output")
        out.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    with open(SCORES, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    def report(label: str, subset: list[dict]) -> None:
        print(f"  {label:<14} {len(subset):>3}  {mean_ci([r['score'] for r in subset])}")

    print("\nv1 model on v2 dev, v2.1 judge (0-1, invalid = 0), with 95% intervals")
    report("all", out)
    for lang in ("en", "fr", "es"):
        report(lang, [r for r in out if r["lang"] == lang])
    for source in ("real", "ai"):
        report(source, [r for r in out if r["source"] == source])
    for lang in ("en", "fr", "es"):
        for source in ("real", "ai"):
            report(f"{lang} {source}", [r for r in out if r["lang"] == lang and r["source"] == source])
    print("By type (AI pattern or real-sentence quota)")
    for t in sorted({r["type"] for r in out}):
        report(t, [r for r in out if r["type"] == t])
    valid = [r for r in out if r["valid_v2"] == "ok"]
    for c in ("topic", "event", "outcome"):
        points = [int(r[c]) for r in valid if r[c] != "n/a"]
        print(f"  mean {c}: {sum(points) / len(points):.2f} of 2 on {len(points)} outputs")
    print(f"Wrote {SCORES.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
