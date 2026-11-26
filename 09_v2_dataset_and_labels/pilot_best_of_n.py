"""Pilot: does best-of-n give better labels than a single label?

    python 09_v2_dataset_and_labels/pilot_best_of_n.py

60 dev sentences (10 per language and source, seed 9). A = the first Flash label (the v1 way),
B = the best-of-n label (labels.py). Both are measured like the student: Haiku decodes and the
Claude v2.1 judge scores. Claude only measures here; these are dev sentences, never training data.
Uses Ollama Cloud and the Anthropic API (decodes and judgements are cached).
"""

import csv
import random
from pathlib import Path

import anthropic

from emoji_eval.roundtrip import JUDGE_V21_CACHE_FILE, decode, judge_v2, load_cache, load_judge_v2_cache
from labels import Labeller

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "08_v2_spanish_and_eval_sets" / "sets" / "dev.csv"
OUT = HERE / "results" / "pilot_best_of_n.csv"
SEED = 9
PER_CELL = 10


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def interval(values: list[float], n: int = 2000) -> tuple[float, float]:
    rng = random.Random(7)
    means = sorted(mean([rng.choice(values) for _ in values]) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> None:
    with open(DEV, newline="", encoding="utf-8") as f:
        dev = list(csv.DictReader(f))
    rng = random.Random(SEED)
    picked = []
    for lang in ("en", "fr", "es"):
        for source in ("real", "ai"):
            picked += rng.sample([r for r in dev if r["lang"] == lang and r["source"] == source], PER_CELL)

    labeller = Labeller()
    client = anthropic.Anthropic(max_retries=5)
    decodes, judgements = load_cache(), load_judge_v2_cache(JUDGE_V21_CACHE_FILE)

    def claude_score(sentence: str, emojis: str, lang: str) -> float:
        if not emojis:
            return 0.0
        guess = decode(client, decodes, emojis, lang)
        return judge_v2(client, judgements, sentence, guess, version="v2.1")["score"]

    out = []
    for i, r in enumerate(picked, 1):
        result = labeller.best(r["sentence"], r["lang"])
        flash_first = result["candidates"][0]["score"] if result["first"] else 0.0
        flash_best = max((c["score"] for c in result["candidates"]), default=0.0)
        out.append({"id": r["id"], "lang": r["lang"], "source": r["source"], "type": r["type"],
                    "sentence": r["sentence"], "first": result["first"], "best": result["best"],
                    "n_candidates": len(result["candidates"]), "changed": result["best"] != result["first"],
                    "flash_first": flash_first, "flash_best": flash_best,
                    "claude_first": claude_score(r["sentence"], result["first"], r["lang"]),
                    "claude_best": claude_score(r["sentence"], result["best"], r["lang"])})
        if i % 10 == 0:
            print(f"  {i}/{len(picked)}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    a, b = [r["claude_first"] for r in out], [r["claude_best"] for r in out]
    diffs = [y - x for x, y in zip(a, b)]
    low, high = interval(diffs)
    changed = [r for r in out if r["changed"]]
    print(f"\n{len(out)} dev sentences. Claude v2.1 score (0-1):")
    print(f"  A single label: {mean(a):.2f}   B best-of-n: {mean(b):.2f}   B - A: {mean(diffs):+.2f} [{low:+.2f}, {high:+.2f}]")
    for lang in ("en", "fr", "es"):
        rows = [r for r in out if r["lang"] == lang]
        print(f"  {lang}: A {mean([r['claude_first'] for r in rows]):.2f}, B {mean([r['claude_best'] for r in rows]):.2f}")
    print(f"Extra candidates drawn for {sum(r['n_candidates'] > 1 for r in out)} sentences; label changed for {len(changed)}")
    if changed:
        print(f"  on changed labels: Flash {mean([r['flash_first'] for r in changed]):.2f} -> "
              f"{mean([r['flash_best'] for r in changed]):.2f}, Claude {mean([r['claude_first'] for r in changed]):.2f} -> "
              f"{mean([r['claude_best'] for r in changed]):.2f}  (better {sum(r['claude_best'] > r['claude_first'] for r in changed)}, "
              f"worse {sum(r['claude_best'] < r['claude_first'] for r in changed)})")
    print(f"Flash label requests sent: {labeller.requests} (plus decodes and judgements)")
    print(f"Wrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
