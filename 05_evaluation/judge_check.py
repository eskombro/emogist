"""Check the gist judge against a human, on the same 0-2 scale.

    python 05_evaluation/judge_check.py          # create the blind rating sheet
    python 05_evaluation/judge_check.py --score  # compare your ratings with the judge

The sheet shows only the original sentence and the decoder's guess. The model and the judge's
score are hidden, and the score is looked up from the test results only when comparing.
This checks the judge only: a person who has seen the test sentences cannot check the decoder.
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORES_FILE = HERE.parent / "04_fine_tuning" / "results" / "test_scores.csv"
SHEET_FILE = HERE / "results" / "judge_check_sheet.csv"
PER_GROUP = 13  # per model and language: 52 pairs in total
SEED = 2026
BOOTSTRAPS = 5000
SCALE = (0, 1, 2)


def read_scored() -> list[dict]:
    with open(SCORES_FILE, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def make_sheet(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; not overwriting your ratings")
    rng = random.Random(SEED)
    rows = read_scored()
    picked = []
    for group in sorted({(r["model"], r["lang"]) for r in rows}):
        members = [r for r in rows if (r["model"], r["lang"]) == group]
        picked += rng.sample(members, PER_GROUP)
    rng.shuffle(picked)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "original", "guess", "your_score"])
        for i, r in enumerate(picked, 1):
            writer.writerow([i, r["sentence"], r["decoded"], ""])
    print(f"Wrote {len(picked)} pairs to {path}")


def weighted_kappa(a: list[int], b: list[int]) -> float:
    """Agreement beyond chance, where a 0-vs-2 disagreement costs more than a 0-vs-1."""
    n = len(a)
    seen = sum((x - y) ** 2 for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[x] * count_b[y] * (x - y) ** 2 for x in SCALE for y in SCALE) / n ** 2
    return 1 - seen / expected if expected else float("nan")


def bootstrap(a: list[int], b: list[int], statistic) -> tuple[float, float]:
    rng = random.Random(SEED)
    n = len(a)
    values = []
    for _ in range(BOOTSTRAPS):
        idx = [rng.randrange(n) for _ in range(n)]
        value = statistic([a[i] for i in idx], [b[i] for i in idx])
        if value == value:
            values.append(value)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def mean_difference(a: list[int], b: list[int]) -> float:
    return sum(x - y for x, y in zip(a, b)) / len(a)


def compare(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {path}: create it first")
    judged = {(r["sentence"], r["decoded"]): int(r["gist"]) for r in read_scored()}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [r["pair"] for r in rows if r["your_score"].strip() not in {"0", "1", "2"}]
    if bad:
        raise SystemExit(f"Pairs without a 0, 1 or 2: {', '.join(bad)}")
    human = [int(r["your_score"]) for r in rows]
    judge = [judged[(r["original"], r["guess"])] for r in rows]
    n = len(rows)

    print(f"\n{n} pairs\n\nYou (rows) vs judge (columns)")
    print("      " + "".join(f"{s:>5}" for s in SCALE))
    for h in SCALE:
        print(f"{h:>6}" + "".join(f"{sum(x == h and y == s for x, y in zip(human, judge)):>5}" for s in SCALE))
    print(f"\nExact agreement: {sum(x == y for x, y in zip(human, judge)) / n:.0%}")
    print(f"Opposite calls (0 vs 2): {sum(abs(x - y) == 2 for x, y in zip(human, judge))}")
    low, high = bootstrap(human, judge, weighted_kappa)
    print(f"Weighted kappa: {weighted_kappa(human, judge):.2f} [{low:.2f}, {high:.2f}]")
    low, high = bootstrap(human, judge, mean_difference)
    print(f"Mean score: you {sum(human) / n:.2f}, judge {sum(judge) / n:.2f}, "
          f"you - judge {mean_difference(human, judge):+.2f} [{low:+.2f}, {high:+.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true", help="compare the filled sheet with the judge")
    parser.add_argument("--sheet", type=Path, default=SHEET_FILE, help="path of the rating sheet")
    args = parser.parse_args()
    compare(args.sheet) if args.score else make_sheet(args.sheet)
