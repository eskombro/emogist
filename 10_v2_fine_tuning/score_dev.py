"""Score the v2 runs on the v2 dev set, the same way as the v1 baseline (07/score_v1_dev.py).

    python 10_v2_fine_tuning/score_dev.py          # count what is missing from the caches, no API calls
    python 10_v2_fine_tuning/score_dev.py --run    # decode + judge what is missing (API), then report

Reads runs/run_<name>/dev_outputs_<name>.csv for each run in RUNS that is there (2B main and control, 4B main).
Haiku decodes (unchanged v1 decoder prompt), the v2.1 judge scores topic/event/outcome; an output
that breaks the v2 format rules scores 0. Writes results/dev_scores_<run>.csv and compares each run
with the v1 model (07/results/v1_dev_scores.csv) and main with control, on the same sentences.
"""

import argparse
import csv
import random
from pathlib import Path

import anthropic

from emoji_eval.roundtrip import (DECODER_PROMPT, JUDGE_PROMPT_V21, JUDGE_V21_CACHE_FILE, decode, judge_v2,
                                  load_cache, load_judge_v2_cache)
from emoji_eval.validate import check_validity_v2

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = {"main": "qwen2b_v2_main", "control": "qwen2b_v2_control", "main_4b": "qwen4b_v2_main"}
BASELINE = REPO / "07_v2_rules_and_scoring" / "results" / "v1_dev_scores.csv"
DEV = REPO / "08_v2_spanish_and_eval_sets" / "sets" / "dev.csv"

# Rough cost per call in dollars, from prompt sizes and 2026-11 prices
# (Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10 per MTok; ~4 characters per token).
DECODE_COST = (len(DECODER_PROMPT) / 4 + 50) * 0.10e-6 + 40 * 0.50e-6
JUDGE_COST = (len(JUDGE_PROMPT_V21) / 4 + 80) * 2e-6 + 250 * 10e-6


def outputs_file(run: str) -> Path:
    return HERE / "runs" / f"run_{RUNS[run]}" / f"dev_outputs_{RUNS[run]}.csv"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bootstrap(values: list[float], n: int = 2000) -> str:
    rng = random.Random(7)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return f"{sum(values) / len(values):+.2f} [{means[int(0.025 * n)]:+.2f}, {means[int(0.975 * n) - 1]:+.2f}]"


def mean_ci(values: list[float]) -> str:
    return bootstrap(values).replace("+", "")


def load_runs() -> dict[str, list[dict]]:
    dev_ids = [r["id"] for r in read_csv(DEV)]
    runs = {}
    for run in RUNS:
        path = outputs_file(run)
        if not path.exists():
            print(f"[{run}] no outputs yet ({path.relative_to(REPO)})")
            continue
        rows = read_csv(path)
        if [r["id"] for r in rows] != dev_ids:
            raise SystemExit(f"[{run}] outputs don't match the dev set ids and order")
        for r in rows:
            r["valid_v2"] = check_validity_v2(r["output"])
        runs[run] = rows
    if not runs:
        raise SystemExit("No run outputs found: unzip the Colab archives into runs/ first")
    return runs


def count_missing(runs: dict[str, list[dict]]) -> None:
    decodes = load_cache()
    judgements = load_judge_v2_cache(JUDGE_V21_CACHE_FILE)
    to_decode, to_judge = set(), 0
    for run, rows in runs.items():
        valid = [r for r in rows if r["valid_v2"] == "ok"]
        print(f"[{run}] {len(valid)}/{len(rows)} valid (v2 rules)")
        for r in valid:
            key = (r["output"], r["lang"])
            if key not in decodes:
                to_decode.add(key)
            elif (r["sentence"], decodes[key]) not in judgements:
                to_judge += 1
    # Each new decode is followed by a judgement (unless two runs share an output and its guess).
    to_judge += len(to_decode)
    cost = len(to_decode) * DECODE_COST + to_judge * JUDGE_COST
    print(f"Missing: {len(to_decode)} Haiku decodes, up to {to_judge} Sonnet judgements, "
          f"roughly ${cost:.2f}. Run with --run to make these calls.")


def score(runs: dict[str, list[dict]]) -> dict[str, list[dict]]:
    decodes = load_cache()
    judgements = load_judge_v2_cache(JUDGE_V21_CACHE_FILE)
    client = anthropic.Anthropic(max_retries=5)
    scored = {}
    for run, rows in runs.items():
        out = []
        for i, r in enumerate(rows, 1):
            row = {k: r[k] for k in ("id", "lang", "source", "type", "sentence", "output", "valid_v2")}
            if r["valid_v2"] == "ok":
                guess = decode(client, decodes, r["output"], r["lang"])
                result = judge_v2(client, judgements, r["sentence"], guess, version="v2.1")
                row.update(guess=guess, **{c: result[c] for c in ("topic", "event", "outcome")},
                           score=result["score"], reason=result["reason"])
            else:
                row.update(guess="", topic="", event="", outcome="", score=0.0, reason="invalid output")
            out.append(row)
            if i % 50 == 0:
                print(f"  [{run}] {i}/{len(rows)}", flush=True)
        path = HERE / "results" / f"dev_scores_{run}.csv"
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0]))
            writer.writeheader()
            writer.writerows(out)
        print(f"Wrote {path.relative_to(REPO)}")
        scored[run] = out
    return scored


def report(scored: dict[str, list[dict]]) -> None:
    baseline = read_csv(BASELINE)
    for r in baseline:
        r["score"] = float(r["score"])
    models = {"v1": baseline, **scored}

    def cells(rows: list[dict]) -> list[tuple[str, list[dict]]]:
        out = [("all", rows)]
        out += [(lang, [r for r in rows if r["lang"] == lang]) for lang in ("en", "fr", "es")]
        out += [("en+fr", [r for r in rows if r["lang"] in ("en", "fr")])]
        out += [(src, [r for r in rows if r["source"] == src]) for src in ("real", "ai")]
        return out

    print("\nScore on v2 dev (0-1, invalid = 0, v2.1 judge), with 95% intervals")
    for name, rows in models.items():
        valid = sum(r["valid_v2"] == "ok" for r in rows)
        print(f"[{name}] valid {valid}/{len(rows)}")
        for label, subset in cells(rows):
            print(f"  {label:<6} {len(subset):>3}  {mean_ci([r['score'] for r in subset])}")

    def paired(a: str, b: str) -> None:
        if a not in models or b not in models:
            return
        print(f"\n{a} - {b}, paired on the same sentences")
        for (label, ra), (_, rb) in zip(cells(models[a]), cells(models[b])):
            assert [r["id"] for r in ra] == [r["id"] for r in rb]
            print(f"  {label:<6} {bootstrap([x['score'] - y['score'] for x, y in zip(ra, rb)])}")

    paired("main", "v1")
    paired("control", "v1")
    paired("main", "control")
    paired("main_4b", "v1")
    paired("main_4b", "main")

    print("\nBy type")
    types = sorted({r["type"] for r in baseline})
    print("  " + f"{'':<14}" + "".join(f"{name:>9}" for name in models))
    for t in types:
        print(f"  {t:<14}" + "".join(
            f"{sum(r['score'] for r in rows if r['type'] == t) / sum(r['type'] == t for r in rows):>9.2f}"
            for rows in models.values()))
    print("\nMean criteria points (of 2) on valid outputs")
    for name, rows in models.items():
        valid = [r for r in rows if r["valid_v2"] == "ok"]
        parts = []
        for c in ("topic", "event", "outcome"):
            points = [int(r[c]) for r in valid if r[c] != "n/a"]
            parts.append(f"{c} {sum(points) / len(points):.2f}")
        print(f"  [{name}] " + ", ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make the API calls and report")
    args = parser.parse_args()
    runs = load_runs()
    count_missing(runs)
    if args.run:
        report(score(runs))


if __name__ == "__main__":
    main()
