"""Human check of the v2 rubric (topic, event, outcome) on the v2 dev set.

    python 07_v2_rules_and_scoring/rubric_check.py --decode              # Haiku decodes every valid output (API)
    python 07_v2_rules_and_scoring/rubric_check.py --sheet --round 1     # blind rating sheet, no API call
    python 07_v2_rules_and_scoring/rubric_check.py --score --round 1     # judges + comparison (API, after rating)

The outputs come from run_v1_on_dev.py (the v1 model). Each sheet shows the original sentence and
the decoder's guess only: no emojis, model, source or judge score. Pass mark, set before rating:
rank correlation of at least 0.7 between the human total and the judge's total.

Round 1: 60 pairs (10 per language and source, seed 7), v2 judge, plus the v1 judge for comparison.
         Result: v2 0.67, failed; v1 0.72.
Round 2: 42 fresh pairs (7 per language and source, seed 8, none from round 1), v2.1 judge: the
         same criteria with two rules the author set after round 1 (mood and exactly who the people
         are are not penalised; reversed roles and negation still are). The v1 judge runs too.
"""

import argparse
import csv
import random
from pathlib import Path

import anthropic
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from emoji_eval.roundtrip import (JUDGE_V21_CACHE_FILE, JUDGE_V2_CACHE_FILE, criteria_score, decode, judge,
                                  judge_v2, load_cache, load_judge_cache, load_judge_v2_cache)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTPUTS = RESULTS / "v1_dev_outputs.csv"
CRITERIA = ("topic", "event", "outcome")
ROUNDS = {
    1: {"prefix": "rubric", "seed": 7, "per_cell": 10, "version": "v2", "cache": JUDGE_V2_CACHE_FILE,
        "with_v1": True},
    2: {"prefix": "rubric2", "seed": 8, "per_cell": 7, "version": "v2.1", "cache": JUDGE_V21_CACHE_FILE,
        "with_v1": True},
}


def paths(n: int) -> dict:
    prefix = ROUNDS[n]["prefix"]
    return {"key": RESULTS / f"{prefix}_pairs.csv",      # which output each pair comes from (hidden from the rater)
            "sheet": RESULTS / f"{prefix}_sheet.csv",    # blank sheet as handed out
            "ratings": RESULTS / f"{prefix}_ratings.csv",  # the author's ratings (filled sheet)
            "scores": RESULTS / f"{prefix}_scores.csv"}  # ratings and judges, per pair


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_rows() -> list[dict]:
    return [r for r in read_csv(OUTPUTS) if r["valid_v2"] == "ok"]


def run_decode() -> None:
    client = anthropic.Anthropic(max_retries=5)
    cache = load_cache()
    rows = valid_rows()
    todo = sum((r["output"], r["lang"]) not in cache for r in rows)
    print(f"{len(rows)} valid outputs, {todo} to decode")
    for i, r in enumerate(rows, 1):
        decode(client, cache, r["output"], r["lang"])
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    print("Done; decodes are in cache/decode_cache.csv")


def make_sheet(n: int) -> None:
    p, cfg = paths(n), ROUNDS[n]
    if p["sheet"].exists():
        raise SystemExit(f"{p['sheet'].name} already exists: delete it first if you really want a new draw")
    cache = load_cache()
    rows = valid_rows()
    missing = [r["id"] for r in rows if (r["output"], r["lang"]) not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} outputs not decoded yet: run --decode first")
    used = {r["id"] for m in ROUNDS if m < n for r in read_csv(paths(m)["key"])}
    rows = [r for r in rows if r["id"] not in used]
    rng = random.Random(cfg["seed"])
    picked = []
    for lang in ("en", "fr", "es"):
        for source in ("real", "ai"):
            cell = [r for r in rows if r["lang"] == lang and r["source"] == source]
            picked += rng.sample(cell, cfg["per_cell"])
    rng.shuffle(picked)
    key, sheet = [], []
    for i, r in enumerate(picked, 1):
        guess = cache[(r["output"], r["lang"])]
        key.append({"pair": i, "id": r["id"], "lang": r["lang"], "source": r["source"], "type": r["type"],
                    "output": r["output"], "original": r["sentence"], "guess": guess})
        sheet.append({"pair": i, "original": r["sentence"], "guess": guess,
                      "topic": "", "event": "", "outcome": "", "note": ""})
    write_csv(p["key"], key)
    write_csv(p["sheet"], sheet)
    print(f"Wrote {p['sheet'].name} ({len(sheet)} pairs, {len(used)} earlier pairs excluded) and {p['key'].name}")


def bootstrap(stat, *columns, n=2000) -> tuple[float, float]:
    """95% interval of stat over pairs resampled with replacement."""
    rng = random.Random(7)
    size = len(columns[0])
    values = []
    for _ in range(n):
        idx = [rng.randrange(size) for _ in range(size)]
        value = stat(*[[c[i] for i in idx] for c in columns])
        if value == value:  # skip NaN (a resample with no variation)
            values.append(value)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def spearman(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def kappa(a, b) -> float:
    return float(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2]))


def score(n: int) -> None:
    """The round's judges on the rated pairs, compared with the author's ratings."""
    p, cfg = paths(n), ROUNDS[n]
    name = f"{cfg['version']} judge"
    ratings = read_csv(p["ratings"])
    key = {r["pair"]: r for r in read_csv(p["key"])}
    if len(ratings) != len(key):
        raise SystemExit(f"{len(ratings)} ratings for {len(key)} pairs")
    for r in ratings:
        k = key[r["pair"]]
        if (r["original"], r["guess"]) != (k["original"], k["guess"]):
            raise SystemExit(f"pair {r['pair']}: text differs from the key")
        for c in CRITERIA:
            if r[c] not in ("0", "1", "2") and not (c == "outcome" and r[c] == "n/a"):
                raise SystemExit(f"pair {r['pair']}: {c} is {r[c]!r}")

    client = anthropic.Anthropic(max_retries=5)
    v1_cache, v2_cache = load_judge_cache(), load_judge_v2_cache(cfg["cache"])
    out = []
    for i, r in enumerate(ratings, 1):
        v2 = judge_v2(client, v2_cache, r["original"], r["guess"], version=cfg["version"])
        k = key[r["pair"]]
        row = {"pair": r["pair"], "lang": k["lang"], "source": k["source"],
               **{f"human_{c}": r[c] for c in CRITERIA},
               "human_score": criteria_score(r["topic"], r["event"], r["outcome"]),
               **{f"judge_{c}": v2[c] for c in CRITERIA}, "judge_score": v2["score"], "judge_reason": v2["reason"]}
        if cfg["with_v1"]:
            row["v1_gist"], row["v1_reason"] = judge(client, v1_cache, r["original"], r["guess"])
        out.append({**row, "note": r["note"]})
        if i % 20 == 0:
            print(f"  judged {i}/{len(ratings)}", flush=True)
    write_csv(p["scores"], out)

    human = [r["human_score"] for r in out]
    judged = [r["judge_score"] for r in out]
    means = f"author {sum(human) / len(human):.2f}, {name} {sum(judged) / len(judged):.2f}"
    if cfg["with_v1"]:
        means += f", v1 judge {sum(r['v1_gist'] for r in out) / len(out) / 2:.2f}"
    print(f"\nRound {n}, {len(out)} pairs. Mean score (0-1): {means}")
    print(f"Rank correlation with the author's total (pass mark for the {name}: 0.7)")
    compared = [(name, judged)] + ([("v1 judge", [r["v1_gist"] for r in out])] if cfg["with_v1"] else [])
    for label, values in compared:
        low, high = bootstrap(spearman, human, values)
        print(f"  {label}: {spearman(human, values):.2f} [{low:.2f}, {high:.2f}]")
    print(f"Per criterion, {name} vs author (weighted kappa)")
    for c in CRITERIA:
        both = [(int(r[f"human_{c}"]), int(r[f"judge_{c}"])) for r in out
                if r[f"human_{c}"] != "n/a" and r[f"judge_{c}"] != "n/a"]
        a, b = [x for x, _ in both], [y for _, y in both]
        low, high = bootstrap(kappa, a, b)
        print(f"  {c}: {kappa(a, b):.2f} [{low:.2f}, {high:.2f}] on {len(both)} pairs")
    print(f"  outcome n/a: author {sum(r['human_outcome'] == 'n/a' for r in out)}, "
          f"judge {sum(r['judge_outcome'] == 'n/a' for r in out)}")
    print("By language (wide intervals)")
    for lang in ("en", "fr", "es"):
        rows = [r for r in out if r["lang"] == lang]
        h, j = [r["human_score"] for r in rows], [r["judge_score"] for r in rows]
        print(f"  {lang}: author {sum(h) / len(h):.2f}, judge {sum(j) / len(j):.2f}, "
              f"rank correlation {spearman(h, j):.2f}")
    print(f"Wrote {p['scores'].relative_to(HERE.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--decode", action="store_true")
    group.add_argument("--sheet", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), default=2)
    args = parser.parse_args()
    if args.decode:
        run_decode()
    elif args.sheet:
        make_sheet(args.round)
    else:
        score(args.round)


if __name__ == "__main__":
    main()
