"""Check Flash as the label selector: does it rank the author's step 7 pairs like the author does?

    python 09_v2_dataset_and_labels/check_flash_judge.py

Flash scores the 102 pairs the author rated in step 7 (same originals and Haiku guesses), with the
v2.1 rubric. Ollama only, no Anthropic calls: the Claude judges' scores are read from step 7's files.
Round 2 is the fair comparison: the author rated it with the same two rules the v2.1 rubric has.
Round 1 was rated before those rules existed.
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "07_v2_rules_and_scoring"))

from flash_roundtrip import FlashRoundTrip  # noqa: E402
from rubric_check import CRITERIA, bootstrap, kappa, paths, read_csv, spearman  # noqa: E402

OUT = HERE / "results" / "flash_judge_check.csv"


def main() -> None:
    flash = FlashRoundTrip()
    out = []
    for n, claude_name in ((1, "Claude v2"), (2, "Claude v2.1")):
        ratings = {r["pair"]: r for r in read_csv(paths(n)["ratings"])}
        for r in read_csv(paths(n)["scores"]):
            pair = ratings[r["pair"]]
            f = flash.judge(pair["original"], pair["guess"])
            out.append({"round": n, "pair": r["pair"], "lang": r["lang"], "source": r["source"],
                        "human_score": float(r["human_score"]), "claude_judge": claude_name,
                        "claude_score": float(r["judge_score"]), "v1_gist": int(r["v1_gist"]),
                        **{f"human_{c}": r[f"human_{c}"] for c in CRITERIA},
                        **{f"flash_{c}": f[c] for c in CRITERIA}, "flash_score": f["score"], "flash_reason": f["reason"]})
        print(f"  round {n} done", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    print("\nRank correlation with the author's total, 95% interval")
    for n in (1, 2):
        rows = [r for r in out if r["round"] == n]
        human = [r["human_score"] for r in rows]
        print(f"Round {n} ({len(rows)} pairs)")
        for name, key in (("Flash v2.1", "flash_score"), (rows[0]["claude_judge"], "claude_score"),
                          ("Claude v1", "v1_gist")):
            values = [r[key] for r in rows]
            low, high = bootstrap(spearman, human, values)
            print(f"  {name:<12} {spearman(human, values):.2f} [{low:.2f}, {high:.2f}]")
        flash_vs_claude = lambda h, a, b: spearman(h, a) - spearman(h, b)  # noqa: E731
        a, b = [r["flash_score"] for r in rows], [r["claude_score"] for r in rows]
        low, high = bootstrap(flash_vs_claude, human, a, b)
        print(f"  Flash minus {rows[0]['claude_judge']}: {flash_vs_claude(human, a, b):+.2f} [{low:+.2f}, {high:+.2f}]")
        print(f"  mean score: author {sum(human) / len(human):.2f}, Flash {sum(a) / len(a):.2f}, "
              f"{rows[0]['claude_judge']} {sum(b) / len(b):.2f}")
        for c in CRITERIA:
            both = [(int(r[f"human_{c}"]), int(r[f"flash_{c}"])) for r in rows
                    if "n/a" not in (r[f"human_{c}"], r[f"flash_{c}"])]
            print(f"  {c}: weighted kappa {kappa([x for x, _ in both], [y for _, y in both]):.2f} on {len(both)}")
    print(f"Wrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
