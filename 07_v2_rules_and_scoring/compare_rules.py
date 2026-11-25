"""Compare the v1 and v2 format rules on every emoji output saved in the repository.

    python 07_v2_rules_and_scoring/compare_rules.py

No model calls. Reads every CSV in the repository, collects the distinct values of columns
named like output/emojis/label, and checks each one with both rule sets.
"""

import csv
from collections import Counter
from pathlib import Path

from emoji_eval.validate import check_conventions_v2, check_validity, check_validity_v2

REPO = Path(__file__).resolve().parent.parent
COLUMN_HINTS = ("output", "emojis", "label")
SKIP = {".venv", "cache"}


def saved_outputs() -> set[str]:
    outputs = set()
    for path in REPO.rglob("*.csv"):
        if SKIP & set(path.relative_to(REPO).parts):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for column, value in row.items():
                    if column and value and len(value) < 120 and any(h in column.lower() for h in COLUMN_HINTS):
                        outputs.add(value)
    return outputs


def reason(check: str) -> str:
    return "non-emoji" if check.startswith("non-emoji") else check


def main() -> None:
    outputs = sorted(saved_outputs())
    v1 = {o: check_validity(o) for o in outputs}
    v2 = {o: check_validity_v2(o) for o in outputs}
    print(f"{len(outputs)} distinct saved outputs")
    print(f"invalid under v1: {sum(c != 'ok' for c in v1.values())}")
    print(f"invalid under v2: {sum(c != 'ok' for c in v2.values())}\n")
    print("v1 reasons:", ", ".join(f"{r} {n}" for r, n in Counter(reason(c) for c in v1.values() if c != "ok").most_common()))
    print("v2 reasons:", ", ".join(f"{r} {n}" for r, n in Counter(reason(c) for c in v2.values() if c != "ok").most_common()))
    newly_valid = [o for o in outputs if v1[o] != "ok" and v2[o] == "ok"]
    short = sum("fewer than 3 emojis" in check_conventions_v2(o) for o in newly_valid)
    print(f"\nvalid in v2 only: {len(newly_valid)} ({short} with the 'fewer than 3 emojis' warning)")
    print("  " + "   ".join(newly_valid))


if __name__ == "__main__":
    main()
