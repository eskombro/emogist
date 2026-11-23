"""Step 3f: draw a random sample of the full dataset for a human to read and score.

20 examples per language and group (real, AI base, AI numbers), in shuffled order so the
source doesn't sit next to the score. The draw is seeded and made by code: Claude doesn't
read or choose the examples. An existing sheet is never overwritten, as it may hold scores.

    python 03_dataset/sample_labels.py

Score each row: gist 2 = kept, 1 = partly kept, 0 = lost.
"""

import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_FILE = HERE / "full" / "train.jsonl"
OUT_FILE = HERE / "review" / "label_sample.csv"
PER_GROUP = 20
SEED = 42
FIELDS = ["row", "lang", "sentence", "emojis", "gist_0_1_2", "notes", "id"]


def group_of(row: dict) -> str:
    return "real" if row["source"] == "real" else "numbers" if row["pattern"] == "numbers" else "base"


def main() -> None:
    if OUT_FILE.exists():
        raise SystemExit(f"{OUT_FILE} already exists: move it first, it may contain your scores")
    with open(TRAIN_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(SEED)
    sample = []
    for lang in ("en", "fr"):
        for group in ("real", "base", "numbers"):
            sample += rng.sample([r for r in rows if r["lang"] == lang and group_of(r) == group], PER_GROUP)
    rng.shuffle(sample)
    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for n, r in enumerate(sample, 1):
            writer.writerow({"row": n, "lang": r["lang"], "sentence": r["sentence"], "emojis": r["emojis"],
                             "gist_0_1_2": "", "notes": "", "id": r["id"]})
    print(f"{OUT_FILE.name}: {len(sample)} examples")


if __name__ == "__main__":
    main()
