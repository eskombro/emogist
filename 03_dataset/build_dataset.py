"""Step 3e: label the full 50/50 dataset and write the training file.

Per language: 750 real + 750 AI sentences (625 from the base patterns, 125 numbers).
Sentences are taken in their existing pool order, and only the automatic checks drop a
label: Claude never picks or rejects a sentence or label. Pilot labels are reused.

    python 03_dataset/build_dataset.py

Resumable: sentences already in labels/full_labels.csv are not sent again.
"""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from label_pilot import LABELS_FILE, run_labelling

HERE = Path(__file__).resolve().parent
SENTENCES = HERE / "sentences"
FULL_LABELS = HERE / "labels" / "full_labels.csv"
OUT_FILE = HERE / "full" / "train.jsonl"
QUOTAS = {"real": 750, "base": 625, "numbers": 125}  # per language
MARGIN = {"real": 820, "base": 665, "numbers": 170}  # labelled, to cover dropped labels


def read(name: str) -> list[dict]:
    with open(SENTENCES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_of(row: dict) -> str:
    return "real" if row["source"] == "real" else "numbers" if row["pattern"] == "numbers" else "base"


def to_label() -> list[dict]:
    real = [{**r, "source": "real", "pattern": ""} for r in read("real_pool.csv")]
    ai = [{**r, "source": "ai", "author": ""} for r in read("ai_pool.csv")]
    chosen = []
    for lang in ("en", "fr"):
        for group, size in MARGIN.items():
            rows = [r for r in real + ai if r["lang"] == lang and group_of(r) == group][:size]
            if len(rows) < size:
                raise SystemExit(f"Only {len(rows)} sentences for {lang}/{group}: need {size}")
            chosen += rows
    return chosen


def main() -> None:
    if not FULL_LABELS.exists():
        shutil.copy(LABELS_FILE, FULL_LABELS)  # the pilot file itself stays unchanged
    sentences = to_label()
    fields = ("id", "lang", "source", "sentence")
    labelled = run_labelling([{k: r[k] for k in fields} for r in sentences], FULL_LABELS)
    meta = {r["id"]: r for r in sentences}

    final, usable = [], Counter()
    for lang in ("en", "fr"):
        for group, size in QUOTAS.items():
            good = [r for r in labelled if r["lang"] == lang and group_of({**r, **meta[r["id"]]}) == group
                    and r["status"] == "ok"]
            usable[lang, group] = len(good)
            if len(good) < size:
                raise SystemExit(f"Only {len(good)} usable labels for {lang}/{group}: need {size}")
            final += [{**r, "group": group} for r in good[:size]]

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in final:
            m = meta[r["id"]]
            f.write(json.dumps({"id": r["id"], "lang": r["lang"], "source": r["source"], "pattern": m["pattern"],
                                "author": m["author"], "generator": m.get("generator", ""), "sentence": r["sentence"],
                                "emojis": r["emojis"], "teacher": r["teacher"], "prompt": r["prompt"]},
                               ensure_ascii=False) + "\n")
    print("usable labels:", dict(usable))
    print(f"{OUT_FILE.name}: {len(final)} examples")


if __name__ == "__main__":
    main()
