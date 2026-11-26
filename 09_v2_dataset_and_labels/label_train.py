"""Step 9c: label the training sentences with Flash and build the training files.

    python 09_v2_dataset_and_labels/label_train.py               # full run
    python 09_v2_dataset_and_labels/label_train.py --limit 20    # test: first 20 per language and source

Every sentence from sentences/real_train.csv and sentences/ai_train.csv gets one Flash label with
the v2 prompt (label_prompt_v2.py): two tries at temperature 0.3, then up to two repairs (labels.py).
History: the first full run used prompt v5, then v6 for the sentences it couldn't label (❓ allowed
right after the question part). The author's review (results/review_round1.csv) found 🫵 used for
"I / me / we" in about 11% of labels, so everything is labelled again with prompt v7: the 🫵 rule,
the author's corrected speech examples, and retries on plain sentence rules (labels.strict_problem).
Sentences without a valid label are dropped, and the margin drawn in 9a/9b fills their place:
  - real: 1,000 labelled sentences per language, the margin cut from the last non-questions in draw order;
  - AI: the first <quota> labelled sentences per language and pattern, in generation order.
Nothing is picked by hand and Claude never sees or chooses a label.

Outputs (full run only):
  labels/train_rows.csv     every kept row with its source, generator, pattern, tags and how it was labelled
  train_main.jsonl          EN + FR + ES (6,000 rows), shuffled with a fixed seed
  train_control.jsonl       the same EN + FR rows (4,000), in the same order
  review_sheet.csv          50 random rows per language for the author's review (verdict: ok / remove)
Labels are cached (cache/flash_label_cache.csv), so a rerun only pays for what is missing.
"""

import argparse
import csv
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import emoji

from emoji_eval.validate import check_conventions_v2, content_count
from generate_ai_train import QUOTAS
from label_prompt_v2 import worksheet_examples
from labels import Labeller
from labels import MODEL as TEACHER

HERE = Path(__file__).resolve().parent
REAL = HERE / "sentences" / "real_train.csv"
AI = HERE / "sentences" / "ai_train.csv"
ROWS = HERE / "labels" / "train_rows.csv"
MAIN = HERE / "train_main.jsonl"
CONTROL = HERE / "train_control.jsonl"
REVIEW = HERE / "review_sheet.csv"  # round 2: 20 per language, none from round 1
ROUND1 = HERE / "results" / "review_round1.csv"
REAL_PER_LANG = 1000
LANGS = ("en", "fr", "es")
THREADS = 8
SEED = 42
REVIEW_PER_LANG = 20
FIELDS = ["id", "lang", "source", "pattern", "tags", "author", "generator", "sentence", "emojis",
          "attempt", "teacher", "prompt"]


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def example_groups() -> set[str]:
    """Groups of 3+ emojis in the author's example labels, to count near-copies."""
    return {g for _, label in worksheet_examples() for g in label.split(" ") if emoji.emoji_count(g) >= 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="test: label only the first N per language and source")
    args = parser.parse_args()

    real, ai = read(REAL), read(AI)
    rows = []
    for lang in LANGS:
        rows += [{**r, "source": "real", "pattern": "", "generator": ""}
                 for r in real if r["lang"] == lang][:args.limit]
        rows += [{**r, "source": "ai", "tags": "", "author": ""}
                 for r in ai if r["lang"] == lang][:args.limit]
    labeller = Labeller("v2.2", tries=2, repairs=2, strict=True)
    with ThreadPoolExecutor(THREADS) as pool:
        results = list(pool.map(lambda r: labeller.label(r["sentence"]), rows))
    for r, (label, attempt) in zip(rows, results):
        r["emojis"], r["attempt"], r["prompt"], r["teacher"] = label, attempt, labeller.prompt_id, TEACHER

    attempts = Counter(r["attempt"] for r in rows)
    print(f"{len(rows)} sentences: first try {attempts[1]}, second try {attempts[2]}, "
          f"repaired {attempts[3] + attempts[4]}, no label {attempts[0]}")
    print(f"❓ spacing fixes applied: {labeller.spaced}")
    print(f"Flash label requests this run: {labeller.requests}, tokens in {labeller.tokens['prompt']}, "
          f"out {labeller.tokens['output']}")
    labelled = [r for r in rows if r["emojis"]]
    groups = example_groups()
    near = sum(any(g in groups for g in r["emojis"].split(" ")) for r in labelled)
    print(f"mean emojis {sum(content_count(r['emojis']) for r in labelled) / len(labelled):.1f}; "
          f"➡️ {sum('➡' in r['emojis'] for r in labelled) / len(labelled):.0%}; "
          f"sharing a 3+ emoji group with an example: {near}; "
          f"style warnings left: {sum(bool([w for w in check_conventions_v2(r['emojis']) if w != 'fewer than 3 emojis']) for r in labelled)}")
    if args.limit:
        for r in random.Random(SEED).sample(labelled, min(15, len(labelled))):
            print(f"[{r['lang']} {r['source']} {r['pattern'] or r['tags']}] {r['sentence']}\n    {r['emojis']}")
        return

    kept = []
    for lang in LANGS:
        # Real: the margin is cut from the last non-question sentences, so no scarce question is lost.
        real = [r for r in labelled if r["lang"] == lang and r["source"] == "real"]
        surplus = len(real) - REAL_PER_LANG
        cut = [r["id"] for r in real if "question" not in r["tags"].split()][-surplus:] if surplus > 0 else []
        kept += [r for r in real if r["id"] not in set(cut)][:REAL_PER_LANG]
        for pattern, quota in QUOTAS.items():
            kept += [r for r in labelled if r["lang"] == lang and r["source"] == "ai" and r["pattern"] == pattern][:quota]
    counts = Counter((r["lang"], r["source"]) for r in kept)
    print("kept:", dict(counts))
    ROWS.parent.mkdir(exist_ok=True)
    with open(ROWS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    order = kept[:]
    random.Random(SEED).shuffle(order)
    for path, langs in ((MAIN, LANGS), (CONTROL, ("en", "fr"))):
        with open(path, "w", encoding="utf-8") as f:
            for r in order:
                if r["lang"] in langs:
                    f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")
        print(f"Wrote {path.name}: {sum(r['lang'] in langs for r in order)} rows")
    if not REVIEW.exists():  # never overwrite the author's verdicts
        rng, seen = random.Random(SEED + 1), {r["id"] for r in read(ROUND1)}
        sample = [r for lang in LANGS
                  for r in rng.sample([r for r in kept if r["lang"] == lang and r["id"] not in seen], REVIEW_PER_LANG)]
        with open(REVIEW, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "lang", "source", "sentence", "emojis", "verdict", "note"],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows({**r, "verdict": "", "note": ""} for r in sample)
        print(f"Wrote {REVIEW.name}: {len(sample)} rows to review")


if __name__ == "__main__":
    main()
