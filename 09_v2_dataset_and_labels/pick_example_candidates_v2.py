"""Create label_examples_v2.csv: the worksheet for the v2 labelling prompt's few-shot examples.

It copies the v1 worksheet (step 3), whose filled rows are the 20 current EN/FR examples, then adds
empty candidates drawn by plain rules from the step 8 real pool:
  - Spanish: everyday, vague, speech and numbers sentences (the same cases as v1 for EN/FR);
  - English, French and Spanish: questions and conditionals (new in v2).
The emojis column of the candidates is left empty on purpose: the examples shape every label, so
they are written by the author, not by Claude. The author may also add hand-written rows.
Worksheet sentences are later kept out of the training data.

Never overwrites an existing worksheet, so manual edits are safe.
"""

import csv
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V1_WORKSHEET = REPO / "03_dataset" / "label_examples.csv"
POOL = REPO / "08_v2_spanish_and_eval_sets" / "sets" / "real_pool.csv"
OUT_FILE = HERE / "label_examples_v2.csv"
SEED = 42
FIELDS = ["type", "lang", "sentence", "emojis", "source"]
VAGUE_ES = re.compile(r"\b(creo|sé|pienso|siento|idea|segur[oa]|entiendo|imagino|pregunto)\b", re.I)
SPECIAL = {"question", "conditional", "speech", "number"}

# case -> (languages, candidates per language, test on a pool row)
CASES = {
    "everyday": (("es",), 6, lambda r, t: ("long" in t or "medium" in t) and not t & SPECIAL),
    "vague": (("es",), 4, lambda r, t: not t & SPECIAL and VAGUE_ES.search(r["sentence"])),
    "speech": (("es",), 4, lambda r, t: "speech" in t and "question" not in t),
    "numbers": (("es",), 4, lambda r, t: "number" in t and "question" not in t),
    "question": (("en", "fr", "es"), 3, lambda r, t: "question" in t),
    "conditional": (("en", "fr", "es"), 3, lambda r, t: "conditional" in t and "question" not in t),
}


def main() -> None:
    if OUT_FILE.exists():
        raise SystemExit(f"{OUT_FILE.name} already exists: not overwriting it.")
    with open(V1_WORKSHEET, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(POOL, newline="", encoding="utf-8") as f:
        pool = list(csv.DictReader(f))
    random.Random(SEED).shuffle(pool)
    used = set()
    for case, (langs, count, test) in CASES.items():
        for lang in langs:
            authors, picked = set(), 0
            for r in pool:
                if picked == count:
                    break
                tags = set(r["tags"].split())
                if (r["lang"] == lang and r["id"] not in used and r["author"] not in authors
                        and 5 <= len(r["sentence"].split()) <= 16 and test(r, tags)):
                    used.add(r["id"])
                    authors.add(r["author"])
                    picked += 1
                    rows.append({"type": case, "lang": lang, "sentence": r["sentence"], "emojis": "",
                                 "source": f"Tatoeba ({r['author']}), CC BY 2.0 FR"})
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_FILE.name}: {len(rows)} rows, {sum(bool(r['emojis'].strip()) for r in rows)} with emojis")


if __name__ == "__main__":
    main()
