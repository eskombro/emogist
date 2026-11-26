"""Step 9b: draw the real training sentences from the step 8 pool, by plain rules.

    python 09_v2_dataset_and_labels/draw_real_train.py

From 08_v2_spanish_and_eval_sets/sets/real_pool.csv (already in a fixed shuffled order, without v1
train/dev/test). Left out: pool sentences used in the v2 dev or test sets (replacements), anything
that is a near-duplicate of a dev or test sentence (v1 and v2, real and AI), and the worksheet
sentences of the labelling prompt.

Quotas per language, walked in pool order (nobody picks sentences): questions up to 20% (every
available one where there are fewer, author's decision), long (9+ words) 30%, and other sentences
fill the rest. At most 10% of a language's sentences from one Tatoeba author, except for questions,
which come from few authors and are all taken (author's decision). 3% margin for
sentences that later get no valid label. Output: sentences/real_train.csv.
"""

import csv
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STEP8 = REPO / "08_v2_spanish_and_eval_sets"
sys.path.insert(0, str(STEP8))
import generate_ai_v2 as g2  # noqa: E402
from rules_v2 import v1  # noqa: E402

OUT = HERE / "sentences" / "real_train.csv"
SIZE = 1000  # per language, after labelling
MARGIN = 1.03
SHARES = {"question": 0.20, "long": 0.30}  # other = the rest
MAX_AUTHOR_SHARE = 0.10


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bucket(row: dict) -> str:
    tags = set(row["tags"].split())
    return "question" if "question" in tags else "long" if "long" in tags else "other"


def main() -> None:
    pool = read(STEP8 / "sets" / "real_pool.csv")
    eval_rows = read(STEP8 / "sets" / "dev.csv") + read(STEP8 / "sets" / "test.csv")
    eval_ids = {r["id"] for r in eval_rows}
    worksheet = {r["sentence"].strip() for r in read(HERE / "label_examples_v2.csv")}
    total = math.ceil(SIZE * MARGIN)
    out, report = [], {}
    for lang in g2.LANGS:
        hidden = g2.held_out(lang)  # v1 and v2 real dev/test
        for r in eval_rows:
            if r["lang"] == lang:
                hidden.add(set(v1.words(r["sentence"])))  # v2 AI dev/test too
        candidates, dropped = [], Counter()
        for r in pool:
            if r["lang"] != lang:
                continue
            if r["id"] in eval_ids:
                dropped["in dev/test"] += 1
            elif r["sentence"].strip() in worksheet:
                dropped["worksheet"] += 1
            elif hidden.seen(set(v1.words(r["sentence"]))):
                dropped["near-duplicate of dev/test"] += 1
            else:
                candidates.append(r)
        available = Counter(bucket(r) for r in candidates)
        quotas = {b: min(available[b], math.ceil(share * total)) for b, share in SHARES.items()}
        quotas["other"] = total - sum(quotas.values())
        filled, authors, chosen = Counter(), Counter(), []
        for r in candidates:
            b = bucket(r)
            # Questions ignore the author cap (author's decision: take every available question);
            # they still count toward each author's total.
            if filled[b] < quotas[b] and (b == "question" or authors[r["author"]] < MAX_AUTHOR_SHARE * total):
                filled[b] += 1
                authors[r["author"]] += 1
                chosen.append(r)
        # Questions come from few authors, so the cap can leave the quota short: the rest is filled
        # with other sentences, in the same order (author's decision for the question shortage).
        taken = {r["id"] for r in chosen}
        for r in candidates:
            if len(chosen) >= total:
                break
            if (r["id"] not in taken and bucket(r) == "other"
                    and authors[r["author"]] < MAX_AUTHOR_SHARE * total):
                filled["other (filling)"] += 1
                authors[r["author"]] += 1
                chosen.append(r)
        chosen.sort(key=lambda r: candidates.index(r))
        if len(chosen) < total:
            raise SystemExit(f"[{lang}] only {len(chosen)} of {total}: {dict(filled)}")
        out += chosen
        report[lang] = (dict(dropped), dict(filled))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pool[0]))
        writer.writeheader()
        writer.writerows(out)
    for lang, (dropped, filled) in report.items():
        print(f"[{lang}] left out: {dropped}; drawn: {filled}")
    print(f"Wrote {OUT.name}: {len(out)} sentences")


if __name__ == "__main__":
    main()
