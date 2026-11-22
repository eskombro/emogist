"""Create label_examples.csv: the worksheet for the labelling prompt's few-shot examples.

It lists the current examples, then real Tatoeba sentences for three hard cases, picked by
plain rules and kept away from the dev, test and training pool sentences. The emojis column
of those candidates is left empty on purpose: the examples shape every label, so they are
written by a person, not by Claude.

Never overwrites an existing worksheet, so manual edits are safe.
"""

import csv
import random
import re
from pathlib import Path

from emoji_eval.paths import TEST_FILE
from emoji_eval.prompt import EXAMPLES
from fetch_real_sentences import NearDuplicates, OUT_DIR, RAW_DIR, STOCK_NAMES, words
from generate_sentences import EXAMPLE_RULES

OUT_FILE = Path(__file__).resolve().parent / "label_examples.csv"
PER_CASE = 4  # candidates per hard case and language
SEED = 42
CASES = {
    "vague": ((5, 9), {"en": r"\b(think|know|sure|mean|understand|believe|feel|idea|wonder)\b",
                       "fr": r"\b(pense|sais|sûre?|comprends|crois|sens|idée|impression)\b"}),
    "speech": ((6, 16), {lang: EXAMPLE_RULES["speech"][lang] for lang in ("en", "fr")}),
    "numbers": ((6, 16), {
        "en": r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|thirty|forty|fifty|hundred)\s+"
              r"(years?|hours?|minutes?|days?|weeks?|months?|dollars?|euros?)\b",
        "fr": r"\b(\d+|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|douze|vingt|trente|quarante|cinquante|cent)\s+"
              r"(ans|heures?|minutes?|jours?|semaines?|mois|euros?|dollars?)\b"}),
}


def held_out(lang: str) -> NearDuplicates:
    seen = NearDuplicates()
    for source in (TEST_FILE, OUT_DIR / "real_dev.csv", OUT_DIR / "real_pool.csv"):
        with open(source, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["lang"] == lang:
                    seen.add(set(words(row["sentence"])))
    return seen


def main() -> None:
    if OUT_FILE.exists():
        raise SystemExit(f"{OUT_FILE.name} already exists: not overwriting it.")
    rows = [{"type": "current", "lang": "en" if i < 6 else "fr", "sentence": text, "emojis": emojis, "source": "existing prompt"}
            for i, (text, emojis) in enumerate(EXAMPLES)]
    for lang in ("en", "fr"):
        seen = held_out(lang)
        with open(RAW_DIR / f"everyday_{lang}.tsv", encoding="utf-8") as f:
            candidates = [line.rstrip("\n").split("\t", 1) for line in f]
        random.Random(f"{SEED}-{lang}").shuffle(candidates)
        for case, ((low, high), rules) in CASES.items():
            rule, authors, picked = re.compile(rules[lang], re.I), set(), 0
            for author, sentence in candidates:
                if picked == PER_CASE:
                    break
                tokens = words(sentence)
                if (author not in authors and low <= len(tokens) <= high and rule.search(sentence)
                        and not set(tokens) & STOCK_NAMES and not seen.seen(set(tokens))):
                    seen.add(set(tokens))
                    authors.add(author)
                    picked += 1
                    rows.append({"type": case, "lang": lang, "sentence": sentence, "emojis": "",
                                 "source": f"Tatoeba ({author}), CC BY 2.0 FR"})
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "lang", "sentence", "emojis", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_FILE.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
