"""Step 8a: real Tatoeba sentences for the v2 dev and test sets, in English, French and Spanish.

    python 08_v2_spanish_and_eval_sets/fetch_real_v2.py

Same pipeline as step 3 (original sentences by self-declared native speakers, everyday
filters, diversity limits), with the v2 rules from rules_v2.py: Spanish, and questions allowed.
Sentences from v1 training, v1 dev and v1 test, and their near-duplicates, are left out.

Writes, in sets/:
    real_dev.csv, real_test.csv   50 per language each, drawn in the shuffled order with quotas
    real_pool.csv                 the remaining selected sentences, for the v2 training data
and raw/everyday_{lang}.tsv: everyday candidates minus dev and test, as generator examples.

Free to run: no model is called. Reproducible for a given download (raw/ holds the snapshot).
"""

import bz2
import csv
import json
from collections import Counter
from pathlib import Path

import requests

import rules_v2
from rules_v2 import LANGS, v1

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW_DIR = HERE / "raw"
SETS_DIR = HERE / "sets"
POOL_SIZE = 3000  # per language, before dev and test are taken out
PER_SET = 50  # real sentences per language in each of dev and test
QUOTAS = {"question": 10, "long": 15, "other": 25}  # per language and set; long = 9+ words
MAX_AUTHOR_PER_SET = 5  # 10% of a language's real sentences in one set
FIELDS = ["id", "lang", "source", "sentence", "author", "tags"]


def download(filename: str) -> Path:
    path = RAW_DIR / Path(filename).name
    if path.exists():
        return path
    RAW_DIR.mkdir(exist_ok=True)
    print(f"Downloading {filename}...")
    with requests.get(f"{v1.BASE_URL}/{filename}", stream=True, timeout=60) as response:
        response.raise_for_status()
        partial = path.with_suffix(path.suffix + ".part")
        with open(partial, "wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
    partial.rename(path)
    return path


def v1_sentences() -> tuple[dict[str, list[str]], set[str]]:
    """Every sentence the v1 model was trained, tuned or tested on, by language, and their ids."""
    texts, ids = {lang: [] for lang in LANGS}, set()
    with open(REPO / "03_dataset" / "full" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts[row["lang"]].append(row["sentence"])
            ids.add(row["id"])
    for path in (REPO / "03_dataset" / "sentences" / "real_dev.csv", REPO / "data" / "test_sentences.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                texts[row["lang"]].append(row["sentence"])
                ids.add(row["id"])
    return texts, ids


def bucket(sentence: str) -> str:
    found = rules_v2.tags(sentence)
    return "question" if "question" in found else "long" if "long" in found else "other"


def assign(kept: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk the shuffled order and fill dev, then test, respecting quotas and the author cap."""
    sets = []
    rest = list(kept)
    for _ in ("dev", "test"):
        filled, authors, chosen, left = Counter(), Counter(), [], []
        for c in rest:
            b = bucket(c["sentence"])
            if filled[b] < QUOTAS[b] and authors[c["author"]] < MAX_AUTHOR_PER_SET:
                filled[b] += 1
                authors[c["author"]] += 1
                chosen.append(c)
            else:
                left.append(c)
        if len(chosen) < PER_SET:
            raise SystemExit(f"Only {len(chosen)} real sentences for {kept[0]['lang']}: {dict(filled)}")
        sets.append(chosen)
        rest = left
    return sets[0], sets[1], rest


def write(name: str, rows: list[dict]) -> None:
    with open(SETS_DIR / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote sets/{name}: {len(rows)} sentences")


def main() -> None:
    base_path = download("sentences_base.tar.bz2")
    users_path = download("user_languages.tar.bz2")
    detailed = {lang: download(f"per_language/{code}/{code}_sentences_detailed.tsv.bz2") for lang, code in LANGS.items()}

    natives = {lang: set() for lang in LANGS}
    code_to_lang = {code: lang for lang, code in LANGS.items()}
    for fields in v1.tar_lines(users_path):
        if len(fields) >= 3 and fields[0] in code_to_lang and fields[1] == "5":
            natives[code_to_lang[fields[0]]].add(fields[2])
    originals = {fields[0] for fields in v1.tar_lines(base_path) if len(fields) >= 2 and fields[1] == "0"}
    excluded_texts, excluded_ids = v1_sentences()

    dev, test, pool = [], [], []
    for lang, path in detailed.items():
        counts, blocked = Counter(), Counter()
        candidates, unique = [], set()
        with bz2.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    continue
                sentence_id, _, text, author = fields[:4]
                counts["total"] += 1
                if sentence_id not in originals or author not in natives[lang]:
                    continue
                counts["original by natives"] += 1
                key = " ".join(v1.words(text))
                if not rules_v2.looks_clean(text) or key in unique:
                    continue
                unique.add(key)
                counts["clean"] += 1
                if not rules_v2.is_everyday(text, lang):
                    continue
                counts["everyday"] += 1
                if f"tatoeba-{sentence_id}" in excluded_ids:
                    counts["used in v1"] += 1
                    continue
                category = rules_v2.blocked_category(text, lang)
                if category:
                    blocked[category] += 1
                    continue
                candidates.append({"id": f"tatoeba-{sentence_id}", "lang": lang, "source": "real",
                                   "sentence": text, "author": author})
        kept, rejected = v1.select(candidates, excluded_texts[lang], POOL_SIZE)
        lang_dev, lang_test, lang_pool = assign(kept)
        for row in lang_dev + lang_test + lang_pool:
            row["tags"] = " ".join(rules_v2.tags(row["sentence"]))
        dev += lang_dev
        test += lang_test
        pool += lang_pool
        held_out = {r["sentence"] for r in lang_dev + lang_test}
        with open(RAW_DIR / f"everyday_{lang}.tsv", "w", encoding="utf-8") as out:
            out.writelines(f"{c['author']}\t{c['sentence']}\n" for c in candidates if c["sentence"] not in held_out)
        print(f"[{lang}] " + " > ".join(f"{n:,} {k}" for k, n in counts.items()) + f" > {len(kept):,} kept")
        print(f"     blocklist removed: {dict(blocked)}; selection: {dict(rejected)}")
        print(f"     dev {len(lang_dev)}, test {len(lang_test)}, pool {len(lang_pool)}; "
              f"pool questions {sum(bucket(r['sentence']) == 'question' for r in lang_pool)}, "
              f"long {sum(bucket(r['sentence']) == 'long' for r in lang_pool)}")

    SETS_DIR.mkdir(exist_ok=True)
    write("real_dev.csv", dev)
    write("real_test.csv", test)
    write("real_pool.csv", pool)


if __name__ == "__main__":
    main()
