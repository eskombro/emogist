"""Step 3a: build the pool of real, human-written sentences from Tatoeba.

Downloads the Tatoeba exports (CC BY 2.0 FR), keeps original sentences written by
self-declared native speakers, filters them towards everyday situations about people
(plain rules, no model involved), and writes:
    sentences/real_dev.csv    50 EN + 50 FR, to compare prompts and pilots (never trained on)
    sentences/real_pool.csv   sentences available for training

Free to run: no model is called. The result is reproducible for a given download.
"""

import bz2
import csv
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

from blocklist import blocked_category
from emoji_eval.paths import TEST_FILE

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "sentences"
BASE_URL = "https://downloads.tatoeba.org/exports"
LANGS = {"en": "eng", "fr": "fra"}

SEED = 42
DEV_SIZE = 50  # per language
POOL_SIZE = 3000  # per language: more than the 1,500 needed, since some labels get dropped
MIN_WORDS, MAX_WORDS = 5, 25
MAX_AUTHOR_SHARE = 0.10  # no single contributor dominates the style
MAX_NAME_SHARE = 0.05  # Tatoeba overuses a few character names
MAX_SAME_OPENING = 5  # sentences sharing their first three words
NEAR_DUPLICATE = 0.7  # word overlap (Jaccard) above which two sentences count as the same
STOCK_NAMES = {"tom", "mary", "john", "sami", "layla", "yanni", "skura", "mennad", "baya", "rima", "ziri"}

# A sentence must mention a person, so general facts and abstract remarks are left out.
PERSON_WORDS = {
    "en": {"i", "me", "my", "mine", "myself", "we", "us", "our", "he", "him", "his", "she", "her",
           "mom", "mum", "dad", "mother", "father", "brother", "sister", "son", "daughter", "wife",
           "husband", "friend", "friends", "boss", "neighbor", "neighbour", "kids", "children"},
    "fr": {"je", "j", "me", "m", "moi", "mon", "ma", "mes", "nous", "notre", "nos", "on", "il",
           "elle", "ils", "elles", "lui", "leur", "maman", "papa", "mère", "père", "frère", "sœur",
           "fils", "fille", "femme", "mari", "ami", "amie", "amis", "copain", "copine", "patron",
           "voisin", "voisine", "enfants"},
}
# French "il" is often impersonal ("il faut", "il y a"): those don't describe a person.
IMPERSONAL_FR = re.compile(r"^il (ne |n['’])?(faut|y a|y avait|y aura|s['’]agit|semble|paraît|pleut|neige)\b", re.I)
MAX_PROPER_NOUNS = 1  # several capitalised names usually mean an encyclopedic sentence

WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+")


def download(filename: str) -> Path:
    path = RAW_DIR / Path(filename).name
    if path.exists():
        return path
    RAW_DIR.mkdir(exist_ok=True)
    print(f"Downloading {filename}...")
    with requests.get(f"{BASE_URL}/{filename}", stream=True, timeout=60) as response:
        response.raise_for_status()
        partial = path.with_suffix(path.suffix + ".part")
        with open(partial, "wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
    partial.rename(path)
    return path


def tar_lines(path: Path):
    """Read the single text file of a .tar.bz2 without extracting it to disk."""
    with tarfile.open(path, "r|bz2") as archive:
        for member in archive:
            if member.isfile():
                for line in archive.extractfile(member):
                    yield line.decode("utf-8").rstrip("\n").split("\t")


def words(sentence: str) -> list[str]:
    return [w.lower() for w in WORD.findall(sentence)]


def looks_clean(sentence: str) -> bool:
    if not MIN_WORDS <= len(words(sentence)) <= MAX_WORDS:
        return False
    if not sentence[0].isupper() or sentence.rstrip("\"”»")[-1] not in ".!?…":
        return False
    if re.search(r"https?:|www\.|[@#<>\[\]{}|_\\]", sentence):
        return False
    if any(c.isalpha() and ord(c) > 0x24F for c in sentence):  # non-Latin script
        return False
    return not sentence.isupper()


def is_everyday(sentence: str, lang: str) -> bool:
    """Rule-based and imperfect: keep statements about people doing or feeling something."""
    if sentence.rstrip("\"”»").endswith("?"):
        return False
    if re.search(r"\b\d{4}\b", sentence):  # years and long numbers are mostly factual
        return False
    tokens = re.findall(r"[^\W\d_]+", sentence)
    if sum(t[0].isupper() and t != "I" for t in tokens[1:]) > MAX_PROPER_NOUNS:
        return False
    if lang == "fr" and IMPERSONAL_FR.match(sentence):
        return False
    return bool({t.lower() for t in tokens} & PERSON_WORDS[lang])


class NearDuplicates:
    """Finds sentences that share most of their words with one already seen."""

    def __init__(self):
        self.sets, self.index = [], defaultdict(list)

    def seen(self, tokens: set[str]) -> bool:
        shared = Counter(i for t in tokens for i in self.index[t])
        return any(n / len(tokens | self.sets[i]) >= NEAR_DUPLICATE for i, n in shared.items())

    def add(self, tokens: set[str]) -> None:
        for t in tokens:
            self.index[t].append(len(self.sets))
        self.sets.append(tokens)


def select(candidates: list[dict], test_sentences: list[str], size: int) -> tuple[list[dict], Counter]:
    """Walk through shuffled candidates, keeping those that respect the diversity limits."""
    random.Random(SEED).shuffle(candidates)
    duplicates = NearDuplicates()
    for sentence in test_sentences:
        duplicates.add(set(words(sentence)))
    kept, rejected = [], Counter()
    authors, openings = Counter(), Counter()
    named = 0
    for c in candidates:
        if len(kept) == size:
            break
        tokens = words(c["sentence"])
        token_set = set(tokens)
        opening = " ".join(tokens[:3])
        has_name = bool(token_set & STOCK_NAMES)
        if authors[c["author"]] >= size * MAX_AUTHOR_SHARE:
            rejected["author cap"] += 1
        elif has_name and named >= size * MAX_NAME_SHARE:
            rejected["stock name cap"] += 1
        elif openings[opening] >= MAX_SAME_OPENING:
            rejected["same opening"] += 1
        elif duplicates.seen(token_set):
            rejected["near-duplicate"] += 1
        else:
            duplicates.add(token_set)
            authors[c["author"]] += 1
            openings[opening] += 1
            named += has_name
            kept.append(c)
    return kept, rejected


def main() -> None:
    base_path = download("sentences_base.tar.bz2")
    users_path = download("user_languages.tar.bz2")
    detailed = {lang: download(f"per_language/{code}/{code}_sentences_detailed.tsv.bz2")
                for lang, code in LANGS.items()}

    natives = {lang: set() for lang in LANGS}
    code_to_lang = {code: lang for lang, code in LANGS.items()}
    for fields in tar_lines(users_path):
        if len(fields) >= 3 and fields[0] in code_to_lang and fields[1] == "5":
            natives[code_to_lang[fields[0]]].add(fields[2])

    originals = {fields[0] for fields in tar_lines(base_path) if len(fields) >= 2 and fields[1] == "0"}

    with open(TEST_FILE, newline="", encoding="utf-8") as f:
        test = list(csv.DictReader(f))

    dev, pool = [], []
    for lang, path in detailed.items():
        total = original = native = clean = 0
        candidates, unique = [], set()
        blocked = Counter()
        blocked_examples = []
        with bz2.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    continue
                sentence_id, _, text, author = fields[:4]
                total += 1
                if sentence_id not in originals:
                    continue
                original += 1
                if author not in natives[lang]:
                    continue
                native += 1
                key = " ".join(words(text))
                if not looks_clean(text) or key in unique:
                    continue
                unique.add(key)
                clean += 1
                if not is_everyday(text, lang):
                    continue
                category = blocked_category(text, lang)
                if category:
                    blocked[category] += 1
                    blocked_examples.append(f"{category}\t{text}")
                else:
                    candidates.append({"id": f"tatoeba-{sentence_id}", "lang": lang,
                                       "sentence": text, "author": author})
        kept, rejected = select(candidates, [r["sentence"] for r in test if r["lang"] == lang],
                                DEV_SIZE + POOL_SIZE)
        # All everyday candidates, used by the generator as a source of real example sentences.
        with open(RAW_DIR / f"everyday_{lang}.tsv", "w", encoding="utf-8") as out:
            out.writelines(f"{c['author']}\t{c['sentence']}\n" for c in candidates)
        dev += kept[:DEV_SIZE]
        pool += kept[DEV_SIZE:]
        print(f"[{lang}] {total:,} sentences > {original:,} original > {native:,} by natives > "
              f"{clean:,} clean > {len(candidates):,} everyday > {len(kept):,} kept")
        print(f"     blocklist removed: {dict(blocked)}")
        (RAW_DIR / f"blocked_{lang}.tsv").write_text("\n".join(blocked_examples), encoding="utf-8")
        print(f"     rejected while selecting: {dict(rejected)}; authors: {len({c['author'] for c in kept})}")

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in (("real_dev.csv", dev), ("real_pool.csv", pool)):
        with open(OUT_DIR / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "lang", "sentence", "author"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {name}: {len(rows)} sentences")


if __name__ == "__main__":
    main()
