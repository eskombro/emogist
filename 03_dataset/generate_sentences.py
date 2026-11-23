"""Step 3b: generate everyday sentences to complement the real pool.

Each request asks for one sentence from a different combination of topic, pattern, tone,
length and speaker, so the set doesn't collapse into the same few ideas. French is written
directly in French, never translated. The generator is an open model, not Claude, because
these sentences become training data. The examples shown to it are real Tatoeba sentences
from the pool, picked by plain rules, so no Claude-written text shapes the training data.

    python 03_dataset/generate_sentences.py --per-lang 20 --out /some/trial.csv
    python 03_dataset/generate_sentences.py --per-lang 665 --numbers-per-lang 170

Resumable: combinations already recorded in the output file are skipped.
"""

import argparse
import csv
import itertools
import random
import re
from collections import Counter
from pathlib import Path

import requests

from blocklist import blocked_category
from emoji_eval.paths import TEST_FILE
from fetch_real_sentences import NearDuplicates, OUT_DIR, RAW_DIR, looks_clean, words

MODEL = "deepseek-v4.1-flash:cloud"
URL = "http://localhost:11434/api/chat"
TEMPERATURE = 0.9  # high on purpose: here we want variety, unlike when labelling
SEED = 42

TOPICS = {
    "en": ["cooking", "public transport", "a pet", "the office", "school", "a birthday", "shopping", "the gym",
           "a road trip", "a neighbor", "a doctor's visit", "moving house", "a wedding", "the weather", "a date",
           "video games", "gardening", "a concert", "a phone that breaks", "a family dinner", "camping", "a restaurant",
           "housework", "a holiday at the beach", "a job interview", "a haircut", "a broken car", "babysitting",
           "a flight delay", "a noisy flatmate", "learning to drive", "a football match", "a lost wallet",
           "baking a cake", "a dentist appointment", "a power cut", "a hiking trail", "a museum visit",
           "online shopping", "a surprise party", "a bike ride", "paying the bills", "a new haircut gone wrong",
           "a school reunion", "a snowy morning", "fixing a leak", "a late-night snack", "a language class"],
    "fr": ["la cuisine", "les transports en commun", "un animal de compagnie", "le bureau", "l'école", "un anniversaire",
           "les courses", "la salle de sport", "un trajet en voiture", "un voisin", "un rendez-vous chez le médecin",
           "un déménagement", "un mariage", "la météo", "un rendez-vous amoureux", "les jeux vidéo", "le jardinage",
           "un concert", "un téléphone qui tombe en panne", "un repas de famille", "le camping", "un restaurant",
           "le ménage", "des vacances à la plage", "un entretien d'embauche", "le coiffeur", "une voiture en panne",
           "le baby-sitting", "un vol retardé", "un colocataire bruyant", "l'auto-école", "un match de foot",
           "un portefeuille perdu", "la préparation d'un gâteau", "un rendez-vous chez le dentiste",
           "une coupure de courant", "une randonnée", "la visite d'un musée", "les achats en ligne",
           "une fête surprise", "une balade à vélo", "les factures", "une coupe de cheveux ratée",
           "des retrouvailles d'anciens élèves", "un matin de neige", "une fuite d'eau", "un grignotage de nuit",
           "un cours de langue"],
}
PATTERNS = {
    "speech": {"en": "someone says, texts or tells someone something. The sentence MUST contain a verb of saying (said, told, asked, texted...) AND what was said",
               "fr": "quelqu'un dit, écrit ou annonce quelque chose à quelqu'un. La phrase DOIT contenir un verbe de parole (a dit, a demandé, a écrit...) ET ce qui a été dit"},
    "past": {"en": "something that happened earlier explains the situation now",
             "fr": "quelque chose qui s'est passé plus tôt explique la situation actuelle"},
    "future": {"en": "a plan, or something that has NOT happened yet. The main verb MUST be about the future (will, going to, tomorrow, next week...)",
               "fr": "un projet, ou quelque chose qui n'est PAS encore arrivé. Le verbe principal DOIT porter sur le futur (je vais, on va, demain, la semaine prochaine...)"},
    "mishap": {"en": "something goes wrong, followed by its consequence",
               "fr": "quelque chose tourne mal, suivi de sa conséquence"},
    "moment": {"en": "a small concrete moment of daily life", "fr": "un petit moment concret de la vie quotidienne"},
    "numbers": {"en": "an exact number matters to the story (an age, a price, a duration, a time of day or a quantity). The sentence MUST contain that number",
                "fr": "un nombre exact compte dans l'histoire (un âge, un prix, une durée, une heure ou une quantité). La phrase DOIT contenir ce nombre"},
}
# The original grid keeps its own shuffled order, so existing IDs still match their cells.
BASE_PATTERNS = ("speech", "past", "future", "mishap", "moment")
TONES = {
    "en": ["amused", "annoyed", "happy", "sad", "relieved", "embarrassed", "proud"],
    "fr": ["amusé", "agacé", "content", "triste", "soulagé", "gêné", "fier"],
}
LENGTHS = {"short": (6, 10), "medium": (11, 16), "long": (17, 24)}
SPEAKERS = {
    "en": ["first person (I)", "first person plural (we)", "about someone close (a relative, a friend, a colleague)"],
    "fr": ["à la première personne (je)", "avec « on » ou « nous »", "à propos d'un proche (famille, ami, collègue)"],
}
PROMPT = {
    "en": "Write one natural English sentence, the way a real person would say it or text it to a friend.\n"
          "Topic: {topic}\nPattern: {pattern}\nFeeling: {tone}\nLength: {low} to {high} words\nPerson: {speaker}\n"
          "Real sentences with this pattern (copy the structure only, never their subject):\n{examples}\n"
          "Be concrete and specific. Show the feeling through what happens, without naming it. No emojis, quotation marks, hashtags or famous names. "
          "Reply with the sentence only.",
    "fr": "Écris une seule phrase en français courant, comme une vraie personne la dirait ou l'enverrait à un ami.\n"
          "Sujet : {topic}\nSituation : {pattern}\nÉmotion : {tone}\nLongueur : {low} à {high} mots\nPersonne : {speaker}\n"
          "Vraies phrases avec cette situation (imite la structure, jamais le sujet) :\n{examples}\n"
          "Sois concret et précis. Fais sentir l'émotion par ce qui se passe, sans la nommer. Pas d'emoji, de guillemets, de hashtags ni de noms célèbres. "
          "Réponds uniquement avec la phrase.",
}
# Plain rules that find real sentences showing each pattern, to use as examples.
EXAMPLE_RULES = {
    "speech": {"en": r"\b(told|said|asked|texted)\b", "fr": r"\b(m['’]a dit|a dit|lui ai dit|m['’]a demandé|a annoncé|m['’]a écrit)\b"},
    "past": {"en": r"\bbecause\b.*\b(had|was|were|did|forgot|lost|left)\b", "fr": r"\b(parce qu|car|depuis que)\b"},
    "future": {"en": r"\b(going to|tomorrow|next (week|month|year))\b|\b(I|we|he|she)(['’]ll| will)\b",
               "fr": r"\b(je vais|on va|nous allons|il va|elle va|demain|la semaine prochaine)\b"},
    "mishap": {"en": r"\b(so|and now)\b", "fr": r"\b(du coup|alors|donc)\b"},
    "moment": {"en": r"\b(I|we)\b", "fr": r"\b(je|j['’]|on)\b"},
    "numbers": {"en": r"\b\d{1,3}\b|\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b",
                "fr": r"\b\d{1,3}\b|\b(deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingt|trente|quarante|cinquante|soixante|cent)\b"},
}
EXAMPLES_PER_REQUEST = 3
EXAMPLE_POOL_SIZE = 1000  # per pattern and language
MAX_EXAMPLES_PER_AUTHOR = 100  # per pattern, so one contributor's style doesn't dominate

FIELDS = ["id", "lang", "sentence", "topic", "pattern", "tone", "length", "speaker", "generator"]


def plan(lang: str, patterns: tuple[str, ...] = BASE_PATTERNS, salt: str = "") -> list[tuple]:
    """Every combination, in a fixed shuffled order (so a resumed run continues the same plan)."""
    cells = list(itertools.product(range(len(TOPICS[lang])), patterns, range(len(TONES[lang])), LENGTHS,
                                   range(len(SPEAKERS[lang]))))
    random.Random(f"{SEED}-{lang}{salt}").shuffle(cells)
    return cells


def example_pools(lang: str) -> dict[str, list[str]]:
    """Real everyday sentences matching each pattern, 6-20 words long. Never close to the
    dev or test sentences, so those stay unseen by the generator."""
    held_out = NearDuplicates()
    for source in (TEST_FILE, OUT_DIR / "real_dev.csv"):
        with open(source, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["lang"] == lang:
                    held_out.add(set(words(row["sentence"])))
    with open(RAW_DIR / f"everyday_{lang}.tsv", encoding="utf-8") as f:
        candidates = [line.rstrip("\n").split("\t", 1) for line in f]

    pools = {}
    for name, rules in EXAMPLE_RULES.items():
        rule, chosen, authors = re.compile(rules[lang], re.I), [], Counter()
        for author, sentence in candidates:
            if len(chosen) == EXAMPLE_POOL_SIZE:
                break
            if (authors[author] < MAX_EXAMPLES_PER_AUTHOR and rule.search(sentence)
                    and 6 <= len(words(sentence)) <= 20 and not held_out.seen(set(words(sentence)))):
                authors[author] += 1
                chosen.append(sentence)
        pools[name] = chosen
    return pools


def generate(lang: str, cell: tuple, index: int, pools: dict[str, list[str]]) -> str:
    topic, pattern, tone, length, speaker = cell
    low, high = LENGTHS[length]
    # A different draw per request, so the generator doesn't imitate the same three sentences.
    draw = f"{SEED}-{lang}-{index}" if pattern != "numbers" else f"{SEED}-{lang}-numbers-{index}"
    picks = random.Random(draw).sample(pools[pattern], EXAMPLES_PER_REQUEST)
    examples = "\n".join(f"- {s}" for s in picks)
    content = PROMPT[lang].format(topic=TOPICS[lang][topic], pattern=PATTERNS[pattern][lang], tone=TONES[lang][tone],
                                  low=low, high=high, speaker=SPEAKERS[lang][speaker], examples=examples)
    response = requests.post(URL, timeout=120, json={
        "model": MODEL, "messages": [{"role": "user", "content": content}], "stream": False, "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": 120}})
    response.raise_for_status()
    return response.json()["message"]["content"].strip().strip('"«» ')


def known_sentences(lang: str) -> NearDuplicates:
    seen = NearDuplicates()
    sources = [TEST_FILE, OUT_DIR / "real_dev.csv", OUT_DIR / "real_pool.csv"]
    for source in sources:
        with open(source, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["lang"] == lang:
                    seen.add(set(words(row["sentence"])))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-lang", type=int, default=20)
    parser.add_argument("--numbers-per-lang", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "ai_pool.csv")
    args = parser.parse_args()

    existing = []
    if args.out.exists():
        with open(args.out, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    is_new = not existing
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        for lang in ("en", "fr"):
            seen = known_sentences(lang)
            pools = example_pools(lang)
            print(f"[{lang}] real examples per pattern: { {k: len(v) for k, v in pools.items()} }")
            for pool in pools.values():  # a generated sentence must not just copy an example
                for example in pool:
                    seen.add(set(words(example)))
            for r in existing:
                if r["lang"] == lang:
                    seen.add(set(words(r["sentence"])))
            series = [("base", plan(lang), f"ai-{lang}-", args.per_lang),
                      ("numbers", plan(lang, ("numbers",), "-numbers"), f"ai-{lang}-num-", args.numbers_per_lang)]
            for name, cells, prefix, target in series:
                done = {r["id"] for r in existing if r["lang"] == lang and (r["pattern"] == "numbers") == (name == "numbers")}
                kept, tried, rejected = len(done), 0, Counter()
                for index, cell in enumerate(cells):
                    if kept >= target or tried >= 2 * target + 20:
                        break
                    row_id = f"{prefix}{index:05d}"
                    if row_id in done:
                        continue
                    tried += 1
                    try:
                        sentence = generate(lang, cell, index, pools)
                    except requests.RequestException as error:
                        rejected["request error"] += 1
                        print(f"  {row_id}: request failed ({type(error).__name__})")
                        continue
                    tokens = set(words(sentence))
                    reason = ("format" if "\n" in sentence or not looks_clean(sentence)
                              else blocked_category(sentence, lang) or ("near-duplicate" if seen.seen(tokens) else None))
                    if not reason and name == "numbers" and not re.search(EXAMPLE_RULES["numbers"][lang], sentence, re.I):
                        reason = "no number"
                    if reason:
                        rejected[reason] += 1
                        continue
                    seen.add(tokens)
                    topic, pattern, tone, length, speaker = cell
                    writer.writerow({"id": row_id, "lang": lang, "sentence": sentence, "topic": TOPICS[lang][topic],
                                     "pattern": pattern, "tone": TONES[lang][tone], "length": length,
                                     "speaker": SPEAKERS[lang][speaker], "generator": MODEL})
                    f.flush()
                    kept += 1
                print(f"[{lang}/{name}] {kept} sentences, {tried} requests this run, rejected: {dict(rejected)}", flush=True)


if __name__ == "__main__":
    main()
