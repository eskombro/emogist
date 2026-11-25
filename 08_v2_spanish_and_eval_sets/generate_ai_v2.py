"""Step 8b: AI-generated everyday sentences for v2, in English, French and Spanish.

    python 08_v2_spanish_and_eval_sets/generate_ai_v2.py --per-pattern 16 --out sets/ai_eval.csv
    python 08_v2_spanish_and_eval_sets/generate_ai_v2.py --dry-run es   # print 3 Spanish prompts, no calls

Step 3's generator (03_dataset/generate_sentences.py) is reused for English and French and
stays unchanged. v2 adds Spanish, two patterns (question, conditional), asks every language
for no slang or crude words, and alternates two open generators on Ollama (DeepSeek V4.1
Flash and gpt-oss 120B) for more variety. Claude never writes these sentences, and the
examples the generators see are real Tatoeba sentences picked by plain rules.

Each request gets one combination of topic, pattern, tone, length and speaker, from a fixed
shuffled plan per language and pattern. Resumable: ids already in the output file are skipped.
"""

import argparse
import csv
import itertools
import random
import re
import sys
from collections import Counter
from pathlib import Path

import requests

import rules_v2
from rules_v2 import v1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_dataset"))
import generate_sentences as g1  # noqa: E402  (step 3)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEED = 42
LANGS = ("en", "fr", "es")

TOPICS = {
    **g1.TOPICS,
    # Neutral wording on purpose: no Spain-only or Latin-America-only words (móvil/celular, coche/carro, piso).
    "es": ["la cocina", "el transporte público", "una mascota", "la oficina", "la escuela", "un cumpleaños",
           "ir al supermercado", "el gimnasio", "un viaje por carretera", "un vecino", "una visita al médico",
           "una mudanza", "una boda", "el clima", "una cita", "los videojuegos", "la jardinería", "un concierto",
           "un teléfono que se rompe", "una cena familiar", "ir de campamento", "un restaurante",
           "las tareas de la casa", "unas vacaciones en la playa", "una entrevista de trabajo", "la peluquería",
           "un vehículo averiado", "cuidar niños", "un vuelo retrasado", "alguien ruidoso con quien compartes casa",
           "aprender a conducir", "un partido de fútbol", "una cartera perdida", "hacer un pastel",
           "una cita con el dentista", "un apagón", "una ruta de senderismo", "la visita a un museo",
           "las compras por internet", "una fiesta sorpresa", "un paseo en bicicleta", "pagar las cuentas",
           "un corte de pelo que sale mal", "un reencuentro de antiguos compañeros", "una mañana de nieve",
           "una fuga de agua", "comer algo a medianoche", "una clase de idiomas"],
}
PATTERNS = {
    name: {**texts, "es": es} for (name, texts), es in zip(g1.PATTERNS.items(), [
        "alguien dice, escribe o cuenta algo a alguien. La frase DEBE contener un verbo de habla (dijo, preguntó, escribió...) Y lo que se dijo",
        "algo que pasó antes explica la situación actual",
        "un plan, o algo que todavía NO ha pasado. El verbo principal DEBE referirse al futuro (voy a, vamos a, mañana, la semana que viene...)",
        "algo sale mal, seguido de su consecuencia",
        "un pequeño momento concreto de la vida diaria",
        "un número exacto importa en la historia (una edad, un precio, una duración, una hora o una cantidad). La frase DEBE contener ese número",
    ])
}
PATTERNS["question"] = {
    "en": "someone asks a question about a concrete everyday situation. The sentence MUST be a question and end with ?",
    "fr": "quelqu'un pose une question sur une situation concrète du quotidien. La phrase DOIT être une question et finir par ?",
    "es": "alguien hace una pregunta sobre una situación concreta del día a día. La frase DEBE ser una pregunta, empezar con ¿ y terminar con ?",
}
PATTERNS["conditional"] = {
    "en": "what would happen, or will happen, if a condition is met. The sentence MUST contain \"if\"",
    "fr": "ce qui arriverait, ou arrivera, si une condition est remplie. La phrase DOIT contenir « si »",
    "es": "lo que pasaría, o pasará, si se cumple una condición. La frase DEBE contener «si»",
}
V2_PATTERNS = ("speech", "past", "future", "mishap", "moment", "numbers", "question", "conditional")
TONES = {**g1.TONES, "es": ["divertido", "molesto", "contento", "triste", "aliviado", "avergonzado", "orgulloso"]}
SPEAKERS = {**g1.SPEAKERS, "es": ["en primera persona (yo)", "en primera persona del plural (nosotros)",
                                  "sobre alguien cercano (un familiar, un amigo, un compañero)"]}
NO_SLANG = {
    "en": " No slang, crude words or regional expressions.",
    "fr": " Pas d'argot, de mots grossiers ni d'expressions régionales.",
    "es": " Usa un español neutro: sin jerga, palabras malsonantes ni expresiones regionales.",
}
PROMPT = {
    "en": g1.PROMPT["en"].replace(" Reply with", NO_SLANG["en"] + " Reply with"),
    "fr": g1.PROMPT["fr"].replace(" Réponds", NO_SLANG["fr"] + " Réponds"),
    "es": "Escribe una sola frase en español, como la diría una persona real o como se la escribiría a un amigo.\n"
          "Tema: {topic}\nSituación: {pattern}\nEmoción: {tone}\nLongitud: de {low} a {high} palabras\nPersona: {speaker}\n"
          "Frases reales con esta situación (imita la estructura, nunca el tema):\n{examples}\n"
          "Sé concreto y preciso. Haz sentir la emoción a través de lo que pasa, sin nombrarla. Sin emojis, comillas, hashtags ni nombres famosos."
          + NO_SLANG["es"] + " Responde solo con la frase.",
}
assert all(NO_SLANG[lang] in PROMPT[lang] for lang in LANGS)

EXAMPLE_RULES = {
    name: {**rules, "es": es} for (name, rules), es in zip(g1.EXAMPLE_RULES.items(), [
        r"\b(me dijo|le dije|dijo que|me preguntó|le pregunté|me escribió|anunció)\b",
        r"\b(porque|ya que|desde que)\b",
        r"\b(voy a|vamos a|va a|vas a|mañana|la semana que viene|la próxima semana)\b",
        r"\b(así que|entonces|por eso)\b",
        r"\b(yo|me|mi|nos)\b",
        r"\b\d{1,3}\b|\b(dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|veinte|"
        r"treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien|ciento)\b",
    ])
}
EXAMPLE_RULES["question"] = {lang: r"\?\s*$" for lang in LANGS}
EXAMPLE_RULES["conditional"] = {lang: r"^\W*(if|si)\b|, (if|si)\b" for lang in LANGS}
# What the generated sentence must show for its pattern (plain checks, as for numbers in step 3).
REQUIRED = {"numbers": "number", "question": "question", "conditional": "conditional"}  # values: rules_v2.tags names
assert all(tag in ("number", "question", "conditional") for tag in REQUIRED.values())

FIELDS = ["id", "lang", "sentence", "topic", "pattern", "tone", "length", "speaker", "generator"]


def plan(lang: str, pattern: str) -> list[tuple]:
    cells = list(itertools.product(range(len(TOPICS[lang])), [pattern], range(len(TONES[lang])), g1.LENGTHS,
                                   range(len(SPEAKERS[lang]))))
    random.Random(f"v2-{SEED}-{lang}-{pattern}").shuffle(cells)
    return cells


def held_out(lang: str) -> v1.NearDuplicates:
    """Every dev and test sentence, v1 and v2: the generator must never see or copy them."""
    seen = v1.NearDuplicates()
    sources = [REPO / "data" / "test_sentences.csv", REPO / "03_dataset" / "sentences" / "real_dev.csv",
               HERE / "sets" / "real_dev.csv", HERE / "sets" / "real_test.csv"]
    for source in sources:
        with open(source, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["lang"] == lang:
                    seen.add(set(v1.words(row["sentence"])))
    return seen


def example_pools(lang: str) -> dict[str, list[str]]:
    hidden = held_out(lang)
    with open(HERE / "raw" / f"everyday_{lang}.tsv", encoding="utf-8") as f:
        candidates = [line.rstrip("\n").split("\t", 1) for line in f]
    pools = {}
    for name in V2_PATTERNS:
        rule, chosen, authors = re.compile(EXAMPLE_RULES[name][lang], re.I), [], Counter()
        for author, sentence in candidates:
            if len(chosen) == g1.EXAMPLE_POOL_SIZE:
                break
            if (authors[author] < g1.MAX_EXAMPLES_PER_AUTHOR and rule.search(sentence)
                    and 6 <= len(v1.words(sentence)) <= 20 and not hidden.seen(set(v1.words(sentence)))):
                authors[author] += 1
                chosen.append(sentence)
        pools[name] = chosen
    return pools


def prompt_for(lang: str, cell: tuple, index: int, pools: dict[str, list[str]]) -> str:
    topic, pattern, tone, length, speaker = cell
    low, high = g1.LENGTHS[length]
    picks = random.Random(f"v2-{SEED}-{lang}-{pattern}-{index}").sample(pools[pattern], g1.EXAMPLES_PER_REQUEST)
    return PROMPT[lang].format(topic=TOPICS[lang][topic], pattern=PATTERNS[pattern][lang], tone=TONES[lang][tone],
                               low=low, high=high, speaker=SPEAKERS[lang][speaker],
                               examples="\n".join(f"- {s}" for s in picks))


# Two open generators for more variety, alternating by plan cell: (model, think, max tokens).
# gpt-oss always reasons first, so it gets the lowest level and room for its thinking (as in step 2).
GENERATORS = [(g1.MODEL, False, 120), ("gpt-oss:120b-cloud", "low", 2000)]


def generator_for(index: int) -> tuple:
    return GENERATORS[index % len(GENERATORS)]


def generate(content: str, generator: tuple) -> str:
    model, think, max_tokens = generator
    response = requests.post(g1.URL, timeout=180, json={
        "model": model, "messages": [{"role": "user", "content": content}], "stream": False, "think": think,
        "options": {"temperature": g1.TEMPERATURE, "num_predict": max_tokens}})
    response.raise_for_status()
    return response.json()["message"]["content"].strip().strip('"«» ')


def known_sentences(lang: str, out: Path) -> v1.NearDuplicates:
    """Everything a new sentence must not repeat: all dev/test sets, v1 pools, v2 real sentences."""
    seen = held_out(lang)
    sources = [REPO / "03_dataset" / "sentences" / "real_pool.csv", REPO / "03_dataset" / "sentences" / "ai_pool.csv",
               HERE / "sets" / "real_pool.csv"] + ([out] if out.exists() else [])
    for source in sources:
        with open(source, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["lang"] == lang:
                    seen.add(set(v1.words(row["sentence"])))
    return seen


def rejection(sentence: str, lang: str, pattern: str, seen: v1.NearDuplicates) -> str | None:
    if "\n" in sentence or not rules_v2.looks_clean(sentence):
        return "format"
    category = rules_v2.blocked_category(sentence, lang)
    if category:
        return category
    if pattern in REQUIRED and REQUIRED[pattern] not in rules_v2.tags(sentence):
        return f"no {pattern}"
    if seen.seen(set(v1.words(sentence))):
        return "near-duplicate"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-pattern", type=int, default=2, help="accepted sentences per language and pattern")
    parser.add_argument("--langs", nargs="+", default=list(LANGS))
    parser.add_argument("--out", type=Path, default=HERE / "sets" / "ai_eval.csv")
    parser.add_argument("--dry-run", metavar="LANG", help="print 3 prompts for this language and stop")
    args = parser.parse_args()

    if args.dry_run:
        pools = example_pools(args.dry_run)
        print({k: len(v) for k, v in pools.items()})
        for pattern in ("question", "conditional", "speech"):
            print(f"\n--- {pattern} ---\n" + prompt_for(args.dry_run, plan(args.dry_run, pattern)[0], 0, pools))
        return

    existing = []
    if args.out.exists():
        with open(args.out, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    done = {r["id"] for r in existing}
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not existing:
            writer.writeheader()
        for lang in args.langs:
            seen = known_sentences(lang, args.out)
            pools = example_pools(lang)
            print(f"[{lang}] real examples per pattern: { {k: len(v) for k, v in pools.items()} }", flush=True)
            for pool in pools.values():
                for example in pool:
                    seen.add(set(v1.words(example)))
            for pattern in V2_PATTERNS:
                kept = sum(r["lang"] == lang and r["pattern"] == pattern for r in existing)
                tried, rejected = 0, Counter()
                for index, cell in enumerate(plan(lang, pattern)):
                    if kept >= args.per_pattern or tried >= 3 * args.per_pattern + 10:
                        break
                    row_id = f"ai2-{lang}-{pattern}-{index:05d}"
                    if row_id in done:
                        continue
                    tried += 1
                    generator = generator_for(index)
                    try:
                        sentence = generate(prompt_for(lang, cell, index, pools), generator)
                    except requests.RequestException as error:
                        rejected["request error"] += 1
                        print(f"  {row_id}: request failed ({type(error).__name__})", flush=True)
                        continue
                    reason = rejection(sentence, lang, pattern, seen)
                    if reason:
                        rejected[reason] += 1
                        continue
                    seen.add(set(v1.words(sentence)))
                    topic, _, tone, length, speaker = cell
                    writer.writerow({"id": row_id, "lang": lang, "sentence": sentence, "topic": TOPICS[lang][topic],
                                     "pattern": pattern, "tone": TONES[lang][tone], "length": length,
                                     "speaker": SPEAKERS[lang][speaker], "generator": generator[0]})
                    f.flush()
                    kept += 1
                print(f"[{lang}/{pattern}] {kept} sentences, {tried} requests this run, rejected: {dict(rejected)}",
                      flush=True)


if __name__ == "__main__":
    main()
