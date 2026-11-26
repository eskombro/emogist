"""Round-trip scorer: how much of the original meaning survives in an emoji sentence.

  1. a fixed decoder (Claude Haiku) turns the emojis back into a sentence, without
     seeing the original - it reads naively, like a demo visitor who knows the rules;
  2. a judge (Claude Sonnet) compares the decoded sentence with the original and gives
     a gist score: 2 = gist kept, 1 = partly, 0 = wrong story. This is the main score.
  3. for reference, a local embedding model also gives a cosine similarity. It rewards
     naming the right objects even when the story is wrong, so it is only secondary.

Decodes and judgements are cached in cache/ (committed), so re-running only pays for new
items, the same input always gets the same result, and anyone can reproduce the scores.
"""

import csv
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from emoji_eval.columns import model_columns
from emoji_eval.paths import CACHE_DIR, ENV_FILE
from emoji_eval.prompt import EXAMPLES  # the hand-made few-shot pairs, used by sanity_check

load_dotenv(ENV_FILE)

CACHE_FILE = CACHE_DIR / "decode_cache.csv"
JUDGE_CACHE_FILE = CACHE_DIR / "judge_cache.csv"

# Why these models (provisional choices, made 2026-11-21 - revisit before final evaluations):
# - Decoder = Haiku: not a teacher or student candidate (no family bias), cheap, good in
#   EN and FR. NOT compared against other decoders yet. The decoder should read like a
#   person, not as well as possible: a too-strong reader inflates every score. To compare,
#   run human_check.py with other decoders (e.g. Sonnet 5.5, a local Gemma) on the same
#   guesses and keep the one closest to humans (highest Spearman, smallest average gap).
#   Changing the decoder invalidates decode_cache.csv and all earlier scores.
# - Judge = Sonnet: a different model from the decoder, so it doesn't grade its own
#   writing, and strong enough to tell "right objects, wrong story" from "gist kept".
#   Validated so far only by the sanity check (matched 1.75 vs shuffled 0.00).
DECODER_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-sonnet-5-5"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANGUAGES = {"en": "English", "fr": "French", "es": "Spanish"}  # Spanish added in step 8 (v2)

# The decoder gets the rules a demo visitor would know, but no examples and no original.
DECODER_PROMPT = """You will see an "emoji sentence": a short sequence of emojis that summarises an everyday sentence. Guess the original sentence.

Conventions:
- Groups separated by spaces are one idea each, in story order: who → action → object → result → emotion.
- ➡️ means "then" or "so".
- ❌ negates the emoji right before it (🧳❌ = the suitcase is missing).
- ⏪ marks the past, ⏩ the future.
- 💬 shows speech or a message: the speaker comes before it, what was said comes after it (🫵 = you).

Write one natural, everyday sentence in the requested language: your best guess at the original, with no comments, notes or alternatives, even if you are unsure."""

DECODER_SCHEMA = {
    "type": "object",
    "properties": {"sentence": {"type": "string"}},
    "required": ["sentence"],
    "additionalProperties": False,
}

# The judge only sees the two sentences, never the emojis or which model made them.
JUDGE_PROMPT = """You compare an original sentence with a guess written by someone who only saw a short emoji summary of it. Decide whether the guess captures the gist of the original: the situation, what happened, and how it turned out.

Ignore: wording, tense, language register, I/he/she/we, small details, and emotions that are added or missing. Who did what only matters if it changes the story (for example, victim and culprit swapped).

Scores:
2 = gist kept: same situation and same main event or outcome.
1 = partly: the topic or situation is right, but the main event or outcome is wrong or missing.
0 = wrong: a different story, or only isolated objects in common.

Give a one-sentence reason first, then the score."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "score": {"type": "integer", "enum": [0, 1, 2]},
    },
    "required": ["reason", "score"],
    "additionalProperties": False,
}

# v2 judge (step 7): three criteria instead of one gist score, so a related guess with the
# wrong facts scores above an unrelated one. v1 above stays unchanged for steps 0 to 6.
JUDGE_V2_CACHE_FILE = CACHE_DIR / "judge_v2_cache.csv"
JUDGE_PROMPT_V2 = """You compare an original sentence with a guess written by someone who only saw a short emoji summary of it. Score how much of the original the guess recovers, on three criteria:

Topic: is the guess about the same situation or subject (the setting, the main people or things)?
Event: does the same main thing happen? Who did what counts here: swapped roles (victim and culprit) or an event that is negated (did not happen instead of happened) make the event wrong.
Outcome: does it end the same way (the result, the consequence, or the emotion the story is about)? Answer "n/a" only when the original has no outcome beyond its event (for example "Ziri was in his room.").

For each criterion: 2 = yes, 1 = partly, 0 = no.

Ignore: wording, tense, language register, I/he/she/we, small details, and emotions that are added or missing, unless the emotion is the outcome.

Give a one-sentence reason first, then the three scores."""

JUDGE_V2_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "topic": {"type": "string", "enum": ["0", "1", "2"]},
        "event": {"type": "string", "enum": ["0", "1", "2"]},
        "outcome": {"type": "string", "enum": ["0", "1", "2", "n/a"]},
    },
    "required": ["reason", "topic", "event", "outcome"],
    "additionalProperties": False,
}


# v2.1 (step 7, after the first human check): same criteria, two rules the author set because
# emojis cannot express them: mood (if / real / plan) and exactly who the people are.
# Reversed roles and negation still count. Own cache, so v2 results stay reproducible.
JUDGE_V21_CACHE_FILE = CACHE_DIR / "judge_v21_cache.csv"
JUDGE_PROMPT_V21 = """You compare an original sentence with a guess written by someone who only saw a short emoji summary of it. Score how much of the original the guess recovers, on three criteria:

Topic: is the guess about the same situation or subject (the setting, the main people or things)?
Event: does the same main thing happen? Two errors make the event wrong: reversed roles (who did what to whom is backwards, for example victim and culprit swapped) and negation (the guess says it did not happen when it did, or the reverse).
Outcome: does it end the same way (the result, the consequence, or the emotion the story is about)? Answer "n/a" only when the original has no outcome beyond its event (for example "Ziri was in his room.").

For each criterion: 2 = yes, 1 = partly, 0 = no.

Emojis cannot show these, so do not penalise them:
- Mood: a condition, wish, plan, request or question guessed as something that really happened (or the reverse) counts as the same event. "If my car stays broken, I'll miss the party" and "My car broke down, so I missed the party" have the same event and outcome.
- Who exactly the people are, beyond number, gender, rough age and a visible job: "my brother" and "a boy", "we" and "they", "my friend" and "a colleague" are the same people. A different person doing the action is not an error unless the roles are reversed.

Also ignore: wording, tense, language register, small details, and emotions that are added or missing, unless the emotion is the outcome.

Give a one-sentence reason first, then the three scores."""


def criteria_score(topic: str, event: str, outcome: str) -> float:
    """v2 score from 0 to 1: the share of the points available. An n/a outcome is left out."""
    points = [int(topic), int(event)] + ([] if outcome == "n/a" else [int(outcome)])
    return sum(points) / (2 * len(points))


# ---------- small CSV caches ----------

def read_cache(path: Path, key_fields: list[str]) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {tuple(r[k] for k in key_fields): r for r in csv.DictReader(f)}


def append_cache(path: Path, row: dict) -> None:
    is_new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def load_cache() -> dict:
    """(emojis, lang) -> decoded sentence."""
    return {k: r["decoded"] for k, r in read_cache(CACHE_FILE, ["emojis", "lang"]).items()}


def load_judge_cache() -> dict:
    """(original, guess) -> (score, reason)."""
    rows = read_cache(JUDGE_CACHE_FILE, ["original", "guess"])
    return {k: (int(r["score"]), r["reason"]) for k, r in rows.items()}


def load_judge_v2_cache(path: Path = JUDGE_V2_CACHE_FILE) -> dict:
    """(original, guess) -> {"topic", "event", "outcome", "reason"}. Pass JUDGE_V21_CACHE_FILE for v2.1."""
    rows = read_cache(path, ["original", "guess"])
    return {k: {f: r[f] for f in ("topic", "event", "outcome", "reason")} for k, r in rows.items()}


# ---------- model calls ----------

def decode(client: anthropic.Anthropic, cache: dict, emojis: str, lang: str) -> str:
    if (emojis, lang) in cache:
        return cache[(emojis, lang)]
    response = client.messages.create(
        model=DECODER_MODEL,
        max_tokens=200,
        # Thinking off: a one-sentence guess doesn't need it, and it keeps calls fast
        # and cheap. (Haiku 5.5 rejects a custom temperature; the cache keeps results stable.)
        thinking={"type": "disabled"},
        system=DECODER_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Language: {LANGUAGES[lang]}\nEmoji sentence: {emojis}",
        }],
        # Structured output: the answer must be {"sentence": "..."}, which leaves no room
        # for preambles ("I'll guess: ...") or notes after the sentence.
        output_config={"format": {"type": "json_schema", "schema": DECODER_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        decoded = ""
    else:
        text = next(b.text for b in response.content if b.type == "text")
        decoded = json.loads(text)["sentence"].strip()
    cache[(emojis, lang)] = decoded
    append_cache(CACHE_FILE, {"emojis": emojis, "lang": lang, "decoded": decoded})
    return decoded


def judge(client: anthropic.Anthropic, cache: dict, original: str, guess: str) -> tuple[int, str]:
    """Gist score of a guess against the original: 2 kept, 1 partly, 0 wrong."""
    if (original, guess) in cache:
        return cache[(original, guess)]
    if not guess.strip():
        return 0, "empty guess"
    response = client.beta.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": f"Original: {original}\nGuess: {guess}"}],
        # Low effort: a short comparison doesn't need deep thinking.
        output_config={"effort": "low",
                       "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        # If Sonnet declines a request, the API retries it on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        score, reason = 0, "judge refused"
    else:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        score, reason = data["score"], data["reason"]
    cache[(original, guess)] = (score, reason)
    append_cache(JUDGE_CACHE_FILE, {"original": original, "guess": guess,
                                    "score": score, "reason": reason})
    return score, reason


def judge_v2(client: anthropic.Anthropic, cache: dict, original: str, guess: str,
             version: str = "v2") -> dict:
    """v2 criteria for a guess: topic, event, outcome ("0"-"2", outcome may be "n/a"),
    reason, and score (0 to 1, see criteria_score). version "v2.1" uses the revised prompt and
    its own cache file (load the cache with load_judge_v2_cache(JUDGE_V21_CACHE_FILE))."""
    prompt, cache_file = {"v2": (JUDGE_PROMPT_V2, JUDGE_V2_CACHE_FILE),
                          "v2.1": (JUDGE_PROMPT_V21, JUDGE_V21_CACHE_FILE)}[version]
    if (original, guess) not in cache:
        if not guess.strip():
            result = {"topic": "0", "event": "0", "outcome": "0", "reason": "empty guess"}
        else:
            response = client.beta.messages.create(
                model=JUDGE_MODEL,
                max_tokens=2000,
                system=prompt,
                messages=[{"role": "user", "content": f"Original: {original}\nGuess: {guess}"}],
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": JUDGE_V2_SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if response.stop_reason == "refusal":
                result = {"topic": "0", "event": "0", "outcome": "0", "reason": "judge refused"}
            else:
                data = json.loads(next(b.text for b in response.content if b.type == "text"))
                result = {f: data[f] for f in ("topic", "event", "outcome", "reason")}
        cache[(original, guess)] = result
        append_cache(cache_file, {"original": original, "guess": guess, **result})
    result = cache[(original, guess)]
    return {**result, "score": criteria_score(result["topic"], result["event"], result["outcome"])}


def similarities(embedder: SentenceTransformer, originals: list[str], decoded: list[str]) -> list[float]:
    """Cosine similarity between each original and its decoded sentence."""
    # With normalised vectors, cosine similarity is just the dot product.
    a = embedder.encode(originals, normalize_embeddings=True)
    b = embedder.encode(decoded, normalize_embeddings=True)
    return [float(x @ y) for x, y in zip(a, b)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------- sanity check ----------

FRENCH_HINTS = {"je", "j'ai", "il", "elle", "on", "un", "une", "le", "la", "les", "mon", "ma", "mes",
                "de", "du", "et", "donc", "mais", "m'a", "à"}


def guess_lang(text: str) -> str:
    """Rough language guess for the few-shot examples, which have no language column."""
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    has_accent = any(c in text for c in "éèêàçùâîô")
    return "fr" if has_accent or len(words & FRENCH_HINTS) >= 2 else "en"


def sanity_check(client: anthropic.Anthropic, embedder: SentenceTransformer) -> None:
    """Hand-made examples should score high; the same decodes paired with the wrong
    originals should score much lower. If not, the scorer isn't measuring meaning."""
    decode_cache, judge_cache = load_cache(), load_judge_cache()
    langs = [guess_lang(text) for text, _ in EXAMPLES]
    originals = [text for text, _ in EXAMPLES]
    decoded = [decode(client, decode_cache, emojis, lang) for (_, emojis), lang in zip(EXAMPLES, langs)]
    shifted = decoded[1:] + decoded[:1]  # every decode paired with a different original

    matched_gist = [judge(client, judge_cache, o, d) for o, d in zip(originals, decoded)]
    shuffled_gist = [judge(client, judge_cache, o, d)[0] for o, d in zip(originals, shifted)]
    matched_sim = similarities(embedder, originals, decoded)
    shuffled_sim = similarities(embedder, originals, shifted)

    for (text, emojis), d, (g, reason), s in zip(EXAMPLES, decoded, matched_gist, matched_sim):
        print(f"gist {g}  sim {s:.3f}  {emojis}\n       original: {text}\n"
              f"       decoded:  {d}\n       judge:    {reason}")
    print(f"\nMatched pairs:  gist {mean([g for g, _ in matched_gist]):.2f}   sim {mean(matched_sim):.3f}")
    print(f"Shuffled pairs: gist {mean(shuffled_gist):.2f}   sim {mean(shuffled_sim):.3f}"
          "   (baseline for unrelated sentences)")


# ---------- scoring a bake-off ----------

def score_results(results_file: Path, scores_file: Path,
                  client: anthropic.Anthropic, embedder: SentenceTransformer) -> None:
    """Decode and judge every valid output in a bake-off results file."""
    with open(results_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = model_columns(rows[0])
    decode_cache, judge_cache = load_cache(), load_judge_cache()

    # Decode and judge every valid output. Invalid ones are skipped: they score 0.
    items = []  # (row, model, emojis, decoded, gist, reason)
    for model in models:
        print(f"Decoding and judging {model}...")
        for row in rows:
            if row[f"{model} check"] == "ok":
                emojis = row[model]
                decoded = decode(client, decode_cache, emojis, row["lang"])
                gist, reason = judge(client, judge_cache, row["sentence"], decoded)
                items.append((row, model, emojis, decoded, gist, reason))

    sims = similarities(embedder, [it[0]["sentence"] for it in items], [it[3] for it in items])

    gist_by, sim_by = {}, {}  # (model, lang) -> list, valid outputs only
    out_rows = []
    for (row, model, emojis, decoded, gist, reason), sim in zip(items, sims):
        key = (model, row["lang"])
        gist_by.setdefault(key, []).append(gist)
        sim_by.setdefault(key, []).append(sim)
        out_rows.append({"id": row["id"], "lang": row["lang"], "model": model,
                         "sentence": row["sentence"], "emojis": emojis, "decoded": decoded,
                         "gist": gist, "judge_reason": reason, "similarity": f"{sim:.3f}"})

    with open(scores_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    totals = {}
    for row in rows:
        totals[row["lang"]] = totals.get(row["lang"], 0) + 1
    print("\nGist score, 0-2 (valid only | invalid counted as 0) · % gist kept · similarity (reference)")
    for model in models:
        for lang in sorted(totals):
            g = gist_by.get((model, lang), [])
            kept = sum(x == 2 for x in g) / totals[lang]
            print(f"  {model:<16} [{lang}]  {mean(g):.2f} | {sum(g) / totals[lang]:.2f}"
                  f"   {kept:4.0%} kept   sim {mean(sim_by.get((model, lang), [])):.3f}"
                  f"   ({len(g)} valid)")
    print(f"\nPer-sentence details (with the judge's reasons) saved to {scores_file}")


def make_clients() -> tuple[anthropic.Anthropic, SentenceTransformer]:
    client = anthropic.Anthropic(max_retries=5)  # the SDK retries 429s and 5xx with backoff
    # Downloaded once (~500 MB), then cached. CPU on purpose: recent PyTorch builds crash
    # on the Apple GPU backend (MPS) with macOS 14, and this model is small enough for CPU.
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return client, embedder
