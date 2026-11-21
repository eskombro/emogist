"""Check the automatic decoder against a human: does Haiku read emoji sentences like people do?

You guess the original sentence for ~32 emoji sentences taken from roundtrip_scores.csv,
without seeing the originals. Your guesses are scored exactly like Haiku's decodes (gist
judge, plus embedding similarity for reference), then compared:
  - rank correlation (Spearman) between your scores and Haiku's, sentence by sentence;
  - average score per model, human vs Haiku: same ranking? is Haiku more generous?

Usage:
    python 01_student_selection/human_check.py            # guess (resumes where you stopped)
    python 01_student_selection/human_check.py --report   # only print the comparison

Guessing makes no API calls; the report sends your guesses to the judge (a few cents, cached).
Ideally someone who has NOT read test_sentences.csv does the guessing; if you have read it,
your memory of the sentences will make your scores look better than a visitor's would.
"""

import argparse
import csv
import random
from pathlib import Path

import anthropic
from sentence_transformers import SentenceTransformer

from emoji_eval.roundtrip import (DECODER_PROMPT, EMBEDDING_MODEL, judge, load_judge_cache,
                                  similarities)

HERE = Path(__file__).resolve().parent
SCORES_FILE = HERE / "results" / "roundtrip_scores.csv"
GUESSES_FILE = HERE / "human_guesses.csv"  # personal answers: not committed (.gitignore)
PER_CELL = 4  # items per (model, language) -> 4 models x 2 languages x 4 = 32
SEED = 42  # fixed, so the same items are picked every time (and resuming works)


def pick_items() -> list[dict]:
    """Stratified sample: PER_CELL items per (model, lang), each sentence used at most once,
    so a guess on one model's version can't help with another model's version."""
    with open(SCORES_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    cells = sorted({(r["model"], r["lang"]) for r in rows})
    used_ids, picked = set(), []
    for _ in range(PER_CELL):
        for cell in cells:
            for r in rows:
                if (r["model"], r["lang"]) == cell and r["id"] not in used_ids:
                    used_ids.add(r["id"])
                    picked.append(r)
                    break
    rng.shuffle(picked)  # mix models and languages
    return picked


def load_guesses() -> dict:
    """(model, id) -> guess."""
    if not GUESSES_FILE.exists():
        return {}
    with open(GUESSES_FILE, newline="", encoding="utf-8") as f:
        return {(r["model"], r["id"]): r["guess"] for r in csv.DictReader(f)}


def save_guess(item: dict, guess: str) -> None:
    is_new = not GUESSES_FILE.exists()
    with open(GUESSES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "lang", "model", "emojis", "guess"])
        if is_new:
            writer.writeheader()
        writer.writerow({"id": item["id"], "lang": item["lang"], "model": item["model"],
                         "emojis": item["emojis"], "guess": guess})


def ask(items: list[dict], guesses: dict) -> None:
    todo = [it for it in items if (it["model"], it["id"]) not in guesses]
    if not todo:
        return
    legend = DECODER_PROMPT.split("Conventions:")[1].split("\n\n")[0]
    print("Guess the original sentence for each emoji sentence (Haiku gets the same rules):")
    print(legend)
    print("\nWrite one everyday sentence in the language shown. Empty line = stop and resume later.\n")
    done = len(items) - len(todo)
    for n, item in enumerate(todo, start=done + 1):
        language = "French" if item["lang"] == "fr" else "English"
        guess = input(f"[{n}/{len(items)}] ({language})  {item['emojis']}\n> ").strip()
        if not guess:
            print("Stopped. Run again to continue.")
            return
        save_guess(item, guess)
        guesses[(item["model"], item["id"])] = guess


def ranks(values: list[float]) -> list[float]:
    """Rank positions (1 = smallest), ties get the average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation: 1 = same order, 0 = unrelated, -1 = opposite order."""
    ra, rb = ranks(a), ranks(b)
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    return cov / (var_a * var_b) ** 0.5 if var_a and var_b else 0.0


def report(items: list[dict], guesses: dict) -> None:
    answered = [it for it in items if (it["model"], it["id"]) in guesses]
    if len(answered) < 8:
        print(f"Only {len(answered)} guesses so far: answer at least 8 for a report.")
        return
    client = anthropic.Anthropic(max_retries=5)
    judge_cache = load_judge_cache()
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    human_guesses = [guesses[(it["model"], it["id"])] for it in answered]

    human_gist = [judge(client, judge_cache, it["sentence"], g)[0] for it, g in zip(answered, human_guesses)]
    haiku_gist = [int(it["gist"]) for it in answered]
    human_sim = similarities(embedder, [it["sentence"] for it in answered], human_guesses)
    haiku_sim = [float(it["similarity"]) for it in answered]

    n = len(answered)
    print(f"\n{n} sentences compared            gist judge (main) | similarity (reference)")
    print(f"Spearman, human vs Haiku:          {spearman(human_gist, haiku_gist):+.2f}      | {spearman(human_sim, haiku_sim):+.2f}")
    print(f"Average, human:                    {sum(human_gist) / n:.2f}       | {sum(human_sim) / n:.3f}")
    print(f"Average, Haiku:                    {sum(haiku_gist) / n:.2f}       | {sum(haiku_sim) / n:.3f}")
    print("(Haiku higher = more generous reader than you)")

    print("\nPer model, gist (human | Haiku, same sentences):")
    for model in sorted({it["model"] for it in answered}):
        idx = [i for i, it in enumerate(answered) if it["model"] == model]
        h = sum(human_gist[i] for i in idx) / len(idx)
        k = sum(haiku_gist[i] for i in idx) / len(idx)
        print(f"  {model:<16} {h:.2f} | {k:.2f}   ({len(idx)} sentences)")

    print("\nDisagreements on the gist (you and Haiku got different scores):")
    for i in sorted(range(n), key=lambda i: -abs(human_gist[i] - haiku_gist[i])):
        if human_gist[i] == haiku_gist[i]:
            break
        it = answered[i]
        print(f"  {it['emojis']}   human {human_gist[i]} vs Haiku {haiku_gist[i]}")
        print(f"     original: {it['sentence']}")
        print(f"     you:      {human_guesses[i]}")
        print(f"     Haiku:    {it['decoded']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="only print the comparison")
    args = parser.parse_args()

    items = pick_items()
    guesses = load_guesses()
    if not args.report:
        ask(items, guesses)
    report(items, guesses)


if __name__ == "__main__":
    main()
