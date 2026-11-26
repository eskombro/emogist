"""Step 9a: AI-generated training sentences, EN, FR and ES, with DeepSeek V4.1 Flash only.

    python 09_v2_dataset_and_labels/generate_ai_train.py                     # full run
    python 09_v2_dataset_and_labels/generate_ai_train.py --scale 0.03        # small test batch

Reuses step 8's generator (08_v2_spanish_and_eval_sets/generate_ai_v2.py): same prompts, topics,
tones, lengths, speakers, real example sentences and rejection rules. Differences:
  - Flash only (gpt-oss was dropped for training data in step 8);
  - quotas per language: 140 questions, 140 conditionals and 120 for each other pattern, plus a
    3% margin for sentences that later get no valid label;
  - plan cells already used by step 8 (dev, test and their spares) are skipped, and every step 8
    sentence (AI and real, all sets), the step 9 worksheet and dev/test count as known sentences,
    so near-duplicates are rejected;
  - requests run in parallel, but sentences are accepted in plan order, so the result doesn't
    depend on timing.
Claude never writes these sentences. Resumable: ids already in the output are skipped.
Output: sentences/ai_train.csv.
"""

import argparse
import csv
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STEP8 = REPO / "08_v2_spanish_and_eval_sets"
sys.path.insert(0, str(STEP8))
import generate_ai_v2 as g2  # noqa: E402  (step 8 generator)
from rules_v2 import v1  # noqa: E402

OUT = HERE / "sentences" / "ai_train.csv"
WORKSHEET = HERE / "label_examples_v2.csv"
QUOTAS = {"question": 140, "conditional": 140, "speech": 120, "past": 120, "future": 120, "mishap": 120,
          "moment": 120, "numbers": 120}
MARGIN = 1.03
FLASH = g2.GENERATORS[0]
assert FLASH[0] == "deepseek-v4.1-flash:cloud"
THREADS = 6
BATCH = 12  # plan cells generated at once; accepted in plan order
FIELDS = g2.FIELDS + ["prompt_tokens", "output_tokens"]


def generate(content: str) -> tuple[str, int, int]:
    model, think, max_tokens = FLASH
    response = requests.post(g2.g1.URL, timeout=180, json={
        "model": model, "messages": [{"role": "user", "content": content}], "stream": False, "think": think,
        "options": {"temperature": g2.g1.TEMPERATURE, "num_predict": max_tokens}})
    response.raise_for_status()
    body = response.json()
    return (body["message"]["content"].strip().strip('"«» '), body.get("prompt_eval_count", 0),
            body.get("eval_count", 0))


def step8_used(lang: str) -> set[int]:
    """Plan indices step 8 already generated (kept or not, its ids are in ai_eval.csv)."""
    with open(STEP8 / "sets" / "ai_eval.csv", newline="", encoding="utf-8") as f:
        return {(r["pattern"], int(r["id"].rsplit("-", 1)[1])) for r in csv.DictReader(f) if r["lang"] == lang}


def known(lang: str) -> v1.NearDuplicates:
    seen = g2.known_sentences(lang, OUT)  # dev/test (v1 and v2 real), v1 pools, v2 real pool, this output
    sources = [STEP8 / "sets" / "ai_eval.csv", STEP8 / "sets" / "dev.csv", STEP8 / "sets" / "test.csv", WORKSHEET]
    for source in sources:
        with open(source, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["lang"] == lang:
                    seen.add(set(v1.words(row["sentence"])))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0, help="fraction of the quotas (e.g. 0.03 for a test)")
    parser.add_argument("--langs", nargs="+", default=list(g2.LANGS))
    args = parser.parse_args()

    OUT.parent.mkdir(exist_ok=True)
    existing = []
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    done = {r["id"] for r in existing}
    tokens = Counter()
    with open(OUT, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(THREADS) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not existing:
            writer.writeheader()
        for lang in args.langs:
            seen, used = known(lang), step8_used(lang)
            pools = g2.example_pools(lang)
            for examples in pools.values():
                for example in examples:
                    seen.add(set(v1.words(example)))
            for pattern, quota in QUOTAS.items():
                target = math.ceil(quota * MARGIN * args.scale)
                kept = sum(r["lang"] == lang and r["pattern"] == pattern for r in existing)
                cells = [(i, c) for i, c in enumerate(g2.plan(lang, pattern))
                         if (pattern, i) not in used and f"ai2t-{lang}-{pattern}-{i:05d}" not in done]
                tried, rejected = 0, Counter()
                while kept < target and cells and tried < 3 * target + 10:
                    batch, cells = cells[:BATCH], cells[BATCH:]
                    prompts = [g2.prompt_for(lang, cell, i, pools) for i, cell in batch]
                    results = list(pool.map(lambda p: _safe(p), prompts))
                    for (i, cell), result in zip(batch, results):
                        if kept >= target:
                            break
                        tried += 1
                        if result is None:
                            rejected["request error"] += 1
                            continue
                        sentence, prompt_tokens, output_tokens = result
                        tokens["prompt"] += prompt_tokens
                        tokens["output"] += output_tokens
                        tokens["requests"] += 1
                        reason = g2.rejection(sentence, lang, pattern, seen)
                        if reason:
                            rejected[reason] += 1
                            continue
                        seen.add(set(v1.words(sentence)))
                        topic, _, tone, length, speaker = cell
                        writer.writerow({"id": f"ai2t-{lang}-{pattern}-{i:05d}", "lang": lang, "sentence": sentence,
                                         "topic": g2.TOPICS[lang][topic], "pattern": pattern,
                                         "tone": g2.TONES[lang][tone], "length": length,
                                         "speaker": g2.SPEAKERS[lang][speaker], "generator": FLASH[0],
                                         "prompt_tokens": prompt_tokens, "output_tokens": output_tokens})
                        f.flush()
                        kept += 1
                print(f"[{lang}/{pattern}] {kept}/{target}, {tried} requests this run, rejected: {dict(rejected)}",
                      flush=True)
    print(f"\nFlash requests this run: {tokens['requests']}, tokens in {tokens['prompt']}, out {tokens['output']}")


def _safe(prompt: str):
    try:
        return generate(prompt)
    except requests.RequestException as error:
        print(f"  request failed ({type(error).__name__})", flush=True)
        return None


if __name__ == "__main__":
    main()
