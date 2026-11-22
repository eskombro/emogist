"""Step 3c: compare labelling prompt versions with the teacher on the real dev set.

Claude only scores each version as a whole (Haiku decodes, Sonnet judges). It never picks
or rejects individual labels. Versions already in the results file are not run again.

    python 03_dataset/compare_prompts.py
    python 03_dataset/compare_prompts.py --versions v1-current v4-human-examples
"""

import argparse
import csv
import random
from pathlib import Path

import requests

from emoji_eval.roundtrip import decode, judge, load_cache, load_judge_cache
from emoji_eval.runner import TEMPERATURE
from emoji_eval.validate import KEYCAP, check_conventions, check_validity
import anthropic
from label_prompt import PROMPTS, messages_for

MODEL = "deepseek-v4.1-flash:cloud"
URL = "http://localhost:11434/api/chat"
HERE = Path(__file__).resolve().parent
DEV_FILE = HERE / "sentences" / "real_dev.csv"
RESULTS_FILE = HERE / "results" / "prompt_comparison.csv"


def label(version: str, sentence: str) -> str:
    response = requests.post(URL, timeout=120, json={
        "model": MODEL, "messages": messages_for(version, sentence), "stream": False, "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": 100}})
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def interval(differences: list[int]) -> str:
    rng = random.Random(42)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(5000))
    return f"{sum(differences) / len(differences):+.2f} [{means[125]:+.2f}, {means[4874]:+.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", choices=list(PROMPTS), default=list(PROMPTS),
                        help="the first one is the baseline")
    versions = parser.parse_args().versions
    source = RESULTS_FILE if RESULTS_FILE.exists() else DEV_FILE
    with open(source, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for version in versions:
        if version not in rows[0]:
            print(f"Labelling with {version}...")
            for row in rows:
                row[version] = label(version, row["sentence"])
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    client = anthropic.Anthropic(max_retries=5)
    decode_cache, judge_cache = load_cache(), load_judge_cache()
    scores = {}
    print(f"\n{'version':<16} {'valid':>5} {'clean':>5} {'keycaps':>7} {'FR':>5} {'EN':>5} {'all':>5}")
    for version in versions:
        scores[version] = []
        valid = clean = keycaps = 0
        for row in rows:
            output = row[version]
            keycaps += KEYCAP in output or "🔟" in output
            if check_validity(output) != "ok":
                scores[version].append(0)
                continue
            valid += 1
            clean += not check_conventions(output)
            guess = decode(client, decode_cache, output, row["lang"])
            scores[version].append(judge(client, judge_cache, row["sentence"], guess)[0])
        by_lang = {l: [s for s, r in zip(scores[version], rows) if r["lang"] == l] for l in ("fr", "en")}
        print(f"{version:<16} {valid:>5} {clean:>5} {keycaps:>7} {sum(by_lang['fr']) / len(by_lang['fr']):>5.2f} "
              f"{sum(by_lang['en']) / len(by_lang['en']):>5.2f} {sum(scores[version]) / len(rows):>5.2f}")
    baseline = versions[0]
    for version in versions[1:]:
        if version != baseline:
            print(f"{version} - {baseline}: " + interval([a - b for a, b in zip(scores[version], scores[baseline])]))


if __name__ == "__main__":
    main()
