"""Style check of the v2 labelling prompt against the v1 prompt, on the 300 v2 dev sentences.

    python 09_v2_dataset_and_labels/style_check_v2.py

Flash labels every dev sentence with both prompts at temperature 0.3, with a second seed if the
format is wrong (as in step 3). v2 then gets up to 2 repair requests: Flash sees its broken label and
the rule it breaks (labels.py). Plain counts only, no scoring: length, ➡️ share,
use of the v2 signs, and whether question sentences (dev tags) got ❓ and condition sentences a ➡️.
Ollama Cloud only, no Claude cost. Dev sentences are never training data.
Output: results/style_check_v2.csv.
"""

import csv
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import emoji

from emoji_eval.validate import check_conventions_v2, content_count
from label_prompt_v2 import worksheet_examples
from labels import Labeller

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "08_v2_spanish_and_eval_sets" / "sets" / "dev.csv"
OUT = HERE / "results" / "style_check_v2.csv"
SETUP = {"v1": {"tries": 2}, "v2": {"tries": 2, "repairs": 2}}
THREADS = 6
SIGNS = ["❓", "💭", "🤔", "❔", "📍", "🔁", "🔀", "🔄"]  # 🔀 and 🔄 dropped: counted to see them go


def share(rows: list[dict], test) -> str:
    hits = sum(bool(test(r)) for r in rows)
    return f"{hits}/{len(rows)} ({hits / len(rows):.0%})" if rows else "-"


def main() -> None:
    with open(DEV, newline="", encoding="utf-8") as f:
        dev = list(csv.DictReader(f))
    for prompt, setup in SETUP.items():
        labeller = Labeller(prompt, **setup)
        first_ok = {}

        def label(r: dict) -> str:
            first_ok[r["id"]] = labeller.first(r["sentence"]) == labeller.candidate(r["sentence"], 0.3, 1)
            return labeller.first(r["sentence"])

        with ThreadPoolExecutor(THREADS) as pool:
            for r, output in zip(dev, pool.map(label, dev)):
                r[prompt] = output
                r[f"{prompt}_retried"] = str(not first_ok[r["id"]] and bool(output))
        print(f"{prompt}: {labeller.requests} new Flash requests")
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dev[0]))
        writer.writeheader()
        writer.writerows(dev)

    examples = {e for _, e in worksheet_examples()}
    questions = [r for r in dev if "question" in r["tags"].split()]
    conditions = [r for r in dev if "conditional" in r["tags"].split()]
    others_q = [r for r in dev if r not in questions]
    print(f"\n{len(dev)} dev sentences: {len(questions)} questions, {len(conditions)} conditions\n")
    lines = [
        ("no valid label", lambda p: share(dev, lambda r: not r[p])),
        ("needed a retry", lambda p: share(dev, lambda r: r[f"{p}_retried"] == "True")),
        ("mean emojis", lambda p: f"{sum(content_count(r[p]) for r in dev if r[p]) / sum(bool(r[p]) for r in dev):.1f}"),
        ("1-2 emojis", lambda p: share(dev, lambda r: r[p] and content_count(r[p]) < 3)),
        ("9-10 emojis", lambda p: share(dev, lambda r: content_count(r[p]) > 8)),
        ("uses ➡️", lambda p: share(dev, lambda r: "➡" in r[p])),
        ("question → ❓ at end", lambda p: share(questions, lambda r: r[p].endswith(" ❓"))),
        ("not question → ❓", lambda p: share(others_q, lambda r: "❓" in r[p])),
        ("condition → uses ➡️", lambda p: share(conditions, lambda r: "➡" in r[p])),
        *[(f"uses {s}", (lambda s: lambda p: share(dev, lambda r: s in r[p]))(s)) for s in SIGNS[2:]],
        ("copy of an example", lambda p: share(dev, lambda r: r[p] in examples)),
        ("style warning left", lambda p: share(dev, lambda r: r[p] and [w for w in check_conventions_v2(r[p])
                                                                     if w != "fewer than 3 emojis"])),
    ]
    print(f"{'':<24}{'v1 prompt':>16}{'v2 prompt':>16}")
    for name, value in lines:
        print(f"{name:<24}{value('v1'):>16}{value('v2'):>16}")
    for lang in ("en", "fr", "es"):
        rows = [r for r in dev if r["lang"] == lang]
        print(f"{lang}: mean emojis v1 {sum(content_count(r['v1']) for r in rows) / len(rows):.1f}, "
              f"v2 {sum(content_count(r['v2']) for r in rows) / len(rows):.1f}")
    for prompt in SETUP:
        counts = Counter(e["emoji"] for r in dev for e in emoji.emoji_list(r[prompt]))
        print(f"top emojis {prompt}: {' '.join(f'{e}{n}' for e, n in counts.most_common(15))}")

    print("\nRandom sample (seed 3):")
    for r in random.Random(3).sample(dev, 15):
        print(f"[{r['lang']} {r['type']}] {r['sentence']}\n    v1 {r['v1']}\n    v2 {r['v2']}")


if __name__ == "__main__":
    main()
