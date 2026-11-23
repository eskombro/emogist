"""Where does fine-tuned Qwen lose the gist? Aggregate breakdown on the 100 real dev sentences.

    python 05_evaluation/error_breakdown.py

Reads the saved scores, so it makes no model calls. Groups come from plain-code rules, overlap,
and are approximate. It reports totals only and selects no sentence for training.
"""

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "04_fine_tuning"))

from score_pilot import interval  # noqa: E402
from emoji_eval.validate import check_conventions  # noqa: E402

SCORES_FILE = ROOT / "04_fine_tuning" / "results" / "full_scores.csv"
OUT_FILE = HERE / "results" / "error_breakdown.csv"
STUDENT, TEACHER = "qwen2b_full", "teacher (Flash)"
KEYCAP = "⃣"
MIN_GROUP = 10

# "one", "un", "une" and "neuf" are left out because they usually aren't numbers.
NUMBER = re.compile(r"\d|\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty|thirty|forty|"
                    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|deux|trois|quatre|cinq|sept|huit|"
                    r"dix|onze|douze|treize|quatorze|quinze|seize|vingt|trente|quarante|cinquante|soixante|"
                    r"cent|mille)\b", re.I)
NEGATION = re.compile(r"n't\b|\b(not|never|no|nobody|nothing|cannot|ne|pas|jamais|rien|personne|aucun|aucune)\b|\bn'", re.I)
SPEECH = re.compile(r"\b(said|say|says|told|tell|tells|asked|ask|dit|dis|disent|demandé|demande|raconté)\b", re.I)
CAUSE = re.compile(r"\b(because|since|so|parce|car|donc|puisque|alors)\b", re.I)
PAST = re.compile(r"\b(was|were|had|did|went|yesterday|ago|hier|était|étais|étaient|avait|avais|ai|as|a|ont|avons)\b", re.I)
FUTURE = re.compile(r"\b(will|tomorrow|going to|demain|va|vais|vont|allons)\b|'ll\b", re.I)


def words(sentence: str) -> int:
    return len(re.findall(r"[\w'’-]+", sentence))


GROUPS = {
    "all": lambda s: True,
    "lang: French": lambda s: s["lang"] == "fr",
    "lang: English": lambda s: s["lang"] == "en",
    "length: up to 5 words": lambda s: words(s["sentence"]) <= 5,
    "length: 6-8 words": lambda s: 6 <= words(s["sentence"]) <= 8,
    "length: 9+ words": lambda s: words(s["sentence"]) >= 9,
    "has a number": lambda s: bool(NUMBER.search(s["sentence"])),
    "no number": lambda s: not NUMBER.search(s["sentence"]),
    "negation": lambda s: bool(NEGATION.search(s["sentence"])),
    "question": lambda s: "?" in s["sentence"],
    "reported speech": lambda s: bool(SPEECH.search(s["sentence"])),
    "cause or result": lambda s: bool(CAUSE.search(s["sentence"])),
    "past marker": lambda s: bool(PAST.search(s["sentence"])),
    "future marker": lambda s: bool(FUTURE.search(s["sentence"])),
}


def mean(values: list[int]) -> float:
    return sum(values) / len(values)


def main() -> None:
    with open(SCORES_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    student = {r["id"]: r for r in rows if r["model"] == STUDENT}
    teacher = {r["id"]: r for r in rows if r["model"] == TEACHER}
    if len(student) != 100 or set(student) != set(teacher):
        raise SystemExit("Expected the same 100 dev sentences for both models")

    out = []
    print(f"{'group':<24}{'n':>4}{'Qwen':>7}{'lost':>6}{'teacher':>9}  teacher - Qwen")
    for name, member in GROUPS.items():
        ids = [i for i, r in student.items() if member(r)]
        if len(ids) < MIN_GROUP:
            print(f"{name:<24}{len(ids):>4}  too few sentences to measure")
            out.append({"group": name, "n": len(ids), "qwen_gist": "", "qwen_lost": "",
                        "teacher_gist": "", "teacher_minus_qwen": ""})
            continue
        qwen =[int(student[i]["gist"]) for i in ids]
        best = [int(teacher[i]["gist"]) for i in ids]
        gap = interval([b - q for b, q in zip(best, qwen)])
        lost = qwen.count(0) / len(qwen)
        print(f"{name:<24}{len(ids):>4}{mean(qwen):>7.2f}{lost:>6.0%}{mean(best):>9.2f}  {gap}")
        out.append({"group": name, "n": len(ids), "qwen_gist": round(mean(qwen), 3), "qwen_lost": round(lost, 3),
                    "teacher_gist": round(mean(best), 3), "teacher_minus_qwen": gap})
    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    print("\nOutput form on sentences with a number:")
    ids = [i for i, r in student.items() if NUMBER.search(r["sentence"])]
    for label, source in (("Qwen", student), ("Teacher", teacher)):
        texts = [source[i]["output"] for i in ids]
        print(f"  {label}: {sum(KEYCAP in t for t in texts)}/{len(ids)} keycaps, "
              f"{sum(bool(re.search(r'[0-9](?!️?' + KEYCAP + ')', t)) for t in texts)}/{len(ids)} plain digits, "
              f"{sum(source[i]['status'] != 'ok' for i in ids)}/{len(ids)} invalid")

    print("\nValidity and conventions on all 100:")
    for label, source in (("Qwen", student), ("Teacher", teacher)):
        valid = [r for r in source.values() if r["status"] == "ok"]
        print(f"  {label}: {len(valid)} valid, {sum(not check_conventions(r['output']) for r in valid)} with no warning")


if __name__ == "__main__":
    main()
