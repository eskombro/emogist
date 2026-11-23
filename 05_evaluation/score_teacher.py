"""Measure the teacher on the test set, as a ceiling for the fine-tuned student.

    python 05_evaluation/score_teacher.py

The teacher gets the final labelling prompt and a single attempt, like the students: the retry
and filters used when building the dataset are not applied here. Its outputs are evaluation
material only and never go into training. Outputs are saved and reused.
"""

import csv
import hashlib
import sys
from pathlib import Path

import anthropic
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "04_fine_tuning"))

from score_pilot import BASELINE, interval  # noqa: E402
from emoji_eval.roundtrip import decode, judge, load_cache, load_judge_cache  # noqa: E402
from emoji_eval.runner import TEMPERATURE  # noqa: E402
from emoji_eval.validate import check_validity  # noqa: E402
from label_prompt import WORKSHEET, messages_for  # noqa: E402

TEACHER = "deepseek-v4.1-flash:cloud"
VERSION = "v4-human-examples"
DATASET_PROMPT_HASH = "4addf4cb"  # the worksheet used for every label in the full dataset
URL = "http://localhost:11434/api/chat"
SENTENCES_FILE = ROOT / "data" / "test_sentences.csv"
STUDENT_SCORES = ROOT / "04_fine_tuning" / "results" / "test_scores.csv"
OUTPUTS_FILE = HERE / "results" / "teacher_test_outputs.csv"
SCORES_FILE = HERE / "results" / "teacher_test_scores.csv"
STUDENT = "qwen2b_full"


def teacher_outputs(sentences: list[dict]) -> dict[str, str]:
    done = {}
    if OUTPUTS_FILE.exists():
        with open(OUTPUTS_FILE, newline="", encoding="utf-8") as f:
            done = {r["id"]: r["output"] for r in csv.DictReader(f)}
    OUTPUTS_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "output"])
        if f.tell() == 0:
            writer.writeheader()
        for row in sentences:
            if row["id"] in done:
                continue
            response = requests.post(URL, timeout=120, json={
                "model": TEACHER, "messages": messages_for(VERSION, row["sentence"]), "stream": False,
                "think": False, "options": {"temperature": TEMPERATURE, "num_predict": 100, "seed": 1}})
            response.raise_for_status()
            done[row["id"]] = response.json()["message"]["content"].strip()
            writer.writerow({"id": row["id"], "output": done[row["id"]]})
            f.flush()
    return done


def main() -> None:
    worksheet_hash = hashlib.sha256(WORKSHEET.read_bytes()).hexdigest()[:8]
    if worksheet_hash != DATASET_PROMPT_HASH:
        raise SystemExit(f"Labelling examples changed ({worksheet_hash}, dataset used {DATASET_PROMPT_HASH})")
    with open(SENTENCES_FILE, newline="", encoding="utf-8") as f:
        sentences = list(csv.DictReader(f))
    outputs = teacher_outputs(sentences)

    client = anthropic.Anthropic(max_retries=5)
    decode_cache, judge_cache = load_cache(), load_judge_cache()
    out = []
    for row in sentences:
        output = outputs[row["id"]]
        status = check_validity(output)
        guess, gist = "", 0
        if status == "ok":
            guess = decode(client, decode_cache, output, row["lang"])
            gist = judge(client, judge_cache, row["sentence"], guess)[0]
        out.append({"model": "teacher (Flash)", "id": row["id"], "lang": row["lang"], "sentence": row["sentence"],
                    "output": output, "status": status, "decoded": guess, "gist": gist})
    with open(SCORES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    with open(STUDENT_SCORES, newline="", encoding="utf-8") as f:
        students = list(csv.DictReader(f))
    gist = {name: {r["id"]: int(r["gist"]) for r in students if r["model"] == name} for name in (BASELINE, STUDENT)}
    gist["teacher"] = {r["id"]: r["gist"] for r in out}
    valid = sum(r["status"] == "ok" for r in out)
    print(f"\nTeacher valid: {valid}/{len(out)}")
    print(f"{'model':<20}{'FR':>6}{'EN':>6}{'all':>6}{'  0/1/2':>10}")
    for name, scores in gist.items():
        by = lambda lang: [scores[r["id"]] for r in sentences if r["lang"] == lang]
        values = list(scores.values())
        print(f"{name:<20}{sum(by('fr')) / 50:>6.2f}{sum(by('en')) / 50:>6.2f}{sum(values) / len(values):>6.2f}"
              f"{f'{values.count(0)}/{values.count(1)}/{values.count(2)}':>10}")
    for label, subset in (("all", sentences), ("FR", [r for r in sentences if r["lang"] == "fr"]),
                          ("EN", [r for r in sentences if r["lang"] == "en"])):
        print(f"teacher - fine-tuned ({label}): "
              + interval([gist["teacher"][r["id"]] - gist[STUDENT][r["id"]] for r in subset]))
    print("teacher - untrained (all): "
          + interval([gist["teacher"][r["id"]] - gist[BASELINE][r["id"]] for r in sentences]))


if __name__ == "__main__":
    main()
