"""Step 3d: label the pilot sentences with the teacher and build three small training sets.

    A = real only, B = AI-generated only, C = half and half
Each set has 200 sentences per language, so the pilot compares the source of the data and
nothing else. The filters here are plain code: Claude never picks or rejects a label.

Resumable: sentences already in labels/pilot_labels.csv are not sent again.
"""

import csv
import hashlib
import json
from pathlib import Path

import requests

from emoji_eval.runner import TEMPERATURE
from emoji_eval.validate import check_conventions, check_validity
from label_prompt import WORKSHEET, messages_for

HERE = Path(__file__).resolve().parent
SENTENCES = HERE / "sentences"
LABELS_FILE = HERE / "labels" / "pilot_labels.csv"
PILOT_DIR = HERE / "pilot"
TEACHER = "deepseek-v4.1-flash:cloud"
URL = "http://localhost:11434/api/chat"
VERSION = "v4-human-examples"
SET_SIZE = 200  # per language
TO_LABEL = 220  # per language and source: a margin for labels that get dropped
MAX_ATTEMPTS = 2
FIELDS = ["id", "lang", "source", "sentence", "emojis", "status", "attempts", "teacher", "prompt"]


def load(name: str, source: str) -> list[dict]:
    with open(SENTENCES / name, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    picked = []
    for lang in ("en", "fr"):
        picked += [{"id": r["id"], "lang": lang, "source": source, "sentence": r["sentence"]}
                   for r in rows if r["lang"] == lang][:TO_LABEL]
    return picked


def label(sentence: str, attempt: int) -> str:
    response = requests.post(URL, timeout=120, json={
        "model": TEACHER, "messages": messages_for(VERSION, sentence), "stream": False, "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": 100, "seed": attempt}})
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def problem(output: str) -> str:
    validity = check_validity(output)
    return validity if validity != "ok" else "; ".join(check_conventions(output))


def run_labelling(sentences: list[dict]) -> list[dict]:
    prompt_id = f"{VERSION}@{hashlib.sha256(WORKSHEET.read_bytes()).hexdigest()[:8]}"
    LABELS_FILE.parent.mkdir(exist_ok=True)
    done = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE, newline="", encoding="utf-8") as f:
            done = {r["id"]: r for r in csv.DictReader(f)}
    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for n, row in enumerate(sentences, 1):
            if row["id"] in done:
                continue
            output, reason, attempt = "", "not labelled", 0
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    output = label(row["sentence"], attempt)
                    reason = problem(output)
                except requests.RequestException as error:
                    output, reason = "", f"request error: {error}"
                if not reason:
                    break
            result = {**row, "emojis": output, "status": reason or "ok", "attempts": attempt,
                      "teacher": TEACHER, "prompt": prompt_id}
            writer.writerow(result)
            f.flush()
            done[row["id"]] = result
            if n % 50 == 0:
                print(f"  {n}/{len(sentences)}", flush=True)
    return [done[r["id"]] for r in sentences]


def build_sets(labelled: list[dict]) -> None:
    good = {(s, l): [r for r in labelled if r["source"] == s and r["lang"] == l and r["status"] == "ok"]
            for s in ("real", "ai") for l in ("en", "fr")}
    for key, rows in good.items():
        if len(rows) < SET_SIZE:
            raise SystemExit(f"Only {len(rows)} usable labels for {key}: need {SET_SIZE}")
    sets = {"A_real": {}, "B_ai": {}, "C_mixed": {}}
    for lang in ("en", "fr"):
        sets["A_real"][lang] = good[("real", lang)][:SET_SIZE]
        sets["B_ai"][lang] = good[("ai", lang)][:SET_SIZE]
        half = SET_SIZE // 2
        sets["C_mixed"][lang] = good[("real", lang)][:half] + good[("ai", lang)][:half]
    PILOT_DIR.mkdir(exist_ok=True)
    for name, by_lang in sets.items():
        rows = by_lang["en"] + by_lang["fr"]
        with open(PILOT_DIR / f"train_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "lang", "source", "sentence", "emojis")},
                                   ensure_ascii=False) + "\n")
        print(f"train_{name}.jsonl: {len(rows)} examples, ➡️ in {sum('➡' in r['emojis'] for r in rows) / len(rows):.0%}")


def main() -> None:
    sentences = load("real_pool.csv", "real") + load("ai_pool.csv", "ai")
    labelled = run_labelling(sentences)
    for source in ("real", "ai"):
        rows = [r for r in labelled if r["source"] == source]
        ok = sum(r["status"] == "ok" for r in rows)
        retried = sum(r["status"] == "ok" and int(r["attempts"]) > 1 for r in rows)
        print(f"{source}: {ok}/{len(rows)} usable ({retried} needed the retry)")
    build_sets(labelled)


if __name__ == "__main__":
    main()
