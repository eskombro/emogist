"""Score pilot runs on the real dev set, against the same student before fine-tuning.

    python 04_fine_tuning/score_pilot.py

Reads every runs/run_*/pilot_outputs_*.csv. The baseline is plain Qwen3.5 2B through Ollama
with the full labelling prompt (rules + 20 examples); its outputs are saved and reused.
The teacher row is DeepSeek V4.1 Flash, from the prompt comparison on the same sentences.
"""

import csv
import random
import sys
from pathlib import Path

import anthropic
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "03_dataset"))

from emoji_eval.roundtrip import decode, judge, load_cache, load_judge_cache  # noqa: E402
from emoji_eval.runner import TEMPERATURE  # noqa: E402
from emoji_eval.validate import check_conventions, check_validity  # noqa: E402
from label_prompt import messages_for  # noqa: E402

DEV_FILE = ROOT / "03_dataset" / "sentences" / "real_dev.csv"
TEACHER_FILE = ROOT / "03_dataset" / "results" / "prompt_comparison.csv"
RUNS = HERE / "runs"
BASELINE_FILE = RUNS / "baseline_qwen2b_prompted.csv"
SCORES_FILE = HERE / "results" / "pilot_scores.csv"
BASELINE = "qwen2b prompted"


def baseline_outputs(dev: list[dict]) -> dict[str, str]:
    if not BASELINE_FILE.exists():
        print("Running the baseline on the dev set...", flush=True)
        rows = []
        for row in dev:
            response = requests.post("http://localhost:11434/api/chat", timeout=180, json={
                "model": "qwen3.5:2b", "messages": messages_for("v4-human-examples", row["sentence"]),
                "stream": False, "think": False, "options": {"temperature": TEMPERATURE, "num_predict": 100}})
            response.raise_for_status()
            rows.append({"id": row["id"], "output": response.json()["message"]["content"].strip()})
        with open(BASELINE_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "output"])
            writer.writeheader()
            writer.writerows(rows)
    with open(BASELINE_FILE, newline="", encoding="utf-8") as f:
        return {r["id"]: r["output"] for r in csv.DictReader(f)}


def read_outputs(path: Path, column: str = "output") -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r[column] for r in csv.DictReader(f)}


def interval(differences: list[int]) -> str:
    rng = random.Random(42)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(5000))
    return f"{sum(differences) / len(differences):+.2f} [{means[125]:+.2f}, {means[4874]:+.2f}]"


def main() -> None:
    with open(DEV_FILE, newline="", encoding="utf-8") as f:
        dev = list(csv.DictReader(f))
    models = {BASELINE: baseline_outputs(dev)}
    for path in sorted(RUNS.glob("run_*/pilot_outputs_*.csv")):
        models[path.stem.replace("pilot_outputs_", "")] = read_outputs(path)
    models["teacher (Flash)"] = read_outputs(TEACHER_FILE, "v4-human-examples")

    client = anthropic.Anthropic(max_retries=5)
    decode_cache, judge_cache = load_cache(), load_judge_cache()
    scores, out = {}, []
    print(f"\n{'model':<20}{'valid':>6}{'clean':>6}{'FR':>6}{'EN':>6}{'all':>6}{'  0/1/2':>10}{'➡️':>6}{'💬':>5}{'⏪⏩':>5}")
    for name, outputs in models.items():
        scores[name] = []
        for row in dev:
            output = outputs[row["id"]]
            status = check_validity(output)
            guess, gist = "", 0
            if status == "ok":
                guess = decode(client, decode_cache, output, row["lang"])
                gist = judge(client, judge_cache, row["sentence"], guess)[0]
            scores[name].append(gist)
            out.append({"model": name, "id": row["id"], "lang": row["lang"], "sentence": row["sentence"],
                        "output": output, "status": status, "decoded": guess, "gist": gist})
        s, texts = scores[name], [outputs[r["id"]] for r in dev]
        fr = [g for g, r in zip(s, dev) if r["lang"] == "fr"]
        en = [g for g, r in zip(s, dev) if r["lang"] == "en"]
        valid = [t for t in texts if check_validity(t) == "ok"]
        print(f"{name:<20}{len(valid):>6}{sum(not check_conventions(t) for t in valid):>6}"
              f"{sum(fr) / len(fr):>6.2f}{sum(en) / len(en):>6.2f}{sum(s) / len(s):>6.2f}"
              f"{f'{s.count(0)}/{s.count(1)}/{s.count(2)}':>10}{sum('➡' in t for t in texts):>6}"
              f"{sum('💬' in t for t in texts):>5}{sum('⏪' in t or '⏩' in t for t in texts):>5}")
    print()
    for name in models:
        if name != BASELINE:
            print(f"{name} - {BASELINE}: " + interval([a - b for a, b in zip(scores[name], scores[BASELINE])]))
    SCORES_FILE.parent.mkdir(exist_ok=True)
    with open(SCORES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
