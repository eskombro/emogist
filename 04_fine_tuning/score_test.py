"""Score the final test outputs of the full-dataset runs.

    python 04_fine_tuning/score_test.py

Reads runs/run_qwen2b_full/test_outputs_qwen2b_full.csv and, if it exists, the Ministral equivalent.
The baseline is untrained Qwen3.5 2B through Ollama with the full labelling prompt; its outputs are saved and reused.
Qwen was declared the winner on the dev set, so Ministral is reported for comparison only.
"""

import csv
from pathlib import Path

import anthropic

from score_pilot import BASELINE, baseline_outputs, interval, read_outputs
from emoji_eval.roundtrip import decode, judge, load_cache, load_judge_cache
from emoji_eval.validate import check_validity

HERE = Path(__file__).resolve().parent
SENTENCES_FILE = HERE.parent / "data" / "test_sentences.csv"
OUTPUT_FILES = {name: HERE / "runs" / f"run_{name}" / f"test_outputs_{name}.csv"
                for name in ("qwen2b_full", "ministral3b_full")}
BASELINE_FILE = HERE / "runs" / "baseline_qwen2b_prompted_test.csv"
SCORES_FILE = HERE / "results" / "test_scores.csv"


def main() -> None:
    with open(SENTENCES_FILE, newline="", encoding="utf-8") as f:
        sentences = list(csv.DictReader(f))
    expected = {r["id"]: r["sentence"] for r in sentences}
    if not OUTPUT_FILES["qwen2b_full"].exists():
        raise SystemExit(f"Missing outputs: {OUTPUT_FILES['qwen2b_full']}")
    models = {BASELINE: baseline_outputs(sentences, BASELINE_FILE)}
    for name, path in OUTPUT_FILES.items():
        if not path.exists():
            print(f"Skipping {name}: no test outputs yet")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            if {r["id"]: r["sentence"] for r in csv.DictReader(f)} != expected:
                raise SystemExit(f"{path.name} does not match the sentences in {SENTENCES_FILE.name}")
        models[name] = read_outputs(path)

    client = anthropic.Anthropic(max_retries=5)
    decode_cache, judge_cache = load_cache(), load_judge_cache()
    scores, out = {}, []
    print(f"\n{'model':<20}{'valid':>6}{'FR':>6}{'EN':>6}{'all':>6}{'  0/1/2':>10}")
    for name, outputs in models.items():
        gists, valid = [], 0
        for row in sentences:
            output = outputs[row["id"]]
            status = check_validity(output)
            guess, gist = "", 0
            if status == "ok":
                valid += 1
                guess = decode(client, decode_cache, output, row["lang"])
                gist = judge(client, judge_cache, row["sentence"], guess)[0]
            gists.append(gist)
            out.append({"model": name, "id": row["id"], "lang": row["lang"], "sentence": row["sentence"],
                        "output": output, "status": status, "decoded": guess, "gist": gist})
        scores[name] = gists
        fr = [g for g, r in zip(gists, sentences) if r["lang"] == "fr"]
        en = [g for g, r in zip(gists, sentences) if r["lang"] == "en"]
        print(f"{name:<20}{valid:>6}{sum(fr) / len(fr):>6.2f}{sum(en) / len(en):>6.2f}"
              f"{sum(gists) / len(gists):>6.2f}"
              f"{f'{gists.count(0)}/{gists.count(1)}/{gists.count(2)}':>10}")

    print()
    for name in scores:
        if name != BASELINE:
            print(f"{name} - {BASELINE}: " + interval([a - b for a, b in zip(scores[name], scores[BASELINE])]))
    if "ministral3b_full" in scores:
        difference = [a - b for a, b in zip(scores["ministral3b_full"], scores["qwen2b_full"])]
        print("ministral3b_full - qwen2b_full: " + interval(difference))
    with open(SCORES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
