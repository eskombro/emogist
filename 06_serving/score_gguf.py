"""Score the saved GGUF outputs on the 100 dev sentences, against the full-precision GGUF.

    python 06_serving/score_gguf.py

Reads results/gguf_dev_*.csv and results/served_*_dev.csv. Scores come from the shared cache
when the same outputs were already judged.
"""

import csv
import sys
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "04_fine_tuning"))

from score_pilot import interval  # noqa: E402
from emoji_eval.roundtrip import decode, judge, load_cache, load_judge_cache  # noqa: E402
from emoji_eval.validate import check_validity  # noqa: E402

RESULTS = HERE / "results"
FILES = {"f16": "gguf_dev_f16.csv", "Q8_0": "gguf_dev_Q8_0.csv", "Q4_K_M": "gguf_dev_Q4_K_M.csv",
         "served (Mac CPU)": "served_mac_cpu_dev.csv", "served (VPS)": "served_vps_dev.csv"}


def main() -> None:
    client = anthropic.Anthropic(max_retries=5)
    decode_cache, judge_cache = load_cache(), load_judge_cache()
    scores, out = {}, []
    print(f"{'version':<18}{'valid':>6}{'FR':>6}{'EN':>6}{'all':>6}{'  0/1/2':>10}")
    for version, name in FILES.items():
        if not (RESULTS / name).exists():
            print(f"{version:<18}no outputs yet")
            continue
        with open(RESULTS / name, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        gists, valid = [], 0
        for row in rows:
            status = check_validity(row["output"])
            guess, gist = "", 0
            if status == "ok":
                valid += 1
                guess = decode(client, decode_cache, row["output"], row["lang"])
                gist = judge(client, judge_cache, row["sentence"], guess)[0]
            gists.append(gist)
            out.append({"version": version, "id": row["id"], "lang": row["lang"], "sentence": row["sentence"],
                        "output": row["output"], "status": status, "decoded": guess, "gist": gist})
        scores[version] = gists
        by_lang = lambda lang: [g for g, r in zip(gists, rows) if r["lang"] == lang]
        print(f"{version:<18}{valid:>6}{sum(by_lang('fr')) / 50:>6.2f}{sum(by_lang('en')) / 50:>6.2f}"
              f"{sum(gists) / len(gists):>6.2f}{f'{gists.count(0)}/{gists.count(1)}/{gists.count(2)}':>10}")
    print()
    for version, gists in scores.items():
        if version != "f16":
            print(f"{version} - f16: " + interval([a - b for a, b in zip(gists, scores["f16"])]))
    with open(RESULTS / "gguf_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
