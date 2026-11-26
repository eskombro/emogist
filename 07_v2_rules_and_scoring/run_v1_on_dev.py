"""Run the v1 model (the served 8-bit GGUF) on the v2 dev set, in English, French and Spanish.

    python 07_v2_rules_and_scoring/run_v1_on_dev.py --llama-server /path/to/llama-server

Same prompt and settings as step 6 (temperature 0). Writes results/v1_dev_outputs.csv with the
v1 and v2 validity of each output. No API calls: the model runs locally.
The v1 model was never trained on Spanish, so its Spanish outputs are a zero-shot baseline.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "06_serving"))

from compare_gguf import build_prompt  # noqa: E402  (step 6, same prompt as the demo)
from emoji_eval.validate import check_validity, check_validity_v2  # noqa: E402

DEV = REPO / "08_v2_spanish_and_eval_sets" / "sets" / "dev.csv"
MODEL = REPO / "demo" / "models" / "emogist-qwen2b-Q8_0.gguf"
OUT = HERE / "results" / "v1_dev_outputs.csv"
PORT = 18093
URL = f"http://127.0.0.1:{PORT}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--gpu-layers", type=int, default=99)
    args = parser.parse_args()
    with open(DEV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with tempfile.TemporaryFile("w") as log:
        process = subprocess.Popen([args.llama_server, "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
                                    "-ngl", str(args.gpu_layers), "-c", "512", "-np", "1"], stdout=log, stderr=log)
        try:
            for _ in range(480):
                if process.poll() is not None:
                    raise SystemExit(f"llama-server exited with code {process.returncode}")
                try:
                    if requests.get(f"{URL}/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.25)
            out = []
            for row in rows:
                response = requests.post(f"{URL}/completion", timeout=120, json={
                    "prompt": build_prompt(row["sentence"]), "n_predict": 60, "temperature": 0, "top_k": 1,
                    "cache_prompt": False, "seed": 1})
                response.raise_for_status()
                output = response.json()["content"].strip()
                out.append({"id": row["id"], "lang": row["lang"], "source": row["source"], "type": row["type"],
                            "sentence": row["sentence"], "output": output,
                            "valid_v1": check_validity(output), "valid_v2": check_validity_v2(output)})
        finally:
            process.terminate()
            process.wait(timeout=30)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    for lang in ("en", "fr", "es"):
        lang_rows = [r for r in out if r["lang"] == lang]
        print(f"[{lang}] valid v1 {sum(r['valid_v1'] == 'ok' for r in lang_rows)}/{len(lang_rows)}, "
              f"v2 {sum(r['valid_v2'] == 'ok' for r in lang_rows)}/{len(lang_rows)}")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
