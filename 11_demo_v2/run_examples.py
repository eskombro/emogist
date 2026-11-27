"""Run the candidate page examples through the three demo models, on one llama-server.

    python 11_demo_v2/run_examples.py --llama-server PATH --base-gguf PATH --v1-lora PATH --v2-lora PATH

The base model is loaded once with both adapters switched off; each request turns on the one it
needs, as the demo will. The candidates are written by Claude for the demo page only: they never
go into training data. Writes examples/outputs.csv. Local, no API calls.
"""

import argparse
import csv
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from emoji_eval.validate import check_validity, check_validity_v2

HERE = Path(__file__).resolve().parent
PROMPTS = {
    "v1": "Convert the sentence into a short emoji sentence that keeps its gist. Answer with emojis only.",
    "v2": "Convert the sentence into an emoji sentence that keeps its gist. Answer with emojis only.",
}
# Adapter scales per model: id 0 = v1 adapter, id 1 = v2 adapter. The untrained model gets the v2 instruction.
MODELS = {"base": ("v2", [0, 0]), "v1": ("v1", [1, 0]), "v2": ("v2", [0, 1])}
PORT = 18120


def complete(prompt: str, sentence: str, scales: list[int]) -> str:
    text = (f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{sentence}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    response = requests.post(f"http://127.0.0.1:{PORT}/completion", timeout=120, json={
        "prompt": text, "n_predict": 60, "temperature": 0, "top_k": 1, "cache_prompt": False, "seed": 1,
        "lora": [{"id": i, "scale": s} for i, s in enumerate(scales)]})
    response.raise_for_status()
    return response.json()["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("--llama-server", "--base-gguf", "--v1-lora", "--v2-lora"):
        parser.add_argument(name, required=True)
    args = parser.parse_args()
    with open(HERE / "examples" / "candidates.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    command = [args.llama_server, "-m", args.base_gguf, "--lora", args.v1_lora, "--lora", args.v2_lora,
               "--lora-init-without-apply", "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99",
               "-c", "512", "-np", "1", "--no-webui"]
    with tempfile.TemporaryFile("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(480):
                if process.poll() is not None:
                    raise SystemExit(f"llama-server exited with code {process.returncode}")
                try:
                    if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.25)
            for row in rows:
                for model, (prompt, scales) in MODELS.items():
                    row[model] = complete(PROMPTS[prompt], row["sentence"], scales)
                row["v1_valid"] = check_validity(row["v1"])
                row["v2_valid"] = check_validity_v2(row["v2"])
        finally:
            process.terminate()
            process.wait(timeout=30)
    with open(HERE / "examples" / "outputs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote examples/outputs.csv ({len(rows)} sentences)")


if __name__ == "__main__":
    main()
