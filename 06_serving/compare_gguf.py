"""Run each GGUF version on the 100 dev sentences and compare with the saved Colab outputs.

    python 06_serving/compare_gguf.py --llama-server /path/to/llama-server --gguf-dir /path/to/gguf

Expects emogist-f16.gguf, emogist-Q8_0.gguf and emogist-Q4_K_M.gguf in the GGUF folder.
The test set stays closed: only dev sentences are used.
"""

import argparse
import csv
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from emoji_eval.validate import check_validity

HERE = Path(__file__).resolve().parent
DEV_OUTPUTS = HERE.parent / "04_fine_tuning" / "runs" / "run_qwen2b_full" / "dev_outputs_qwen2b_full.csv"
RESULTS = HERE / "results"
VERSIONS = ("f16", "Q8_0", "Q4_K_M")
PORT = 18080
URL = f"http://127.0.0.1:{PORT}"
SYSTEM_PROMPT = "Convert the sentence into a short emoji sentence that keeps its gist. Answer with emojis only."


def build_prompt(sentence: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{sentence}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def wait_until_ready(process: subprocess.Popen) -> None:
    for _ in range(480):
        if process.poll() is not None:
            raise SystemExit(f"llama-server exited with code {process.returncode}")
        try:
            if requests.get(f"{URL}/health", timeout=1).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise SystemExit("llama-server did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--gguf-dir", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=99)
    args = parser.parse_args()
    with open(DEV_OUTPUTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    RESULTS.mkdir(exist_ok=True)

    print(f"{'version':<8}{'size':>9}{'identical':>11}{'valid':>7}{'s each':>8}")
    for version in VERSIONS:
        path = args.gguf_dir / f"emogist-{version}.gguf"
        with tempfile.TemporaryFile("w") as log:
            process = subprocess.Popen([args.llama_server, "-m", str(path), "--host", "127.0.0.1", "--port", str(PORT),
                                        "-ngl", str(args.gpu_layers), "-c", "512", "-np", "1"],
                                       stdout=log, stderr=log)
            try:
                wait_until_ready(process)
                started = time.time()
                outputs = []
                for row in rows:
                    response = requests.post(f"{URL}/completion", timeout=120, json={
                        "prompt": build_prompt(row["sentence"]), "n_predict": 60, "temperature": 0, "top_k": 1,
                        "cache_prompt": False, "seed": 1})
                    response.raise_for_status()
                    outputs.append({"id": row["id"], "lang": row["lang"], "sentence": row["sentence"],
                                    "output": response.json()["content"].strip()})
                seconds = (time.time() - started) / len(rows)
            finally:
                process.terminate()
                process.wait(timeout=30)
        same = sum(o["output"] == r["output"] for o, r in zip(outputs, rows))
        valid = sum(check_validity(o["output"]) == "ok" for o in outputs)
        print(f"{version:<8}{path.stat().st_size / 1e9:>6.2f} GB{same:>11}{valid:>7}{seconds:>8.2f}")
        with open(RESULTS / f"gguf_dev_{version}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(outputs[0]))
            writer.writeheader()
            writer.writerows(outputs)


if __name__ == "__main__":
    main()
