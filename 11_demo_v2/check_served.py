"""Check the v2 demo's model setup (one base + two adapters) before serving it. Local, no API calls.

    python 11_demo_v2/check_served.py --llama-server PATH [--models demo/models]

1. v2 parity: base + v2 adapter against the step 10 Colab outputs on the 300 v2 dev sentences.
2. v1 parity: base + v1 adapter against the merged v1 GGUF served today (07/results/v1_dev_outputs.csv,
   same dev sentences, Mac GPU).
3. CPU speed: each model on the CPU with 3 threads, as on the server, on the first 30 dev sentences.

Parity runs on the GPU like the files they are compared with. Writes results/served_parity.csv and
results/cpu_timing.csv.
"""

import argparse
import csv
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from emoji_eval.validate import check_validity, check_validity_v2

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROMPTS = {
    "v1": "Convert the sentence into a short emoji sentence that keeps its gist. Answer with emojis only.",
    "v2": "Convert the sentence into an emoji sentence that keeps its gist. Answer with emojis only.",
}
# name: (instruction, adapter scales [v1, v2]); "merged_v1" runs on the merged v1 file without adapters.
MODELS = {"base": ("v2", [0, 0]), "v1": ("v1", [1, 0]), "v2": ("v2", [0, 1]), "merged_v1": ("v1", None)}
V2_COLAB = REPO / "10_v2_fine_tuning" / "runs" / "run_qwen2b_v2_main" / "dev_outputs_qwen2b_v2_main.csv"
V1_MERGED = REPO / "07_v2_rules_and_scoring" / "results" / "v1_dev_outputs.csv"
PORT = 18130


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Server:
    """llama-server with the demo's options; adapters loaded switched off."""

    def __init__(self, binary: str, model: Path, adapters: list[Path], gpu: bool):
        self.command = [binary, "-m", str(model), "--host", "127.0.0.1", "--port", str(PORT), "-c", "512",
                        "-np", "1", "--no-webui", "--cache-ram", "0"]
        for adapter in adapters:
            self.command += ["--lora", str(adapter)]
        if adapters:
            self.command.append("--lora-init-without-apply")
        self.command += ["-ngl", "99"] if gpu else ["-ngl", "0", "-t", "3", "-tb", "3", "--no-repack"]

    def __enter__(self):
        self.log = tempfile.TemporaryFile("w")
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=self.log)
        for _ in range(480):
            if self.process.poll() is not None:
                raise SystemExit(f"llama-server exited with code {self.process.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).ok:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise SystemExit("llama-server did not start")

    def __exit__(self, *_):
        self.process.terminate()
        self.process.wait(timeout=30)
        self.log.close()

    def complete(self, model: str, sentence: str) -> tuple[str, float]:
        prompt, scales = MODELS[model]
        text = (f"<|im_start|>system\n{PROMPTS[prompt]}<|im_end|>\n<|im_start|>user\n{sentence}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        body = {"prompt": text, "n_predict": 60, "temperature": 0, "top_k": 1, "cache_prompt": False, "seed": 1}
        if scales is not None:
            body["lora"] = [{"id": i, "scale": s} for i, s in enumerate(scales)]
        started = time.perf_counter()
        response = requests.post(f"http://127.0.0.1:{PORT}/completion", json=body, timeout=180)
        response.raise_for_status()
        return response.json()["content"].strip(), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--models", type=Path, default=REPO / "demo" / "models")
    parser.add_argument("--timing-sentences", type=int, default=30)
    args = parser.parse_args()
    base = args.models / "qwen2b-base-Q8_0.gguf"
    adapters = [args.models / "emogist-v1-adapter-f16.gguf", args.models / "emogist-v2-adapter-f16.gguf"]
    merged = args.models / "emogist-qwen2b-Q8_0.gguf"
    v2_colab, v1_merged = read_csv(V2_COLAB), read_csv(V1_MERGED)
    assert [r["id"] for r in v2_colab] == [r["id"] for r in v1_merged]

    rows = []
    with Server(args.llama_server, base, adapters, gpu=True) as server:
        for colab, old in zip(v2_colab, v1_merged):
            v2, _ = server.complete("v2", colab["sentence"])
            v1, _ = server.complete("v1", colab["sentence"])
            rows.append({"id": colab["id"], "lang": colab["lang"], "sentence": colab["sentence"],
                         "v2_colab": colab["output"], "v2_served": v2, "v2_valid": check_validity_v2(v2),
                         "v1_merged": old["output"], "v1_served": v1, "v1_valid": check_validity(v1)})
    write_csv(HERE / "results" / "served_parity.csv", rows)
    n = len(rows)
    print(f"v2 served = Colab: {sum(r['v2_served'] == r['v2_colab'] for r in rows)}/{n}, "
          f"valid (v2 rules) {sum(r['v2_valid'] == 'ok' for r in rows)}/{n} "
          f"(Colab {sum(check_validity_v2(r['v2_colab']) == 'ok' for r in rows)}/{n})")
    print(f"v1 adapter = merged v1: {sum(r['v1_served'] == r['v1_merged'] for r in rows)}/{n}, "
          f"valid (v1 rules) {sum(r['v1_valid'] == 'ok' for r in rows)}/{n} "
          f"(merged {sum(check_validity(r['v1_merged']) == 'ok' for r in rows)}/{n})")

    sentences = [r["sentence"] for r in v2_colab[:args.timing_sentences]]
    timing = []
    for label, model_file, names, files in (("merged", merged, ["merged_v1"], []),
                                            ("base+adapters", base, ["base", "v1", "v2"], adapters)):
        with Server(args.llama_server, model_file, files, gpu=False) as server:
            server.complete(names[0], "I can do it now.")  # warm-up read from disk
            for name in names:
                seconds = [server.complete(name, s)[1] for s in sentences]
                timing.append({"setup": label, "model": name, "sentences": len(seconds),
                               "median_s": round(statistics.median(seconds), 3),
                               "mean_s": round(statistics.mean(seconds), 3)})
    write_csv(HERE / "results" / "cpu_timing.csv", timing)
    print("\nCPU, 3 threads (seconds per sentence)")
    for t in timing:
        print(f"  {t['model']:<10} {t['setup']:<14} median {t['median_s']:.2f}  mean {t['mean_s']:.2f}")


if __name__ == "__main__":
    main()
