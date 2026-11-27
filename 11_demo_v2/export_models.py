"""Build the three model files the v2 demo serves.

    pip install -r 06_serving/requirements-export.txt huggingface_hub
    python 11_demo_v2/export_models.py --llama-src /path/to/llama.cpp --llama-quantize /path/to/llama-quantize \
        --out demo/models

1. qwen2b-base-Q8_0.gguf: the untouched Qwen3.5 2B, converted to f16 and quantized to Q8_0 (as in step 6).
2. emogist-v1-adapter-f16.gguf: the step 4 adapter (run_qwen2b_full).
3. emogist-v2-adapter-f16.gguf: the step 10 main adapter (run_qwen2b_v2_main).

llama-server loads the base with both adapters, and the demo switches them per request.
The converters come from the llama.cpp source of the same build as the server (b12359).
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parent.parent
BASE = "unsloth/Qwen3.5-2B"
ADAPTERS = {
    "emogist-v1-adapter-f16.gguf": REPO / "04_fine_tuning" / "runs" / "run_qwen2b_full" / "adapter",
    "emogist-v2-adapter-f16.gguf": REPO / "10_v2_fine_tuning" / "runs" / "run_qwen2b_v2_main" / "adapter",
}


def run(command: list[str], llama_src: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(llama_src / "gguf-py")}
    subprocess.run(command, check=True, env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-src", type=Path, required=True, help="llama.cpp source folder (build b12359)")
    parser.add_argument("--llama-quantize", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base_dir = Path(snapshot_download(BASE))

    base_out = args.out / "qwen2b-base-Q8_0.gguf"
    if not base_out.exists():
        with tempfile.TemporaryDirectory() as tmp:
            f16 = Path(tmp) / "base-f16.gguf"
            run([sys.executable, str(args.llama_src / "convert_hf_to_gguf.py"), str(base_dir),
                 "--outtype", "f16", "--outfile", str(f16)], args.llama_src)
            subprocess.run([args.llama_quantize, str(f16), str(base_out), "Q8_0"], check=True)

    for name, adapter in ADAPTERS.items():
        out = args.out / name
        if not out.exists():
            run([sys.executable, str(args.llama_src / "convert_lora_to_gguf.py"), "--base", str(base_dir),
                 "--outtype", "f16", "--outfile", str(out), str(adapter)], args.llama_src)

    for name in ("qwen2b-base-Q8_0.gguf", *ADAPTERS):
        path = args.out / name
        print(f"{name}  {path.stat().st_size / 1e6:,.0f} MB  sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
