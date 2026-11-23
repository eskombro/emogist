"""Merge the fine-tuned adapter into Qwen3.5 2B and check that the merged model still matches Colab.

    python 06_serving/merge_adapter.py --out /path/to/merged

The base must load with the full image-text class. The text-only class has different layer names,
so the adapter would silently stay inactive.
"""

import argparse
import csv
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "04_fine_tuning" / "runs" / "run_qwen2b_full"
ADAPTER = RUN / "adapter"
BASE = "unsloth/Qwen3.5-2B"
SYSTEM_PROMPT = "Convert the sentence into a short emoji sentence that keeps its gist. Answer with emojis only."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder for the merged model")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    base = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.float16, low_cpu_mem_usage=True)
    peft_model = PeftModel.from_pretrained(base, ADAPTER)
    lora_layers = sum("lora_B" in name for name, _ in peft_model.named_parameters())
    if not lora_layers:
        raise SystemExit("The adapter did not attach to the base model")
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(args.out)
    for name in ("chat_template.jinja", "processor_config.json"):
        shutil.copy(ADAPTER / name, args.out / name)
    print(f"Merged {lora_layers} adapter layers into {args.out}")
    del merged, peft_model, base

    model = AutoModelForImageTextToText.from_pretrained(args.out, dtype=torch.float16, low_cpu_mem_usage=True)
    model = model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.out)
    with open(RUN / "dev_outputs_qwen2b_full.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    same = 0
    for row in rows:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["sentence"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=60, do_sample=False, use_cache=True,
                                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        same += answer == row["output"]
    print(f"{same}/{len(rows)} dev outputs identical to the Colab run")


if __name__ == "__main__":
    main()
