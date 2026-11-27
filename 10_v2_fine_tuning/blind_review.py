"""Blind review of v1 vs v2 by Sonnet, from the point of view of a first-time user.

    python 10_v2_fine_tuning/blind_review.py --sentences                    # 1 Sonnet call: 100 test sentences
    python 10_v2_fine_tuning/blind_review.py --convert --llama-server PATH \
        --base-gguf PATH --v2-lora PATH                                     # local, no API
    python 10_v2_fine_tuning/blind_review.py --evaluate                     # 2 Sonnet calls, A/B swapped

The test sentences are written by Claude for evaluation only: they never go into training data.
Sonnet sees the two converters as A and B and is told nothing about the project.
v1 = the served 8-bit GGUF with its instruction; v2 = the main run's adapter on the 8-bit base
with the v2 instruction (the same setup as the local v2 demo).
Files go to results/blind_review/; each stage reuses what an earlier run saved.
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
load_dotenv(REPO / ".env")
OUT = HERE / "results" / "blind_review"
SENTENCES = OUT / "sentences.csv"
OUTPUTS = OUT / "outputs.csv"
MODEL = "claude-sonnet-5-5"
V1_GGUF = REPO / "demo" / "models" / "emogist-qwen2b-Q8_0.gguf"
PROMPTS = {
    "v1": "Convert the sentence into a short emoji sentence that keeps its gist. Answer with emojis only.",
    "v2": "Convert the sentence into an emoji sentence that keeps its gist. Answer with emojis only.",
}

SENTENCE_PROMPT = """Write 100 sentences to test how good a text-to-emoji converter is. The converter takes one sentence in English, French or Spanish and returns a sequence of emojis meant to convey that sentence.

Write a varied test set that would show its strengths and weaknesses. Mix the three languages."""

SENTENCE_SCHEMA = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "items": {
        "type": "object",
        "properties": {"lang": {"type": "string", "enum": ["en", "fr", "es"]}, "sentence": {"type": "string"}},
        "required": ["lang", "sentence"], "additionalProperties": False}}},
    "required": ["sentences"], "additionalProperties": False,
}

EVAL_PROMPT = """Below are {n} sentences. Two different text-to-emoji converters, A and B, turned each sentence into emojis.

Judge them from the point of view of a user who knows nothing about these tools: they type a sentence and see only the emojis that come back.

Evaluate the two converters. Say which criteria you use, compare A and B, and give your conclusions. Then, for each sentence, say which output you prefer: A, B or tie.

{rows}"""

EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name", "description"], "additionalProperties": False}},
        "comparison": {"type": "string"},
        "conclusions": {"type": "string"},
        "preferences": {"type": "array", "items": {
            "type": "object",
            "properties": {"n": {"type": "integer"}, "preferred": {"type": "string", "enum": ["A", "B", "tie"]}},
            "required": ["n", "preferred"], "additionalProperties": False}},
    },
    "required": ["criteria", "comparison", "conclusions", "preferences"], "additionalProperties": False,
}


def ask(client: anthropic.Anthropic, prompt: str, schema: dict, effort: str) -> tuple[dict, dict]:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise SystemExit(f"Sonnet stopped: {response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    usage = {"model": response.model, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    return json.loads(text), usage


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_sentences(client: anthropic.Anthropic) -> None:
    if SENTENCES.exists():
        print(f"{SENTENCES.relative_to(REPO)} exists: keeping it")
        return
    data, usage = ask(client, SENTENCE_PROMPT, SENTENCE_SCHEMA, "medium")
    rows = [{"n": i, "lang": s["lang"], "sentence": s["sentence"].strip()} for i, s in enumerate(data["sentences"], 1)]
    print(f"Sonnet wrote {len(rows)} sentences ({usage})")
    if len(rows) < 100:
        raise SystemExit("Fewer than 100 sentences: not saved")
    write_csv(SENTENCES, rows[:100])
    (OUT / "sentences_usage.json").write_text(json.dumps(usage, indent=1))
    for lang in ("en", "fr", "es"):
        print(f"  {lang}: {sum(r['lang'] == lang for r in rows[:100])}")


def run_model(server: str, gguf: Path, lora: Path | None, prompt: str, sentences: list[dict], port: int) -> list[str]:
    command = [server, "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-ngl", "99",
               "-c", "512", "-np", "1", "--no-webui"]
    if lora:
        command += ["--lora", str(lora)]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(480):
                if process.poll() is not None:
                    raise SystemExit(f"llama-server exited with code {process.returncode}")
                try:
                    if requests.get(f"{url}/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.25)
            outputs = []
            for row in sentences:
                text = (f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{row['sentence']}<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
                response = requests.post(f"{url}/completion", timeout=120, json={
                    "prompt": text, "n_predict": 60, "temperature": 0, "top_k": 1, "cache_prompt": False, "seed": 1})
                response.raise_for_status()
                outputs.append(response.json()["content"].strip())
            return outputs
        finally:
            process.terminate()
            process.wait(timeout=30)


def convert(args: argparse.Namespace) -> None:
    from emoji_eval.validate import check_validity_v2
    sentences = read_csv(SENTENCES)
    v1 = run_model(args.llama_server, V1_GGUF, None, PROMPTS["v1"], sentences, 18101)
    v2 = run_model(args.llama_server, Path(args.base_gguf), Path(args.v2_lora), PROMPTS["v2"], sentences, 18102)
    rows = [{**s, "v1": a, "v2": b, "v1_valid": check_validity_v2(a), "v2_valid": check_validity_v2(b)}
            for s, a, b in zip(sentences, v1, v2)]
    write_csv(OUTPUTS, rows)
    print(f"Wrote {OUTPUTS.relative_to(REPO)}; identical outputs: {sum(r['v1'] == r['v2'] for r in rows)}/{len(rows)}")


def evaluate(client: anthropic.Anthropic) -> None:
    rows = read_csv(OUTPUTS)
    for name, a in (("eval_1", "v1"), ("eval_2", "v2")):
        path = OUT / f"{name}.json"
        if path.exists():
            print(f"{path.relative_to(REPO)} exists: keeping it")
            continue
        b = "v2" if a == "v1" else "v1"
        lines = "\n\n".join(f"{r['n']}. {r['sentence']}\nA: {r[a]}\nB: {r[b]}" for r in rows)
        data, usage = ask(client, EVAL_PROMPT.format(n=len(rows), rows=lines), EVAL_SCHEMA, "high")
        path.write_text(json.dumps({"A": a, "B": b, "usage": usage, **data}, indent=1, ensure_ascii=False))
        print(f"Wrote {path.relative_to(REPO)} ({usage})")
    summarise(rows)


def summarise(rows: list[dict]) -> None:
    langs = {r["n"]: r["lang"] for r in rows}
    for name in ("eval_1", "eval_2"):
        data = json.loads((OUT / f"{name}.json").read_text())
        to_model = {"A": data["A"], "B": data["B"], "tie": "tie"}
        prefs = {p["n"]: to_model[p["preferred"]] for p in data["preferences"]}
        print(f"\n[{name}] A = {data['A']}, B = {data['B']}, {len(prefs)} preferences")
        for scope in ("all", "en", "fr", "es"):
            picks = [m for n, m in prefs.items() if scope == "all" or langs.get(str(n)) == scope]
            print(f"  {scope:<4} v1 {picks.count('v1'):>3}  v2 {picks.count('v2'):>3}  tie {picks.count('tie'):>3}")


def main() -> None:
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--sentences", action="store_true")
    stage.add_argument("--convert", action="store_true")
    stage.add_argument("--evaluate", action="store_true")
    stage.add_argument("--summary", action="store_true", help="reprint the counts, no API calls")
    parser.add_argument("--llama-server")
    parser.add_argument("--base-gguf")
    parser.add_argument("--v2-lora")
    args = parser.parse_args()
    if args.convert:
        convert(args)
    elif args.summary:
        summarise(read_csv(OUTPUTS))
    else:
        client = anthropic.Anthropic(max_retries=5)
        make_sentences(client) if args.sentences else evaluate(client)


if __name__ == "__main__":
    sys.exit(main())
