"""Run models on the test set and record validity + convention warnings.

Each step passes its own model list, so students and teachers are run the same way.
"""

import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from emoji_eval.columns import SENTENCE_COLUMNS
from emoji_eval.judgement import judgement_report
from emoji_eval.paths import ENV_FILE, TEST_FILE
from emoji_eval.prompt import build_messages
from emoji_eval.validate import check_conventions, check_validity

load_dotenv(ENV_FILE)

PROVIDERS = {
    "ollama": {"url": "http://localhost:11434/api/chat", "key_env": None},
    "mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "key_env": "MISTRAL_API_KEY"},
    "deepseek": {"url": "https://api.deepseek.com/chat/completions", "key_env": "DEEPSEEK_API_KEY"},
}
PAUSE = {"mistral": 1.5}  # seconds between calls, for free-tier rate limits

TEMPERATURE = 0.3  # low, so the comparison measures the model more than luck


def load_sentences() -> list[dict]:
    """Each row: {"id": "fr-01", "lang": "fr", "sentence": "..."}."""
    with open(TEST_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def call_model(provider: str, model_id: str, sentence: str, max_tokens: int = 100,
               think: bool | str = False) -> str:
    """max_tokens: raise it for models that think before answering (by choice or because
    they ignore think=False). Only the final answer is returned, never the thinking.
    think: True/False, or "low"/"medium"/"high" for models that only take a level
    (gpt-oss can't turn reasoning off, only lower it)."""
    config = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if config["key_env"]:
        key = os.environ.get(config["key_env"])
        if not key:
            raise RuntimeError(f"Set {config['key_env']} to use {provider} models")
        headers["Authorization"] = f"Bearer {key}"

    if provider == "ollama":
        body = {
            "model": model_id,
            "messages": build_messages(sentence),
            "stream": False,
            "think": think,  # Ollama's request field is "think" (the response field is "thinking")
            "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
        }
    else:  # OpenAI-style /chat/completions APIs
        body = {
            "model": model_id,
            "messages": build_messages(sentence),
            "temperature": TEMPERATURE,
            # At least 1000: if a model "thinks" before answering, a low limit
            # would cut it off and return an empty answer.
            "max_tokens": max(max_tokens, 1000),
        }

    for attempt in range(5):
        response = requests.post(config["url"], headers=headers, json=body, timeout=120)
        if response.status_code != 429:
            break
        wait = float(response.headers.get("Retry-After", 2 ** attempt))
        print(f"   429, retrying in {wait:.0f}s")
        time.sleep(wait)
    if not response.ok:
        raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
    data = response.json()
    if provider == "ollama":
        return data["message"]["content"].strip()
    return data["choices"][0]["message"]["content"].strip()


def run_bakeoff(models: list[tuple], results_file: Path) -> None:
    """models: (display name, provider, model id), optionally followed by a settings dict:
    {"max_tokens": n, "think": True}. Writes one row per sentence, with an output, a
    check and a warnings column per model.

    If results_file already exists, its columns are kept and only the models not yet in
    it are run, so adding a candidate doesn't call (and pay for) the others again.
    To rerun a model, delete its three columns, or the whole file."""
    rows = load_sentences()
    existing_fields = []
    if results_file.exists():
        with open(results_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames)
            previous = {r["id"]: r for r in reader}
        for row in rows:
            old = previous.get(row["id"], {})
            row.update({k: v for k, v in old.items() if k not in SENTENCE_COLUMNS})
        kept = [m[0] for m in models if m[0] in existing_fields]
        if kept:
            print(f"Already in {results_file.name}, not rerun: {', '.join(kept)}")
        models = [m for m in models if m[0] not in existing_fields]
        if not models:
            print("Nothing new to run.")
            return

    summary = {}  # (model name, lang) -> number of valid outputs
    warned = {}  # (model name, lang) -> number of valid outputs with at least one warning
    warning_types = {}  # model name -> {warning: count}

    for name, provider, model_id, *settings in models:
        options = settings[0] if settings else {}
        max_tokens, think = options.get("max_tokens", 100), options.get("think", False)
        print(f"\n=== {name} ===")
        for row in rows:
            start = time.time()
            try:
                output = call_model(provider, model_id, row["sentence"], max_tokens, think)
            except Exception as error:  # keep going if one call fails
                output = f"ERROR: {error}"
            status = check_validity(output)
            key = (name, row["lang"])
            summary[key] = summary.get(key, 0) + (status == "ok")
            # Only check conventions on valid outputs: on broken ones they'd be noise.
            warnings = check_conventions(output) if status == "ok" else []
            warned[key] = warned.get(key, 0) + bool(warnings)
            for warning in warnings:
                counts = warning_types.setdefault(name, {})
                counts[warning] = counts.get(warning, 0) + 1
            row[name] = output
            row[f"{name} check"] = status
            row[f"{name} warnings"] = "; ".join(warnings)
            if status != "ok":
                flag = f"   <-- {status}"
            elif warnings:
                flag = f"   (warn: {', '.join(warnings)})"
            else:
                flag = ""
            print(f"{row['id']}. {output}   ({time.time() - start:.1f}s){flag}")
            time.sleep(PAUSE.get(provider, 0))

    fieldnames = existing_fields or list(SENTENCE_COLUMNS)
    for name, *_ in models:
        fieldnames += [name, f"{name} check", f"{name} warnings"]
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    totals = {}
    for row in rows:
        totals[row["lang"]] = totals.get(row["lang"], 0) + 1
    print("\nValid outputs (with convention warnings):")
    for (name, lang), valid in summary.items():
        print(f"  {name} [{lang}]: {valid}/{totals[lang]} valid, {warned[(name, lang)]} with warnings")
    print("\nWarnings by type:")
    for name, *_ in models:
        counts = warning_types.get(name, {})
        details = ", ".join(f"{w}: {n}" for w, n in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"  {name}: {details or 'none'}")
    judgement_report(results_file)
    print(f"\nSaved to {results_file}")
