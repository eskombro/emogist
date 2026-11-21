"""Run models on the test set and record validity + convention warnings.

Each step passes its own model list, so students and teachers are run the same way.
"""

import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

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


def call_model(provider: str, model_id: str, sentence: str) -> str:
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
            "think": False,  # Ollama's request field is "think" (the response field is "thinking")
            "options": {"temperature": TEMPERATURE, "num_predict": 100},
        }
    else:  # OpenAI-style /chat/completions APIs
        body = {
            "model": model_id,
            "messages": build_messages(sentence),
            "temperature": TEMPERATURE,
            # Generous on purpose: if a model "thinks" before answering, a low limit
            # would cut it off and return an empty answer.
            "max_tokens": 1000,
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


def run_bakeoff(models: list[tuple[str, str, str]], results_file: Path) -> None:
    """models: (display name, provider, model id). Writes one row per sentence,
    with an output, a check and a warnings column per model."""
    rows = load_sentences()
    summary = {}  # (model name, lang) -> number of valid outputs
    warned = {}  # (model name, lang) -> number of valid outputs with at least one warning
    warning_types = {}  # model name -> {warning: count}

    for name, provider, model_id in models:
        print(f"\n=== {name} ===")
        for row in rows:
            start = time.time()
            try:
                output = call_model(provider, model_id, row["sentence"])
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

    fieldnames = ["id", "lang", "sentence"]
    for name, _, _ in models:
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
    for name, _, _ in models:
        counts = warning_types.get(name, {})
        details = ", ".join(f"{w}: {n}" for w, n in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"  {name}: {details or 'none'}")
    print(f"\nSaved to {results_file}")
