"""Round trip with DeepSeek V4.1 Flash on Ollama Cloud: decode an emoji sentence, then score the guess.

Used to choose among label candidates (best-of-n). Claude is never used here: these scores pick
training labels. The prompts are the same as the Claude decoder and the v2.1 judge
(emoji_eval/roundtrip.py), so the selection follows the rubric the author validated.
Results are cached in cache/, so reruns cost nothing.
"""

import json
import re
from pathlib import Path

import requests

from emoji_eval.roundtrip import (DECODER_PROMPT, DECODER_SCHEMA, JUDGE_PROMPT_V21, JUDGE_V2_SCHEMA, LANGUAGES,
                                  append_cache, criteria_score, read_cache)

REPO = Path(__file__).resolve().parent.parent
MODEL = "deepseek-v4.1-flash:cloud"
URL = "http://localhost:11434/api/chat"
DECODE_CACHE = REPO / "cache" / "flash_decode_cache.csv"
JUDGE_CACHE = REPO / "cache" / "flash_judge_v21_cache.csv"


# Ollama Cloud ignores the JSON schema for this model, so the format is also asked for in words
# (the only addition to the shared prompts) and the answer is checked against the schema here.
FORMAT_NOTE = "\n\nAnswer with JSON only, no other text, with these keys: {keys}."


def parse(text: str, schema: dict) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in {text[:80]!r}")
    data = json.loads(match.group(0))
    for key, spec in schema["properties"].items():
        data[key] = str(data[key]).strip()
        if "enum" in spec and data[key] not in spec["enum"]:
            raise ValueError(f"{key} = {data[key]!r}")
    return data


def chat(system: str, user: str, schema: dict, max_tokens: int) -> dict:
    """One Flash call at temperature 0, no thinking, JSON answer checked against the schema (one retry)."""
    system += FORMAT_NOTE.format(keys=", ".join(f'"{k}"' for k in schema["properties"]))
    for attempt in range(2):
        response = requests.post(URL, timeout=180, json={
            "model": MODEL, "stream": False, "think": False, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": max_tokens}})
        response.raise_for_status()
        try:
            return parse(response.json()["message"]["content"], schema)
        except (ValueError, KeyError) as error:
            if attempt:
                raise RuntimeError(f"Flash gave no valid answer: {error}") from error


class FlashRoundTrip:
    def __init__(self) -> None:
        self.decodes = {k: r["decoded"] for k, r in read_cache(DECODE_CACHE, ["emojis", "lang"]).items()}
        self.judgements = read_cache(JUDGE_CACHE, ["original", "guess"])

    def decode(self, emojis: str, lang: str) -> str:
        if (emojis, lang) not in self.decodes:
            data = chat(DECODER_PROMPT, f"Language: {LANGUAGES[lang]}\nEmoji sentence: {emojis}", DECODER_SCHEMA, 200)
            self.decodes[(emojis, lang)] = data["sentence"].strip()
            append_cache(DECODE_CACHE, {"emojis": emojis, "lang": lang, "decoded": self.decodes[(emojis, lang)]})
        return self.decodes[(emojis, lang)]

    def judge(self, original: str, guess: str) -> dict:
        """topic, event, outcome ("0"-"2", outcome may be "n/a"), reason, and score from 0 to 1."""
        if (original, guess) not in self.judgements:
            data = chat(JUDGE_PROMPT_V21, f"Original: {original}\nGuess: {guess}", JUDGE_V2_SCHEMA, 400)
            row = {"original": original, "guess": guess, **{f: str(data[f]) for f in ("topic", "event", "outcome")},
                   "reason": data["reason"]}
            self.judgements[(original, guess)] = row
            append_cache(JUDGE_CACHE, row)
        r = self.judgements[(original, guess)]
        return {f: r[f] for f in ("topic", "event", "outcome", "reason")} | {
            "score": criteria_score(r["topic"], r["event"], r["outcome"])}

    def score(self, original: str, emojis: str, lang: str) -> dict:
        """Full round trip for one label: Flash decodes the emojis, then scores its own guess."""
        guess = self.decode(emojis, lang)
        return {"guess": guess, **self.judge(original, guess)}
