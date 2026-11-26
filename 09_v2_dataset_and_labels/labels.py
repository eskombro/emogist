"""Flash label candidates for a sentence, and the best-of-n choice.

Candidate 1 is the v1 setup (temperature 0.3). If Flash's own round trip gives it less than full
marks, more candidates are drawn at a higher temperature and the best valid one wins (ties: the
earliest). Plain code and Flash only: Claude never sees or picks a training label.
Candidates are cached in cache/flash_label_cache.csv, so reruns cost nothing.
"""

import hashlib
import re
import sys
import threading
from pathlib import Path

import requests

from emoji_eval.roundtrip import append_cache, read_cache
from emoji_eval.runner import TEMPERATURE
from emoji_eval.validate import check_conventions_v2, check_validity_v2
from flash_roundtrip import MODEL, URL, FlashRoundTrip

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "03_dataset"))
import label_prompt as v1  # noqa: E402  (step 3 prompt, used by the pilot)
import label_prompt_v2 as v2  # noqa: E402

CACHE = REPO / "cache" / "flash_label_cache.csv"
# prompt -> (cache id: version @ worksheet hash, messages for a sentence)
PROMPTS = {
    "v1": (f"v4-human-examples@{hashlib.sha256(v1.WORKSHEET.read_bytes()).hexdigest()[:8]}",
           lambda sentence: v1.messages_for("v4-human-examples", sentence)),
    # The v2 id only hashes the worksheet, so it must not be used for new requests once the rules
    # text changes: the full run's labels are reused from the cache under it, and v2.1 (rules text
    # included in the hash) labels what is missing.
    "v2": (f"{v2.VERSION}@{hashlib.sha256(v2.WORKSHEET.read_bytes()).hexdigest()[:8]}", v2.messages_for),
    "v2.2": (f"{v2.VERSION_7}@{hashlib.sha256(v2.WORKSHEET.read_bytes() + v2.SYSTEM_PROMPT_V2.encode()).hexdigest()[:8]}",
             v2.messages_for),
    "v2.1": (f"{v2.VERSION_6}@{hashlib.sha256(v2.WORKSHEET.read_bytes() + v2.SYSTEM_PROMPT_V2.encode()).hexdigest()[:8]}",
             v2.messages_for),
}
EXTRA_TEMPERATURE = 0.8
EXTRA = 3  # extra candidates when the first one loses something


def problem(output: str) -> str:
    validity = check_validity_v2(output)
    return validity if validity != "ok" else "; ".join(w for w in check_conventions_v2(output)
                                                        if w != "fewer than 3 emojis")


def space_question_marks(output: str) -> str:
    """Format fix (author's decision, prompt v7 on): put ❓ in its own group when Flash glued it to a
    neighbour (👥❓ -> 👥 ❓). Only spaces change: no emoji is added, removed or chosen."""
    output = re.sub(r"(?<=[^\s])❓", " ❓", output)
    return re.sub(r"❓(?=[^\s\ufe0f])", "❓ ", output)


def strict_problem(output: str, sentence: str) -> str:
    """problem(), plus plain rules that need the sentence (prompt v7 on, after the author's review):
    no copy of an example label, ❓ exactly when the sentence has a "?", and no dropped sign."""
    found = [p for p in [problem(output)] if p]
    if output in v2.example_labels():
        found.append("copy of an example")
    if "?" in sentence and "❓" not in output:
        found.append("question without ❓")
    if "❓" in output and "?" not in sentence:
        found.append("❓ without a question")
    if any(sign in output for sign in v2.DROPPED):
        found.append("dropped sign")
    return "; ".join(found)


# How each broken rule is explained to Flash in a repair request. Only the rule is named: the code
# never suggests emojis, so the new label is still written by Flash alone.
HINTS = {
    "emojis": "it has {n} emojis; the limit is 10 (➡️ not counted). Keep only the main event and its result",
    "non-emoji": "it contains characters that are not emojis",
    "multiple lines": "it is on several lines; answer on a single line",
    "❓ not alone in its group": "❓ must be a group on its own, right after the question part",
    "alone in a group": "{sign} must be attached to the emoji it applies to, not a group on its own",
    "❔ in a question": "a question never uses ❔: remove the ❔, the ❓ group already marks the question",
    "copy of an example": "it repeats an example answer; write the emoji sentence for this sentence",
    "question without ❓": "the sentence asks a question, so the question part needs ❓ as its own group",
    "❓ without a question": "the sentence is not a question, so it has no ❓",
    "dropped sign": "🔀 and 🔄 are not used; show a condition with ➡️ and \"every / all\" with plain emojis",
}


def hint(problems: str) -> str:
    """Plain-language version of problem()'s answer, for the repair request."""
    out = []
    for p in problems.split("; "):
        if p.endswith(" emojis") and p.split()[0].isdigit():
            out.append(HINTS["emojis"].format(n=p.split()[0]))
        elif p.endswith("alone in a group"):
            out.append(HINTS["alone in a group"].format(sign=p.split()[0]))
        else:
            out.append(next((v for k, v in HINTS.items() if p.startswith(k)), p))
    return "; ".join(out)


class Labeller:
    """prompt: "v1" (the step 3 teacher prompt, used by the pilot) or "v2" (label_prompt_v2.py).
    tries: how many seeds first() may use before giving up (2 = the v1 way: one retry).
    repairs: after that, how many times Flash is shown its last broken label and the rule it breaks,
    and asked to fix it (0 = the v1 way). Safe to call from several threads."""

    def __init__(self, prompt: str = "v1", tries: int = 2, repairs: int = 0, strict: bool = False) -> None:
        self.prompt_id, self.messages_for = PROMPTS[prompt]
        self.strict = strict  # also apply strict_problem's sentence rules and the ❓ spacing (prompt v7 on)
        self.spaced = 0  # labels accepted after the ❓ spacing fix
        self.tries, self.repairs = tries, repairs
        self.cache = read_cache(CACHE, ["prompt", "sentence", "temperature", "seed"])
        self.flash = FlashRoundTrip()
        self.requests = 0  # Flash label requests actually sent (not cached)
        self.tokens = {"prompt": 0, "output": 0}  # of those requests, to estimate the Ollama cost
        self.lock = threading.Lock()

    def candidate(self, sentence: str, temperature: float, seed: int, broken: str = "") -> str:
        """One Flash label. With broken, a repair: Flash sees its broken label and the rule it breaks."""
        prompt_id = f"{self.prompt_id}|repair:{broken}" if broken else self.prompt_id
        key = (prompt_id, sentence, str(temperature), str(seed))
        if key not in self.cache:
            messages = self.messages_for(sentence)
            if broken:
                messages += [{"role": "assistant", "content": broken},
                             {"role": "user", "content": f"This breaks a rule: {hint(self.check(broken, sentence))}. "
                                                         "Write the emoji sentence again, following all the rules."}]
            response = requests.post(URL, timeout=120, json={
                "model": MODEL, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": temperature, "num_predict": 100, "seed": seed}})
            response.raise_for_status()
            row = {"prompt": prompt_id, "sentence": sentence, "temperature": str(temperature),
                   "seed": str(seed), "emojis": response.json()["message"]["content"].strip()}
            body = response.json()
            with self.lock:
                self.requests += 1
                self.tokens["prompt"] += body.get("prompt_eval_count", 0)
                self.tokens["output"] += body.get("eval_count", 0)
                self.cache[key] = row
                append_cache(CACHE, row)
        return self.cache[key]["emojis"]

    def fix(self, output: str) -> str:
        if not self.strict:
            return output
        fixed = space_question_marks(output)
        if fixed != output:
            with self.lock:
                self.spaced += 1
        return fixed

    def check(self, output: str, sentence: str) -> str:
        return strict_problem(output, sentence) if self.strict else problem(output)

    def first(self, sentence: str) -> str:
        """The label only (see label)."""
        return self.label(sentence)[0]

    def label(self, sentence: str) -> tuple[str, int]:
        """(label, attempt that produced it). Up to self.tries seeds while the format is wrong (the
        first two at 0.3, any further ones at EXTRA_TEMPERATURE), then up to self.repairs repair
        requests on the last broken label. ("", 0) if all fail."""
        attempt = 0
        for seed in range(1, self.tries + 1):
            attempt += 1
            output = self.fix(self.candidate(sentence, TEMPERATURE if seed <= 2 else EXTRA_TEMPERATURE, seed))
            if not self.check(output, sentence):
                return output, attempt
        for _ in range(self.repairs):
            attempt += 1
            output = self.fix(self.candidate(sentence, TEMPERATURE, 1, broken=output))
            if not self.check(output, sentence):
                return output, attempt
        return "", 0

    def best(self, sentence: str, lang: str) -> dict:
        """Best-of-n label with every candidate and its Flash round-trip score."""
        candidates = []
        first = self.first(sentence)
        if first:
            candidates.append(first)
        scored = [{"emojis": c, **self.flash.score(sentence, c, lang)} for c in candidates]
        if not scored or scored[0]["score"] < 1:
            for seed in range(1, EXTRA + 1):
                output = self.candidate(sentence, EXTRA_TEMPERATURE, seed)
                if not problem(output) and output not in [s["emojis"] for s in scored]:
                    scored.append({"emojis": output, **self.flash.score(sentence, output, lang)})
        if not scored:
            return {"first": "", "best": "", "candidates": []}
        best = max(scored, key=lambda s: s["score"])  # max keeps the earliest on ties
        return {"first": first, "best": best["emojis"], "candidates": scored}
