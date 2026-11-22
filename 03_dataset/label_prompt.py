"""Versions of the labelling prompt, compared on the real dev set by compare_prompts.py.

The shared prompt in emoji_eval/prompt.py stays frozen, so steps 1 and 2 remain comparable.
"""

import csv
from pathlib import Path

from emoji_eval.prompt import EXAMPLES, SYSTEM_PROMPT
from emoji_eval.validate import check_conventions, check_validity

WORKSHEET = Path(__file__).resolve().parent / "label_examples.csv"

OLD_RULE_1 = '1. Use 3 to 10 emojis. Never use letters from any alphabet, digits, or punctuation such as "...".'
NEW_RULE_1 = ('1. Use 3 to 10 emojis. Never use letters from any alphabet, digits, keycap number emojis '
              '(1️⃣, 2️⃣, 🔟), or punctuation such as "...". To show a quantity, repeat an emoji (☕☕☕) '
              'or leave the number out.')
# v2's ban lost information (ages, times) and sometimes repeated an emoji a dozen times,
# so v3 allows keycaps when the exact number matters and caps repetition at 4.
NUMBERS_RULE_1 = ('1. Use 3 to 10 emojis. Never use letters from any alphabet, plain digits, or punctuation '
                  'such as "...". For a small quantity, repeat an emoji up to 4 times (☕☕☕☕). When the '
                  'exact number matters (an age, a time, a price), use keycap number emojis (3️⃣0️⃣).')
assert OLD_RULE_1 in SYSTEM_PROMPT

PROMPTS = {
    "v1-current": SYSTEM_PROMPT,
    "v2-no-keycaps": SYSTEM_PROMPT.replace(OLD_RULE_1, NEW_RULE_1),
    "v3-keycaps-allowed": SYSTEM_PROMPT.replace(OLD_RULE_1, NUMBERS_RULE_1),
    "v4-human-examples": SYSTEM_PROMPT.replace(OLD_RULE_1, NUMBERS_RULE_1),
}


def worksheet_examples() -> list[tuple[str, str]]:
    """The human-reviewed examples: worksheet rows with emojis. A broken one stops the run."""
    with open(WORKSHEET, newline="", encoding="utf-8") as f:
        examples = [(r["sentence"], r["emojis"].strip()) for r in csv.DictReader(f) if r["emojis"].strip()]
    for sentence, emojis in examples:
        if check_validity(emojis) != "ok" or check_conventions(emojis):
            raise ValueError(f"Bad example: {sentence} -> {emojis}")
    return examples


def messages_for(version: str, sentence: str) -> list[dict]:
    examples = worksheet_examples() if version == "v4-human-examples" else EXAMPLES
    messages = [{"role": "system", "content": PROMPTS[version]}]
    for text, emojis in examples:
        messages += [{"role": "user", "content": text}, {"role": "assistant", "content": emojis}]
    messages.append({"role": "user", "content": sentence})
    return messages
