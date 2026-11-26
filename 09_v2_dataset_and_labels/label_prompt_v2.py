"""The v2 labelling prompt: the v1 teacher prompt (step 3, "v4-human-examples") with the v2 rules.

Changes from v1:
  - rule 1: 1 to 10 emojis, usually 3 to 8 (v2 validity); short answers only for simple sentences;
  - new rules for questions (❓), think/maybe (💭/🤔), unknown (❔), place (📍) and again (🔁),
    as set by the author in CONVENTIONS.md (🔀 for "if" and 🔄 for "every" were tried and dropped);
  - few-shot examples: the author's worksheet label_examples_v2.csv (EN, FR, ES).
The rules text is written from CONVENTIONS.md; every example label is the author's.
"""

import csv
import sys
from pathlib import Path

from emoji_eval.validate import check_conventions_v2, check_validity_v2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "03_dataset"))
from label_prompt import NUMBERS_RULE_1, PROMPTS  # noqa: E402  (step 3 prompt)

WORKSHEET = Path(__file__).resolve().parent / "label_examples_v2.csv"
VERSION = "v5-v2-rules"  # first full run; v6 = ❓ after the question part (see labels.py)
VERSION_6 = "v6-question-anywhere"
VERSION_7 = "v7-you-rule"  # 🫵 only "you", plus the sentence rules in labels.strict_problem
DROPPED = ("🔀", "🔄")  # tried in step 9 and dropped: a reader without the rules reads "shuffle" / "switch"

V1_PROMPT = PROMPTS["v4-human-examples"]
RULE_1_V2 = ('1. Use as few emojis as the idea needs: usually 3 to 8, never more than 10. Use only 1 or 2 when '
             'the sentence is that simple. Never use letters from any alphabet, plain digits, or punctuation '
             'such as "...". For a small quantity, repeat an emoji up to 4 times (☕☕☕☕). When the exact '
             'number matters (an age, a time, a price), use keycap number emojis (3️⃣0️⃣).')
V2_RULES = """
These signs have a fixed meaning:
- Question: if the sentence asks a question (including "why", "when", "what" questions, requests and \
tag questions), put ❓ as its own group right after the question part: at the end when the question \
comes last (🫵🐈🍽️ ❓), after each question when there are several. A question that starts with who, \
what, which or why also gets ❓ and never ❔.
- Condition: for "if", write the "if" part, then ➡️ and what follows (🌧️ ➡️ 🏠).
- 💭 = think, know, believe. 🤔 = maybe, doubt.
- ❔ attached to something unknown or asked about, in a sentence that is not a question \
(🫵💍❔ = whether you are married, 👨📍❔ = where he is).
- 📍 attached to a place or to "where" only, never to "when" or "why" (📍🚽 = where the toilet is).
- 🔁 attached to an action = again, once more (🍔🔁 = eating again).
- ❔, 📍 and 🔁 are always attached to another emoji, never alone in a group.
- 🫵 is only "you", the person being spoken to. "I / me" is always 🙋 and "we / us" is 👥, also inside \
speech."""
assert NUMBERS_RULE_1 in V1_PROMPT
SYSTEM_PROMPT_V2 = V1_PROMPT.replace(NUMBERS_RULE_1, RULE_1_V2).rstrip() + "\n" + V2_RULES


def worksheet_examples() -> list[tuple[str, str]]:
    """The author's examples, in worksheet order. A label breaking a v2 rule stops the run
    (the "fewer than 3 emojis" warning is allowed: short labels are taught on purpose)."""
    with open(WORKSHEET, newline="", encoding="utf-8") as f:
        examples = [(r["sentence"].strip(), r["emojis"].strip()) for r in csv.DictReader(f) if r["emojis"].strip()]
    for sentence, emojis in examples:
        if any(sign in emojis for sign in DROPPED):
            raise ValueError(f"Bad example: {sentence} -> {emojis} (uses a dropped sign: {' '.join(DROPPED)})")
        warnings = [w for w in check_conventions_v2(emojis) if w != "fewer than 3 emojis"]
        if check_validity_v2(emojis) != "ok" or warnings:
            raise ValueError(f"Bad example: {sentence} -> {emojis} ({check_validity_v2(emojis)}, {warnings})")
    return examples


def example_labels() -> set[str]:
    return {emojis for _, emojis in worksheet_examples()}


def messages_for(sentence: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT_V2}]
    for text, emojis in worksheet_examples():
        messages += [{"role": "user", "content": text}, {"role": "assistant", "content": emojis}]
    messages.append({"role": "user", "content": sentence})
    return messages


if __name__ == "__main__":
    print(SYSTEM_PROMPT_V2)
    print(f"\n{len(worksheet_examples())} examples")
