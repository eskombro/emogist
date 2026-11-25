# Emoji sentence conventions

An *emoji sentence* summarises the **gist** of an everyday sentence. It is not a word-for-word translation: it keeps who did what, what happened, and how it turned out, and drops the rest.

These rules are the reference for every step: the task prompt ([`emoji_eval/prompt.py`](emoji_eval/prompt.py)), the validator ([`emoji_eval/validate.py`](emoji_eval/validate.py)) and the decoder legend in the scorer ([`emoji_eval/roundtrip.py`](emoji_eval/roundtrip.py)).

## Grammar

| Rule | Example |
|---|---|
| One idea per group, groups separated by a single space, in story order: **who → action → object → result → emotion** | `🏖️😴 ➡️ 🦞🔥 😭` |
| ➡️ means "then" or "so" | `📺👀 ➡️ 🍕🔥💀`: watching TV, so the pizza burned |
| ❌ goes right **after** the thing it negates | `🔑❌` = keys missing, `👥❌` = nobody |
| ⏪ past / ⏩ future, **only when time matters**, placed before the event it applies to | `🍔🍟🍔🍟 ➡️ ⏩🤢`: I'll be sick later |
| Speech: speaker, then 💬, then what was said (🫵 = you) | `👴💬🫵😊 🙋🥹`: an old man told me I looked happy |
| Small quantities are drawn by repeating an emoji, up to 4 times | `💄⏳⏳⏳`: three hours of make-up |
| When the exact number matters (age, time, price), use keycap numbers | `👦 ⏪ 1️⃣3️⃣`: when I was thirteen |
| An emotion only if the sentence implies one, and the one it implies | `👠❌ 🤦`: forgot my shoes → facepalm, not tears |
| Always keep at least one emoji that clearly shows the main topic | |

## Format (checked automatically)

- 3 to 10 emojis; ➡️ doesn't count toward the limit.
- Single line; no letters, plain digits or punctuation. Keycap numbers (`3️⃣0️⃣`) are emojis and are allowed.

Breaking these makes an output **invalid**. Breaking the style rules above (❌ starting a group, several ⏪/⏩, missing group spaces, an emoji repeated more than 4 times…) only produces a **warning**.

## V2 (step 7 on)

Steps 0 to 6 use the rules above unchanged. From [step 7](07_v2_rules_and_scoring/) on, one format rule changes and the grammar stays the same:

- **1 to 10 emojis**; ➡️ still doesn't count. A lone ➡️ is still invalid.
- **Fewer than 3 emojis is a warning, not an error.** Use as few emojis as the idea needs, usually 3 to 8. One or two are fine only when the sentence is that simple: `🍽️🧼`, washing the dishes.

The checks are `check_validity_v2` and `check_conventions_v2` in [`emoji_eval/validate.py`](emoji_eval/validate.py).
