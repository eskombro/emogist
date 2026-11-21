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
| Numbers and times are drawn, never written | `💄⏳⏳⏳`: three hours of make-up |
| An emotion only if the sentence implies one, and the one it implies | `👠❌ 🤦`: forgot my shoes → facepalm, not tears |
| Always keep at least one emoji that clearly shows the main topic | |

## Format (checked automatically)

- 3 to 10 emojis; ➡️ doesn't count toward the limit.
- Single line; no letters, digits or punctuation.

Breaking these makes an output **invalid**. Breaking the style rules above (❌ starting a group, several ⏪/⏩, missing group spaces…) only produces a **warning**.
