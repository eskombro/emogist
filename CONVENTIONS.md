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

From [step 9](09_v2_dataset_and_labels/) on, these signs are added for the new sentence types. Each one must read naturally to someone who has never seen these rules:

| Meaning | Emoji | Example |
|---|---|---|
| A question | ❓ as its own group, right after the question part: at the end when the question comes last, after each question when there are several | "Did you feed the cat?" → `🫵🐈🍽️ ❓`; "Can I have a hug?" "I have a boyfriend." → `🫵🤗 ❓ 👩💬 👨❤️` |
| A condition ("if") | No sign: the "if" part, then ➡️ to its consequence, as in v1 | "If it rains, we'll stay home." → `🌧️ ➡️ 🏠` |
| Thinking, knowing, believing | 💭 | "I think he's lying." → `🙋💭 👨🤥` |
| Maybe, doubt, wondering | 🤔 | "Maybe he's lying." → `👨🤥 🤔` |
| Something unknown or asked about, inside a sentence that is not a question | ❔ attached to what is unknown. A question gets ❓ only, never ❔, including "why / when / what" questions; 🤔 is doubt about whether something is true | "Elle m'a demandé si j'étais marié." → `👩💬 🫵💍❔`; "I don't know where he is." → `👨📍❔`; "He had too many questions." → `👨 ❔❔❔` |
| A place, "where" | 📍 attached to the place | "¿Dónde está el baño?" → `📍🚽 ❓` |
| Again, once more | 🔁 attached to the action | "eating again" → `🍔🔁`; "Ella dijo que quería viajar a Japón otra vez." → `👩💬 🤩✈️🇯🇵🔁` |

The author's few-shot examples teach these rules to the teacher, through the v2 labelling prompt ([`09_v2_dataset_and_labels/label_prompt_v2.py`](09_v2_dataset_and_labels/label_prompt_v2.py)). The check only sees the emojis, not the sentence, so it tests positions, not whether a sign is needed:
- ❓ always a group on its own;
- ❔, 📍 and 🔁 attached to something, never a group on their own;
- ❔ and ❓ never in the same label.

💭 and 🤔 have no position rule and aren't checked. Like all conventions, these only produce warnings.

**Dropped after the step 9 quality check:** 🔀 for "if" and 🔄 for "every / all". A decoder that only knew the v1 rules read them as "shuffle" and "switch" (conditions scored 0.31 with 🔀 against 0.70 without), and demo visitors can't be expected to learn them first. "Every / all" is drawn plainly (`👥😂` = everyone laughed).
