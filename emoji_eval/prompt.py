"""The task prompt: system rules + hand-made few-shot examples (see CONVENTIONS.md).

Changing anything here changes every model's prompt, so earlier results stop being
comparable: re-run the bake-off after an edit.
"""

SYSTEM_PROMPT = """You convert a sentence into an "emoji sentence": a short sequence of emojis that lets someone guess the gist of the sentence. It is not a word-for-word translation: keep the main actors, the action, the result and the emotion, and drop the rest.

Rules:
1. Use 3 to 10 emojis. Never use letters from any alphabet, digits, or punctuation such as "...".
2. One idea per group of emojis, in story order: who → action → object → result → emotion.
3. Use ➡️ to mean "then" or "so".
4. Put ❌ right after the thing it negates (🧳❌ = the suitcase is missing).
5. Use ⏪ for the past and ⏩ for the future, only when time matters.
6. For speech or messages, write the speaker, then 💬, then what was said (🫵 = you).
7. Always keep at least one emoji that clearly shows the main topic.
8. Prefer common, simple emojis.
9. Separate groups with a single space.
10. Answer with the emoji sentence only, on a single line, nothing else."""

# Few-shot examples, sent as previous chat turns. Written by hand; each one was checked
# with the round-trip scorer. Topics deliberately avoid the test set.
EXAMPLES = [
    ("I forgot my keys and got locked out of my apartment.", "🔑❌ 🚪🔒 😩"),
    ("My cat knocked my coffee off the table this morning.", "🌅 🐈 ☕💥 😱"),
    ("We missed the last train and had to walk home in the rain.", "🚆💨 🏃❌ ➡️ 🚶🌧️🏠"),
    ("I burned the pizza while watching a movie.", "📺👀 ➡️ 🍕🔥💀"),
    ("I fell asleep on the beach and got a terrible sunburn.", "🏖️😴 ➡️ 🦞🔥 😭"),
    ("I drank four coffees and coded all night to fix that bug.", "🧑‍💻 ☕☕☕☕ 🌙➡️☀️ 🐛✅ 😵‍💫"),
    ("J'ai renversé du vin rouge sur ma chemise blanche à un mariage.", "💒 🍷 ➡️ 👕⚪🔴 😱"),
    ("J'ai lancé une nouvelle app mais personne ne l'utilise.", "🧑‍💻 📲🆕🚀 ➡️ 👥❌ 😢"),
    ("Hier il a plu, donc aujourd'hui je prends mon parapluie.", "⏪🌧️ ➡️ 🙋☂️"),
    ("J'ai mangé trop de fast-food, je vais être malade après.", "🍔🍟🍔🍟 ➡️ ⏩🤢"),
    ("J'ai mis 3 heures à me maquiller, et j'ai oublié mes chaussures.", "💄⏳⏳⏳ ➡️ 👠❌ 🤦"),
    ("Un vieil homme m'a dit que j'avais l'air heureux, ça m'a touché.", "👴💬🫵😊 🙋🥹"),
]


def build_messages(sentence: str) -> list[dict]:
    """System prompt, then the examples as previous user/assistant turns, then the sentence."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, emojis in EXAMPLES:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": emojis})
    messages.append({"role": "user", "content": sentence})
    return messages
