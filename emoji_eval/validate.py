"""Format checks: hard validity rules (errors) and style conventions (warnings)."""

import itertools

import emoji

VS16 = "️"  # variation selector: turns a symbol into its emoji form
ZWJ = "‍"  # zero-width joiner: glues emojis into one (🧑‍💻 = 🧑 + ZWJ + 💻)
KEYCAP = "⃣"  # combining keycap: turns a digit into an emoji (4 + VS16 + keycap = 4️⃣)
MAX_REPEATS = 4  # a quantity can be shown by repetition (☕☕☕☕), but not endlessly


def content_count(output: str) -> int:
    """Number of emojis, not counting ➡️, which is a connector rather than content."""
    return emoji.emoji_count(output) - output.count("➡")


MIN_EMOJIS_V2 = 1  # v2 (step 7): short answers are allowed, only tracked as a warning
SHORT_V2 = 3  # below this, check_conventions_v2 warns


def check_validity(output: str) -> str:
    """Return "ok" or a short reason why the output breaks the v1 format rules (3 to 10 emojis).

    v1 is used by steps 0 to 6 and must not change; v2 lives in check_validity_v2.
    """
    return _check_validity(output, min_count=3)


def check_validity_v2(output: str) -> str:
    """Return "ok" or a short reason why the output breaks the v2 format rules (1 to 10 emojis)."""
    return _check_validity(output, min_count=MIN_EMOJIS_V2)


def _check_validity(output: str, min_count: int) -> str:
    if "\n" in output:
        return "multiple lines"
    leftover = emoji.replace_emoji(output, replace="").replace(" ", "")
    # Variation selectors and joiners can be left over from emoji sequences.
    leftover = leftover.replace(VS16, "").replace(ZWJ, "")
    # Non-standard joined sequences (e.g. 👨‍❤️‍👩) aren't matched as one emoji and can
    # leave a lone part such as ❤ behind. It still displays as an emoji, so allow it.
    leftover = "".join(c for c in leftover if not emoji.is_emoji(c))
    if leftover:
        return f"non-emoji: {leftover}"
    count = content_count(output)
    if not min_count <= count <= 10:
        return f"{count} emojis"
    return "ok"


def check_conventions(output: str) -> list[str]:
    """Return the style conventions a valid output breaks (empty list = all good).

    These are warnings, not errors: the output is still usable, just less consistent.
    """
    warnings = []
    groups = output.split(" ")
    if any(group.startswith("❌") for group in groups):
        warnings.append("❌ starts a group")  # should follow the thing it negates
    if output.count("⏪") + output.count("⏩") > 1:
        warnings.append("several ⏪/⏩")  # time markers only when time matters
    if output.startswith("➡") or output.rstrip(VS16).endswith("➡"):
        warnings.append("dangling ➡️")  # "then" needs something on both sides
    if "➡️➡️" in output or "➡➡" in output:
        warnings.append("repeated ➡️")
    if "  " in output:
        warnings.append("double space")
    # Keycap numbers (3️⃣0️⃣) are allowed: ages, times or prices can't be drawn. Malformed
    # ones (10️⃣, where the 1 is a plain digit) already fail check_validity.
    found = [e["emoji"] for e in emoji.emoji_list(output)]
    if any(len(list(run)) > MAX_REPEATS for _, run in itertools.groupby(found)):
        warnings.append(f"emoji repeated more than {MAX_REPEATS} times")
    # A long sequence with no space at all means the groups weren't separated.
    if " " not in output and content_count(output) > 4:
        warnings.append("no group spaces")
    return warnings


def check_conventions_v2(output: str) -> list[str]:
    """The v1 conventions, plus a warning for answers shorter than 3 emojis (allowed in v2)."""
    warnings = check_conventions(output)
    if content_count(output) < SHORT_V2:
        warnings.append(f"fewer than {SHORT_V2} emojis")
    return warnings
