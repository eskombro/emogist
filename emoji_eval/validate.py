"""Format checks: hard validity rules (errors) and style conventions (warnings)."""

import emoji

VS16 = "️"  # variation selector: turns a symbol into its emoji form
ZWJ = "‍"  # zero-width joiner: glues emojis into one (🧑‍💻 = 🧑 + ZWJ + 💻)


def content_count(output: str) -> int:
    """Number of emojis, not counting ➡️, which is a connector rather than content."""
    return emoji.emoji_count(output) - output.count("➡")


def check_validity(output: str) -> str:
    """Return "ok" or a short reason why the output breaks the format rules."""
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
    if not 3 <= count <= 10:
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
    # A long sequence with no space at all means the groups weren't separated.
    if " " not in output and content_count(output) > 4:
        warnings.append("no group spaces")
    return warnings
