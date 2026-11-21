"""Column layout of a bake-off results CSV, shared by the runner, scorer and reports."""

SENTENCE_COLUMNS = ("id", "lang", "sentence", "tags")


def model_columns(row: dict) -> list[str]:
    """Model names in a results row: every column except the sentence columns and the
    per-model "<name> check" / "<name> warnings" columns."""
    return [c for c in row if c not in SENTENCE_COLUMNS and not c.endswith((" check", " warnings"))]
