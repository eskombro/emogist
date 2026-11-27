"""Build the sentence list behind the demo's random button.

    python 11_demo_v2/make_random_list.py

Reads examples/random_shortlist.csv: sentences picked by hand from the candidate page examples
(written by Claude for the page only, never for training), 30 per language, chosen for a clear and
fun v2 answer, with v1 and v2 both valid. The 18 page examples are included. Only rows whose `keep`
column is "yes" go into demo/web/random.json, grouped by language, so the author can strike rows
by setting `keep` to "no" (or add rows from examples/outputs.csv) and rerun this script.
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo" / "web" / "random.json"


def main() -> None:
    with open(HERE / "examples" / "random_shortlist.csv", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["keep"].strip().lower() == "yes"]
    sentences = {"en": [], "fr": [], "es": []}
    for row in rows:
        sentences[row["lang"]].append(row["sentence"])
    OUT.write_text(json.dumps(sentences, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(HERE.parent)}: " + ", ".join(f"{k} {len(v)}" for k, v in sentences.items()))


if __name__ == "__main__":
    main()
