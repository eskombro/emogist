"""Step 8c: combine real and AI sentences into the v2 dev and test sets, with the author's review.

    python 08_v2_spanish_and_eval_sets/build_sets.py

Per language and set: 50 real (from fetch_real_v2.py) and 50 AI (from generate_ai_v2.py).
AI quotas per language and set: 7 questions, 7 conditionals, 6 of each other pattern.
AI sentences are assigned in generation order (the order of ai_eval.csv), alternating dev and
test; the rest are spares.

Review: review_sheet.csv lists every sentence in the sets with a `verdict` column, filled by
the author with `ok` or `remove`. Rerunning replaces each removed sentence with the next spare
of the same language, source and type (in the fixed order: nobody chooses the replacement),
and adds the replacement to the sheet with an empty verdict. Verdicts are kept between runs.

Free to run: no model is called.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import rules_v2
from fetch_real_v2 import MAX_AUTHOR_PER_SET, QUOTAS, bucket

HERE = Path(__file__).resolve().parent
SETS = HERE / "sets"
SHEET = SETS / "review_sheet.csv"
LANGS = ("en", "fr", "es")
AI_QUOTAS = {"question": 7, "conditional": 7, "speech": 6, "past": 6, "future": 6, "mishap": 6, "moment": 6,
             "numbers": 6}
FIELDS = ["id", "set", "lang", "source", "type", "sentence", "author", "generator", "tags"]
SHEET_FIELDS = ["set", "id", "lang", "source", "type", "sentence", "verdict", "note"]


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sheet_delimiter() -> str:
    """The review sheet may come back from a spreadsheet tab-separated; keep whichever it uses."""
    if SHEET.exists():
        with open(SHEET, encoding="utf-8") as f:
            return "\t" if "\t" in f.readline() else ","
    return ","


def verdicts() -> dict[str, dict]:
    """Verdicts by id. Sentence text always comes from the set files, never from the sheet."""
    if not SHEET.exists():
        return {}
    with open(SHEET, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter=sheet_delimiter())}


def real_sets(removed: set[str]) -> dict[str, list[dict]]:
    """The drawn real sentences, with removed ones replaced by the next pool sentence of the same type."""
    # In the shuffled order of fetch_real_v2.py, split alternately so dev and test never share spares.
    pool_all = read(SETS / "real_pool.csv")
    out = {}
    used = set()
    for k, name in enumerate(("dev", "test")):
        pool = pool_all[k::2]
        rows = []
        for r in read(SETS / f"real_{name}.csv"):
            if r["id"] not in removed:
                rows.append(r)
                continue
            authors = Counter(x["author"] for x in rows)
            wanted = bucket(r["sentence"])
            for p in pool:
                if (p["id"] not in used and p["id"] not in removed and p["lang"] == r["lang"]
                        and bucket(p["sentence"]) == wanted and authors[p["author"]] < MAX_AUTHOR_PER_SET):
                    used.add(p["id"])
                    rows.append(p)
                    break
            else:
                raise SystemExit(f"No spare real sentence for {r['id']}")
        for r in rows:
            r.update(source="real", type=bucket(r["sentence"]), generator="", set=name)
        out[name] = rows
    return out


def ai_sets(removed: set[str]) -> tuple[dict[str, list[dict]], Counter]:
    """The first 2 x quota sentences of each language and pattern alternate dev/test, in plan
    order. A removed one is replaced by the next spare, in the same set, so nothing moves."""
    # File order = generation order. Not plan order: a later run can retry an early plan cell
    # that was rejected before, and that sentence must become a spare, not shift the sets.
    groups = defaultdict(list)
    for r in read(SETS / "ai_eval.csv"):
        groups[(r["lang"], r["pattern"])].append(r)
    out, spares = {"dev": [], "test": []}, Counter()
    for (lang, pattern), items in groups.items():
        need = AI_QUOTAS[pattern]
        if len(items) < 2 * need:
            raise SystemExit(f"Only {len(items)} AI sentences for {lang}/{pattern}: need {2 * need}")
        # Spares alternate dev/test too, so a set only ever draws from its own spares.
        free = {name: [r for j, r in enumerate(items[2 * need:]) if r["id"] not in removed and j % 2 == k]
                for k, name in enumerate(("dev", "test"))}
        for i, r in enumerate(items[:2 * need]):
            name = "dev" if i % 2 == 0 else "test"
            if r["id"] in removed:
                if not free[name]:
                    raise SystemExit(f"No spare AI sentence left for {lang}/{pattern}/{name}")
                r = free[name].pop(0)
            r.update(source="ai", type=pattern, author="", set=name)
            out[name].append(r)
        spares[(lang, pattern)] = len(free["dev"]) + len(free["test"])
    return out, spares


def main() -> None:
    previous = verdicts()
    removed = {i for i, r in previous.items() if r["verdict"].strip().lower() == "remove"}
    real = real_sets(removed)
    ai, spares = ai_sets(removed)

    sheet = []
    for name in ("dev", "test"):
        rows = sorted(real[name] + ai[name], key=lambda r: (LANGS.index(r["lang"]), r["source"], r["type"], r["id"]))
        for r in rows:
            r["tags"] = " ".join(rules_v2.tags(r["sentence"]))
        with open(SETS / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        counts = Counter((r["lang"], r["source"]) for r in rows)
        print(f"{name}.csv: {len(rows)} sentences " + ", ".join(f"{l}/{s} {n}" for (l, s), n in sorted(counts.items())))
        for r in rows:
            old = previous.get(r["id"], {})
            sheet.append({"set": name, "id": r["id"], "lang": r["lang"], "source": r["source"], "type": r["type"],
                          "sentence": r["sentence"], "verdict": old.get("verdict", ""), "note": old.get("note", "")})
    # Removed rows stay on the sheet, at the end, so the decisions are kept.
    sheet += [r for i, r in previous.items() if i in removed]
    delimiter = sheet_delimiter()
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sheet)

    todo = sum(not r["verdict"].strip() for r in sheet)
    gens = Counter(r["generator"] for name in ai for r in ai[name])
    print(f"review_sheet.csv: {todo} sentences without a verdict, {len(removed)} removed so far")
    print(f"AI generators in the sets: {dict(gens)}")
    print(f"AI spares left: {sum(spares.values())} (lowest: {min(spares.values())})")


if __name__ == "__main__":
    main()
