"""v2 sentence rules: Spanish word lists, questions allowed, and plain-code tags.

Step 3's rules (03_dataset/fetch_real_sentences.py, blocklist.py) stay unchanged; this module
reuses them for English and French and adds what v2 needs. Plain rules only, no model.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_dataset"))

import blocklist  # noqa: E402  (step 3)
import fetch_real_sentences as v1  # noqa: E402  (step 3)

LANGS = {"en": "eng", "fr": "fra", "es": "spa"}

# Spanish drops subject pronouns ("Fui al mercado"), so this list mostly catches object and
# possessive forms and family words. Second person is left out, as in English and French.
PERSON_WORDS = {
    **v1.PERSON_WORDS,
    "es": {"yo", "me", "mi", "mis", "mí", "conmigo", "nosotros", "nosotras", "nos", "nuestro",
           "nuestra", "nuestros", "nuestras", "él", "ella", "ellos", "ellas", "le", "les",
           "mamá", "papá", "madre", "padre", "hermano", "hermana", "hijo", "hija", "hijos",
           "esposa", "esposo", "mujer", "marido", "novio", "novia", "amigo", "amiga", "amigos",
           "amigas", "jefe", "jefa", "vecino", "vecina", "vecinos", "niños", "abuelo", "abuela"},
}
# Impersonal openings that describe no person ("Hay que...", "Llueve", "Se dice que...").
IMPERSONAL = {
    "fr": v1.IMPERSONAL_FR,
    "es": re.compile(r"^(no )?(hay|había|habrá|hubo|llueve|llovía|nieva|nevaba|parece que|se dice|se puede|"
                     r"hace (frío|calor|sol|viento|buen tiempo|mal tiempo))\b", re.I),
}

BLOCKED_ES = {
    "sexual": r"sexo|sexual\w*|porno\w*|erótic\w*|pene|penes|vagina\w*|masturb\w*|foll\w*|polla|coño|"
              r"tetas?|vergas?|putas?|putos?|prostitu\w*|pedófil\w*|incesto\w*",
    "violence": r"viol(?:ar|ó|aron|ación|aciones|ador|adores|ada|adas)|asesin\w*|matar|mata|matan|mató|"
                r"mataron|matado|decapit\w*|degoll\w*|apuñal\w*|estrangul\w*|tortur\w*|masacre\w*|"
                r"genocidio\w*|suicid\w*|terroris\w*",
    "hate": r"nazi\w*|hitler|maric[oó]n\w*|negrata\w*|sudaca\w*|subnormal\w*|retrasad[oa]s?|zorras?",
}
BLOCKED_ES_PATTERNS = {name: re.compile(rf"(?<![^\W\d_])(?:{terms})(?![^\W\d_])", re.I)
                       for name, terms in BLOCKED_ES.items()}

OPENING_MARKS = "¿¡«\"“'"


def blocked_category(sentence: str, lang: str) -> str | None:
    if lang != "es":
        return blocklist.blocked_category(sentence, lang)
    for name, pattern in BLOCKED_ES_PATTERNS.items():
        if pattern.search(sentence):
            return name
    return None


def looks_clean(sentence: str) -> bool:
    """Step 3's check, except that a sentence may open with ¿, ¡ or a quotation mark."""
    return v1.looks_clean(sentence.lstrip(OPENING_MARKS))


def is_everyday(sentence: str, lang: str) -> bool:
    """Step 3's rule with questions allowed: v2 measures them on purpose."""
    if re.search(r"\b\d{4}\b", sentence):
        return False
    tokens = re.findall(r"[^\W\d_]+", sentence)
    if sum(t[0].isupper() and t != "I" for t in tokens[1:]) > v1.MAX_PROPER_NOUNS:
        return False
    if lang in IMPERSONAL and IMPERSONAL[lang].match(sentence.lstrip(OPENING_MARKS)):
        return False
    return bool({t.lower() for t in tokens} & PERSON_WORDS[lang])


# ---------- tags, extended from step 5's error breakdown ----------

NUMBER = re.compile(
    r"\d|\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand|million|deux|trois|quatre|cinq|sept|huit|dix|onze|douze|treize|"
    r"quatorze|quinze|seize|vingt|trente|quarante|cinquante|soixante|cent|mille|dos|tres|cuatro|cinco|seis|"
    r"siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciséis|diecisiete|dieciocho|diecinueve|veinte|"
    r"treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien|ciento|mil|millón|millones)\b", re.I)
NEGATION = re.compile(r"n't\b|\b(not|never|no|nobody|nothing|cannot|ne|pas|jamais|rien|personne|aucun|aucune|"
                      r"nunca|nada|nadie|ningún|ninguno|ninguna|jamás|tampoco)\b|\bn'", re.I)
SPEECH = re.compile(r"\b(said|say|says|told|tell|tells|asked|ask|dit|dis|disent|demandé|demande|raconté|"
                    r"dijo|dije|dicho|dice|dicen|preguntó|pregunté|contó|conté|respondió|contestó)\b", re.I)
CONDITIONAL = re.compile(r"^(if|si)\b|[,;] ?(if|si)\b", re.I)


def word_count(sentence: str) -> int:
    return len(re.findall(r"[\w'’-]+", sentence))


def length_group(sentence: str) -> str:
    n = word_count(sentence)
    return "short" if n <= 5 else "medium" if n <= 8 else "long"


def tags(sentence: str) -> list[str]:
    found = [length_group(sentence)]
    if "?" in sentence:
        found.append("question")
    for name, pattern in (("number", NUMBER), ("negation", NEGATION), ("speech", SPEECH),
                          ("conditional", CONDITIONAL)):
        if pattern.search(sentence.lstrip(OPENING_MARKS)):
            found.append(name)
    return found
