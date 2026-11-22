"""Plain word lists keeping unsuitable sentences out of a dataset that will be published.

Lists are per language, because a harmless word in one can be offensive in the other
("bite", "retard"). Deliberately simple: it can miss some sentences and remove a few
harmless ones.
"""

import re

BLOCKED = {
    "en": {
        "sexual": r"sex|sexy|sexual\w*|porn\w*|orgasm\w*|erotic\w*|penis\w*|vagina\w*|genital\w*|"
                  r"masturbat\w*|blowjob\w*|horny|boobs?|cock|pussy|fuck\w*|whores?|sluts?|"
                  r"prostitut\w*|incest\w*|pedophil\w*|paedophil\w*",
        "violence": r"rap(?:e|es|ed|ing|ist|ists)|murder\w*|kill|kills|killed|killer\w*|killing|"
                    r"behead\w*|tortur\w*|suicid\w*|massacre\w*|genocide\w*|stab(?:s|bed|bing)?|"
                    r"strangl\w*|slaughter\w*|terroris\w*",
        "hate": r"nigger\w*|nigga\w*|faggot\w*|fags?|retard|retards|retarded|kike\w*|tranny|"
                r"chinks?|cunt\w*|nazi\w*|hitler|bitch\w*",
    },
    "fr": {
        "sexual": r"sexe\w*|sexuel\w*|porno\w*|érotique\w*|pénis|vagin\w*|branl\w*|baise|baisent|"
                  r"baisé\w*|baiseu\w*|niqu(?:e|es|er|é|ée|és|ées|ent)|bites?|couilles?|putes?|"
                  r"putain\w*|salopes?|prostitu\w*|pédophil\w*|inceste\w*",
        "violence": r"viols?|violée?s?|violeur\w*|assassin\w*|meurtr\w*|tuer|tue|tues|tuent|tué\w*|"
                    r"décapit\w*|égorg\w*|poignard\w*|étrangl\w*|génocide\w*|tortur\w*|"
                    r"massacre\w*|suicid\w*|terroris\w*",
        "hate": r"nazi\w*|hitler|nègres?|négresse\w*|pédés?|bougnoul\w*|youpin\w*|enculé\w*|connasse\w*",
    },
}
PATTERNS = {
    lang: {name: re.compile(rf"(?<![^\W\d_])(?:{terms})(?![^\W\d_])", re.I) for name, terms in lists.items()}
    for lang, lists in BLOCKED.items()
}


def blocked_category(sentence: str, lang: str) -> str | None:
    """Return the first matching category, or None if the sentence is fine."""
    for name, pattern in PATTERNS[lang].items():
        if pattern.search(sentence):
            return name
    return None
