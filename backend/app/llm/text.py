"""Clean LLM prose before it reaches the UI: plain punctuation, no emojis or decorative symbols."""

import re
import unicodedata

_PUNCT = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    "…": "...", " ": " ", " ": " ", " ": " ", "→": "->", "←": "<-",
    "≤": "<=", "≥": ">=", "×": "x", "•": "-",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT)))


def plain(text: str) -> str:
    """Map typographic punctuation to ASCII and drop emojis and other symbol characters. Letters in any script are kept."""
    text = _PUNCT_RE.sub(lambda m: _PUNCT[m.group(0)], text)
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "N", "M")):  # letters, numbers, combining marks
            out.append(ch)
        # symbols (So, Sk, Sm, Sc), emoji modifiers, variation selectors, private use: dropped
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
