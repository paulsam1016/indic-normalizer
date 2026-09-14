"""Language registry + lightweight script detection.

`get_normalizer(lang)` maps a language name/code to its ``(normalize_fn, source)``.
Detection note: this library does NOT auto-detect *language* — script != language
(Hindi/Marathi/Sanskrit all share Devanagari), and reliable language-ID needs a model.
It DOES offer `script_of(text)`, a trivial Unicode-block check, so a caller that only
needs script routing can do so without guessing. For ASR scoring you already know the
language (it's the benchmark language), so pass it explicitly.
"""

from . import universal

# Language name / code -> registry key. Extend as normalizers are added.
_ALIASES = {
    "ml": "ml", "mal": "ml", "malayalam": "ml",
}


def _lang_code(lang):
    """Registry key for a language name/code, or None (-> universal)."""
    if not lang:
        return None
    return _ALIASES.get(str(lang).strip().lower())


def _load(code):
    """Import and return ``(normalize_fn, source)`` for a registered code, or None.

    Lazy so a heavy per-language dependency (e.g. indic-nlp-library) only loads when
    that language is actually used.
    """
    if code == "ml":
        from .languages import malayalam
        return malayalam.normalize, "indic_normalizer.malayalam"
    return None


def get_normalizer(lang=None):
    """``(normalize_fn, source)`` for ``lang``; the universal normalizer as fallback."""
    code = _lang_code(lang)
    if code is not None:
        loaded = _load(code)
        if loaded is not None:
            return loaded
    return universal.normalize, "indic_normalizer.universal"


# --- optional: Unicode-block script detection (script routing, NOT language ID) ------
# (first codepoint of each Brahmic block -> a representative language code)
_SCRIPT_BLOCKS = [
    (0x0900, 0x097F, "hi"),   # Devanagari
    (0x0980, 0x09FF, "bn"),   # Bengali/Assamese
    (0x0A00, 0x0A7F, "pa"),   # Gurmukhi
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0B00, 0x0B7F, "or"),   # Oriya
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
]


def script_of(text):
    """Best-effort script guess by dominant Brahmic Unicode block, or None.

    Returns a representative language code for the most frequent Indic script in
    ``text`` (e.g. 'ml' for Malayalam), or None if no Indic script is present. This is
    SCRIPT detection, not language ID — use it only when script is enough to route.
    """
    counts = {}
    for ch in text or "":
        cp = ord(ch)
        for lo, hi, code in _SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.get)
