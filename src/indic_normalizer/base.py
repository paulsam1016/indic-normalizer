"""Shared normalization primitives for Indic-script ASR scoring.

These are the language-agnostic building blocks every normalizer reuses, so the
per-language modules only add their script-specific rules on top. All steps are
meaning-preserving and safe to apply to BOTH reference and hypothesis (see
NORMALIZATION.md for the design rationale and the Whisper "vanishing vowels" pitfall
this deliberately avoids — we remove an explicit punctuation set, never a Unicode
category, so vowel signs / virama are preserved).

Pure stdlib; no third-party dependency.
"""

import re
import unicodedata

# ``⁇`` / ``<unk>`` OOV markers (Soniox and some base models emit these).
UNK = re.compile(r"⁇|<unk>")

# Zero-width formatting chars: ZWNJ, ZWJ, BOM. Drop AFTER any chillu/ZWJ fix that
# relies on the ZWJ still being present.
ZERO_WIDTH = re.compile("[‌‍﻿]")

# Punctuation + symbols to strip, ASCII AND Indic (danda ।॥, curly quotes, dashes,
# ellipsis). An EXPLICIT set — never a Unicode Mark/Symbol/Punctuation *category*
# filter, which would delete Indic vowel signs and the virama.
PUNCT = re.compile(r"[.,!?;:\"'`~^*_=+/\\|<>(){}\[\]@#$%&…।॥“”‘’«»\-–—]")

SPACE = re.compile(r"\s+")


def to_nfc(text: str) -> str:
    """Unicode NFC — one canonical codepoint sequence per grapheme."""
    return unicodedata.normalize("NFC", text or "")


def strip_unk(text: str) -> str:
    """Replace ``⁇`` / ``<unk>`` markers with a space (removed from both sides)."""
    return UNK.sub(" ", text)


def strip_zero_width(text: str) -> str:
    """Remove ZWNJ / ZWJ / BOM."""
    return ZERO_WIDTH.sub("", text)


def fold_digits(text: str) -> str:
    """Map every Unicode *decimal* digit to ASCII 0-9 (Devanagari, Malayalam, …).

    Uses the Unicode decimal property, so it covers all scripts uniformly and leaves
    non-digit characters untouched.
    """
    out = []
    for ch in text:
        d = unicodedata.decimal(ch, None)
        out.append(str(d) if d is not None else ch)
    return "".join(out)


def strip_punct(text: str) -> str:
    """Replace ASCII + Indic punctuation/symbols with a space."""
    return PUNCT.sub(" ", text)


def collapse_space(text: str) -> str:
    """Collapse runs of whitespace to a single space and trim."""
    return SPACE.sub(" ", text).strip()
