"""Language-agnostic ASR text normalization for WER/CER — all Indic scripts + English.

Only normalizations valid across *every* language are applied, so references and
hypotheses compare fairly without favouring any one script or convention:

  * Unicode NFC — one canonical codepoint sequence per grapheme;
  * ``<unk>`` / ``⁇`` marker removal;
  * zero-width formatting char removal (ZWNJ, ZWJ, BOM);
  * punctuation & symbol removal — ASCII **and** Indic danda ``।`` / ``॥``, quotes,
    dashes, etc. (an explicit set, NOT a Unicode Mark/Symbol category — that would
    delete Indic vowel signs and the virama; see NORMALIZATION.md);
  * any Unicode decimal digit → ASCII 0-9;
  * lowercase (English casing; a no-op for the case-less Indic scripts);
  * whitespace collapse.

Deliberately NOT done — these are language-specific: chillu / ZWJ canonicalization,
digit *verbalization*, and English-number-word canonicalization. They live in the
per-language modules (e.g. ``languages/malayalam.py``).
"""

from . import base


def normalize(text: str) -> str:
    """Script-, punctuation- and case-insensitive normalization (see module docstring)."""
    text = base.to_nfc(text)
    text = base.strip_unk(text)
    text = base.strip_zero_width(text)
    text = base.fold_digits(text)
    text = base.strip_punct(text)
    text = text.lower()
    return base.collapse_space(text)
