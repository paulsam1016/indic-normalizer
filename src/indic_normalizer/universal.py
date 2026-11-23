"""Language-agnostic ASR text normalization for WER/CER — all Indic scripts + English.

Only normalizations valid across *every* language are applied, so references and
hypotheses compare fairly without favouring any one script or convention:

  * Unicode NFC — one canonical codepoint sequence per grapheme;
  * ``<unk>`` / ``⁇`` marker removal;
  * invisible formatting char removal (ZWNJ, ZWJ, BOM, word joiner, soft hyphen; a zero-width
    space becomes a space);
  * punctuation & symbol removal — ASCII **and** Indic danda ``।`` / ``॥``, quotes,
    dashes, etc. (an explicit set, NOT a Unicode Mark/Symbol category — that would
    delete Indic vowel signs and the virama; see NORMALIZATION.md);
  * any Unicode decimal digit → ASCII 0-9, and (eval/train) digit-grouping commas dropped
    (``70,000`` / ``1,75,000`` → one number);
  * lowercase (English casing; a no-op for the case-less Indic scripts);
  * whitespace collapse.

Deliberately NOT done — these are language-specific: chillu / ZWJ canonicalization,
digit *verbalization*, and English-number-word canonicalization. They live in the
per-language modules (e.g. ``languages/malayalam.py``).
"""

from . import base

# Three modes (see NORMALIZATION.md "Mode selector"): eval, train, inference. The universal
# normalizer has no number rules, so eval and train behave identically here; mode only
# governs group-comma and punctuation removal (E7, E5) and lowercasing (E10), which inference
# keeps.
_resolve_mode = base.resolve_mode


def normalize(text: str, mode: str = "eval", segmentation: bool = False, itn=None,
              spell_small=None, mixed=True, strip_number_signs: bool = False) -> str:
    """Script-, (mode-dependent) punctuation- and case-insensitive normalization.

    mode="eval" (default) and mode="train" strip punctuation and lowercase;
    mode="inference" preserves both for a readable transcript. Encoding/digit/whitespace
    canonicalization is identical in both. mode="train" keeps what belongs to a number for an
    external verbalizer (2.5, 4:30, 70,000, ₹500, $24, 20%); ``strip_number_signs=True`` removes
    the signs before / after a number. ``segmentation``, ``itn``, ``spell_small`` and ``mixed`` are
    accepted for a uniform API but have no effect here (no per-language or number rules).
    """
    mode = _resolve_mode(mode)
    _ = segmentation, itn, spell_small, mixed  # API parity; universal has no W-family or number rules
    text = base.to_nfc(text)
    text = base.strip_unk(text)
    text = base.strip_zero_width(text)
    text = base.fold_digits(text)
    if mode in ("eval", "train"):
        if mode == "eval":
            text = base.drop_group_commas(text)   # E7: 1,000 -> 1000 (BEFORE E5 splits it)
        else:
            text = base.hold_number_marks(text, strip_number_signs)   # T1: 2.5, $24, 20% kept
            text = base.hold_group_commas(text)   # train: 70,000 kept for the external verbalizer
        text = base.join_apostrophes(text)   # E14: it's -> its (BEFORE E5 splits it)
        text = base.strip_punct(text)     # E5
        text = base.restore_number_marks(base.restore_group_commas(text))
        text = text.lower()               # E10
    return base.collapse_space(text)
