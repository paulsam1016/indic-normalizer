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

# Three modes (see NORMALIZATION.md "Mode selector"): eval, train, inference. The universal
# normalizer has no number rules, so eval and train behave identically here; mode only
# governs punctuation removal (E5) and lowercasing (E10), which inference keeps.
_EVAL_MODES = {"eval", "evaluation"}
_TRAIN_MODES = {"train", "training", "fine-tune", "finetune"}
_INFER_MODES = {"inference", "infer", "itn"}


def _resolve_mode(mode) -> str:
    m = str(mode or "eval").strip().lower()
    if m in _INFER_MODES:
        return "inference"
    if m in _EVAL_MODES:
        return "eval"
    if m in _TRAIN_MODES:
        return "train"
    raise ValueError(
        f"unknown mode {mode!r}; use 'eval', 'train' (aliases: training, fine-tune) or 'inference'"
    )


def normalize(text: str, mode: str = "eval", segmentation: bool = False) -> str:
    """Script-, (mode-dependent) punctuation- and case-insensitive normalization.

    mode="eval" (default) and mode="train" strip punctuation and lowercase;
    mode="inference" preserves both for a readable transcript. Encoding/digit/whitespace
    canonicalization is identical in both. ``segmentation`` is accepted for a uniform API but
    has no effect here (the universal normalizer has no per-language rules).
    """
    mode = _resolve_mode(mode)
    _ = segmentation  # accepted for API parity; universal has no W-family rules
    text = base.to_nfc(text)
    text = base.strip_unk(text)
    text = base.strip_zero_width(text)
    text = base.fold_digits(text)
    if mode in ("eval", "train"):
        text = base.strip_punct(text)     # E5
        text = text.lower()               # E10
    return base.collapse_space(text)
