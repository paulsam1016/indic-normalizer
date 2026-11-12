"""English (Latin script) ASR text normalization for WER/CER scoring: English as the MAIN language.

The same three modes as every normalizer (NORMALIZATION.md "Mode selector"), on the shared steps
(NFC, invisible characters, ``⁇`` / ``<unk>``, digit folding, group commas, whitespace):

* eval — scoring, numbers as words: canonical forms (can't -> can not, um / uh / hmm dropped,
  OK -> okay, & -> and), currency and percent as words (₹500 -> five hundred rupees, 20% ->
  twenty percent), every number spelled out the way the digits module writes English training
  targets (24 -> twenty four, 2.5 -> two point five, 21st -> twenty first, 4:05 -> four oh five,
  1995 -> nineteen ninety five), apostrophe join (John's -> johns), punctuation removed,
  lowercase. ``itn=True`` scores in digits instead (twenty four -> 24, and 24 stays).
* train — training targets: apostrophe join, punctuation removed, lowercase; numbers and the
  spoken forms are left as written (an external verbalizer handles numbers).
* inference — reader-facing: punctuation and case kept; number words -> digits (english_itn),
  in English style by default (``spell_small=True``: "I have five" stays, "five days" -> 5 days).

These are the steps the code-mixed English handler (``indic_normalizer.mixed``) applies to English
inside another main language; here they apply to the whole text. ``mixed`` is accepted for a
uniform API: there are no code-mixed handlers for other scripts inside English yet.
"""

from __future__ import annotations

from .. import base
from .. import mixed as _mixed
from .english_itn import itn as _itn

_resolve_mode = base.resolve_mode
_EN = _mixed.HANDLERS["en"]


def normalize(text: str, mode: str = "eval", segmentation: bool = False, itn=None,
              spell_small=None, mixed=True) -> str:
    """Normalize English ASR text; see the module docstring. ``itn`` defaults to the mode's
    default (on in inference, off in eval; no effect in train); ``spell_small`` defaults to True
    (English style) for English as the main language. ``segmentation`` / ``mixed``: accepted for
    a uniform API, no effect."""
    mode = _resolve_mode(mode)
    itn = (mode == "inference") if itn is None else bool(itn)
    _ = segmentation
    _mixed.resolve(mixed, spell_small)                 # validate, as every normalizer does
    opts = _EN.Options(spell_small=True if spell_small is None else bool(spell_small))

    # --- shared ---
    text = base.to_nfc(text)
    text = base.strip_unk(text)
    text = base.strip_zero_width(text)
    text = base.fold_digits(text)
    # E7 (group commas) is not done here: eval reads the grouping first (2,000,000 -> two
    # million), train keeps it for the external verbalizer, inference leaves it as written.

    if mode == "inference":
        if itn:
            text = _itn(text, spell_small=opts.spell_small)
        return base.collapse_space(text)

    if mode == "eval":
        text = _EN.canonical(text, opts, every=True)   # E17: can't -> can not, um, OK, &
        text = _EN.currency(text, opts, every=True)    # E15: ₹500 / $24 / 20% -> 500 rupees ...
        text = base.CURRENCY_SIGN.sub(" ", text)
        if itn:
            text = base.drop_group_commas(_itn(text, spell_small=opts.spell_small))
            text = _EN.currency(text, opts, every=True)   # ITN writes 20%: 20 percent again
        else:
            text = _EN.verbalize(text, opts, every=True)  # E8: numbers -> English words (grouping read)
        text = base.drop_group_commas(text)            # E7
    else:
        text = base.hold_group_commas(text)            # train: 70,000 / 100,000 kept for the verbalizer
    text = base.join_apostrophes(text)                 # E14: John's -> johns, it's -> its (train)
    text = base.strip_punct(text)                      # E5
    text = base.restore_group_commas(text)
    text = text.lower()                                # E10
    return base.collapse_space(text)                   # E11
