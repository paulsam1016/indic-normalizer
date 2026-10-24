"""indic_normalizer.digits — digit normalization (numbers -> spoken words), separate from core.

Core (``indic_normalizer.normalize`` / ``get_normalizer``) is untouched by this module. Use it
to turn numbers in TRAINING TARGETS into the words that were actually spoken, so the targets
contain no digits or number punctuation (4:00-5:00, 2.5, 70%, 23-ന്).

    from indic_normalizer import digits

    digits.verbalize("ഫീസ് 1,75,000 ആണ്, 4:00-5:00 വരെ")
    # -> 'ഫീസ് ഒരു ലക്ഷത്തി എഴുപത്തയ്യായിരം ആണ്, നാല് മണി അഞ്ച് മണി വരെ'

    digits.find(text)          # [NumberSpan(core='1,75,000', cls='grouped', token_index=1, ...), ...]
    digits.candidates("25")    # [ഇരുപത്തഞ്ച് (ml, contracted), ഇരുപത്തിയഞ്ച് (ml, glide), ട്വന്റി ഫൈവ് (en)]

Choosing the reading
--------------------
A digit does not say HOW it was spoken: in the eval calls ~40% of numbers are said in English
(ഫോർ, തൗസന്റ്, ലാക്ക്), halves as രണ്ടര, times as നാലര / ആറേമുക്കാൽ. ``candidates()`` lists
the plausible readings (native contracted, native glide, English, plus class-specific forms);
``verbalize(text, chooser=...)`` calls your chooser for every number so the pick can come
from the audio — e.g. score each candidate with an ASR model (RNN-T / CTC log-likelihood of
the candidate text on that word's audio window, located via ``span.token_index`` in a
word-timestamped transcript) and return the best. Without a chooser the first candidate
(native, ``style``) is used.

The Malayalam forms are in ``indic_normalizer.digits.ml``; other languages can be added as
sibling modules.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

from . import hi, ml
from .detect import CLASSES, classify, find
from .types import Candidate, NumberSpan

Chooser = Callable[[NumberSpan, List[Candidate]], Union[Candidate, str, int]]

_LANGS = {"ml": ml, "mal": ml, "malayalam": ml, "hi": hi, "hin": hi, "hindi": hi}


def _lang(lang: str):
    try:
        return _LANGS[str(lang).lower()]
    except KeyError:
        raise ValueError(f"digits: unsupported language {lang!r}; available: ml, hi") from None


def candidates(number: Union[str, NumberSpan], lang: str = "ml",
               style: str = "contracted", latin: bool = False, next_word: str = "") -> List[Candidate]:
    """Spoken readings of one number (a token core like '4:00-5:00' or a NumberSpan).
    First = default. ``style`` is "contracted" (ഇരുപത്തഞ്ച്) or "glide" (ഇരുപത്തിയഞ്ച്).
    ``latin=True`` (the sentence is English): Latin-script English readings (five, twelve thirty)."""
    if isinstance(number, NumberSpan):
        return _lang(lang).candidates(number.core, number.cls, style, latin, next_word)
    return _lang(lang).candidates(number, None, style, latin, next_word)


def next_word(text: str, span: NumberSpan) -> str:
    """The word after a number (sentence punctuation stripped), for context-dependent readings."""
    rest = text[span.end:].split()
    return rest[0].strip(".,?!;:") if rest else ""


def _pick(choice, cands: List[Candidate]) -> str:
    if isinstance(choice, Candidate):
        return choice.text
    if isinstance(choice, int):
        return cands[choice].text
    if choice is None:
        return cands[0].text
    return str(choice)


def verbalize(text: str, lang: str = "ml", style: str = "contracted",
              chooser: Optional[Chooser] = None, latin: Optional[bool] = None) -> str:
    """Replace every number in ``text`` with words. Sentence punctuation around a number is
    kept (only the number itself is rewritten); spacing elsewhere is unchanged.

    ``chooser(span, candidates)`` may return a Candidate, an index into ``candidates``, a
    string, or None (= default first candidate).
    ``latin``: read numbers as Latin-script English; None = decide from ``text`` itself (an
    English sentence: at least half of its letters Latin)."""
    if not text:
        return text or ""
    from .detect import join_currency
    text = join_currency(text)                       # ₹ 500 / Rs. 500 -> read as one amount
    if latin is None:
        from .latin import is_english
        latin = is_english(text)
    out, pos = [], 0
    for span in find(text):
        cands = candidates(span, lang, style, latin, next_word(text, span))
        choice = chooser(span, cands) if chooser else None
        out.append(text[pos:span.start])
        out.append(_pick(choice, cands))
        pos = span.end
    out.append(text[pos:])
    return "".join(out)


__all__ = ["verbalize", "candidates", "find", "classify", "Candidate", "NumberSpan",
           "CLASSES", "Chooser", "ml"]
