"""indic-normalizer — meaning-preserving text normalization for Indic-script ASR scoring.

The point of this library: when you score ASR with WER/CER you normalize BOTH the
reference and the hypothesis so the metric penalizes real mis-transcriptions, not
encoding/convention differences. The normalization must be *stable* (same input →
same output) and *meaning-preserving* (never collapse a real distinction). See
NORMALIZATION.md for the full design rationale and the Whisper "vanishing vowels"
pitfall this avoids.

Usage
-----
    from indic_normalizer import get_normalizer, normalize

    fn, source = get_normalizer("ml")               # Malayalam, eval mode (default)
    fn("ട്വൻറ്റി ഫോർ")                                # -> 'ഇരുപത്തിനാല്'

    fn, _ = get_normalizer("ml", mode="inference")  # reader-facing ITN
    fn("ഇരുപത്തിനാല് രൂപ")                            # -> '24 രൂപ'

    normalize("ഇ–മെയിൽ", lang="ml")                  # one-shot, eval
    normalize("24 രൂപ.", lang="ml", mode="inference")# one-shot, ITN -> '24 രൂപ.'
    normalize("hello, world")                        # no lang -> universal normalizer

`get_normalizer(lang, mode)` returns ``(normalize_fn, source_label)``. A language with a
dedicated normalizer (e.g. Malayalam) uses it; everything else falls back to the
language-agnostic ``universal`` normalizer. ``mode`` is "eval" (default; aliases
"training"/"fine-tune", model-facing verbalization) or "inference" (reader-facing ITN:
numbers -> digits, punctuation + case preserved). See NORMALIZATION.md "Mode selector".

Adding a language: drop ``languages/<lang>.py`` (exposing ``normalize(text, mode)``) and
register its aliases in ``registry._ALIASES`` + a branch in ``registry._load``.
"""

from . import base, universal
from .registry import get_normalizer, script_of

__version__ = "0.1.0"


def normalize(text, lang=None, mode="eval", segmentation=False):
    """One-shot normalize ``text`` for ``lang``/``mode`` (universal normalizer if no lang).

    ``segmentation`` (eval-only, default OFF) enables the W3/W4/W5 boundary rules. The
    C1/C2/C6 CER folds run automatically in eval mode.
    """
    fn, _ = get_normalizer(lang, mode=mode, segmentation=segmentation)
    return fn(text)


__all__ = ["get_normalizer", "normalize", "script_of", "universal", "base", "__version__"]
