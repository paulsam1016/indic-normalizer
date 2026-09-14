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

    fn, source = get_normalizer("ml")   # Malayalam
    fn("ട്വൻറ്റി ഫോർ")                    # -> 'ഇരുപത്തിനാല്'

    normalize("ഇ–മെയിൽ", lang="ml")      # one-shot
    normalize("hello, world")            # no lang -> universal normalizer

`get_normalizer(lang)` returns ``(normalize_fn, source_label)``. A language with a
dedicated normalizer (e.g. Malayalam) uses it; everything else falls back to the
language-agnostic ``universal`` normalizer.

Adding a language: drop ``languages/<lang>.py`` (exposing ``normalize(text)``) and
register its aliases in ``registry._ALIASES`` + a branch in ``registry._load``.
"""

from . import base, universal
from .registry import get_normalizer, script_of

__version__ = "0.1.0"


def normalize(text, lang=None):
    """One-shot normalize ``text`` for ``lang`` (or the universal normalizer if None)."""
    fn, _ = get_normalizer(lang)
    return fn(text)


__all__ = ["get_normalizer", "normalize", "script_of", "universal", "base", "__version__"]
