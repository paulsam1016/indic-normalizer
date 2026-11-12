# Port of indic-nlp-library 0.92 (MIT, (c) 2013-present Anoop Kunchukuttan)
# indicnlp/normalize/indic_normalize.py: NormalizerI._normalize_punctuations,
# BaseNormalizer.normalize and MalayalamNormalizer.normalize, restricted to the
# configuration indic-normalizer uses (IndicNormalizerFactory().get_normalizer("ml")
# with defaults: no nukta removal, nasals "do_nothing", no chandra / vowel-ending
# normalisation, no chillu canonicalisation, no geminated-T correction). Behaviour is
# identical for that configuration; the rest of indic-nlp (and its pandas / sphinx /
# morfessor dependency chain) is not needed.
#
# MIT License
#
# Copyright (c) 2013-present Anoop Kunchukuttan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""indic-nlp-library's Malayalam script normalizer, default configuration."""

from __future__ import annotations

import re

_BOM, _BOM2, _WJ, _SHY = "\ufeff", "\ufffe", "\u2060", "\u00ad"
_ZWSP, _NBSP, _ZWNJ, _ZWJ = "\u200b", "\u00a0", "\u200c", "\u200d"

_PUNCT = (
    (_BOM, ""), ("„", '"'), ("“", '"'), ("”", '"'), ("–", "-"), ("—", " - "), ("´", "'"),
    ("‘", "'"), ("‚", "'"), ("’", "'"), ("''", '"'), ("´´", '"'), ("…", "..."),
)
# Old (Unicode <= 5.0) chillu encoding: consonant + virama + ZWJ -> atomic chillu.
_OLD_CHILLU = (
    ("\u0d23\u0d4d\u200d", "\u0d7a"), ("\u0d28\u0d4d\u200d", "\u0d7b"), ("\u0d30\u0d4d\u200d", "\u0d7c"),
    ("\u0d32\u0d4d\u200d", "\u0d7d"), ("\u0d33\u0d4d\u200d", "\u0d7e"), ("\u0d15\u0d4d\u200d", "\u0d7f"),
)
_VISARGA = re.compile(r"([\u0d00-\u0d7f]):")


class MalayalamNormalizer:
    def normalize(self, text: str) -> str:
        for a, b in _OLD_CHILLU:
            text = text.replace(a, b)
        # BaseNormalizer.normalize
        text = text.replace(_BOM, "").replace(_BOM2, "").replace(_WJ, "").replace(_SHY, "")
        text = text.replace(_ZWSP, " ").replace(_NBSP, " ")
        text = text.replace(_ZWNJ, "").replace(_ZWJ, "")
        for a, b in _PUNCT:
            text = text.replace(a, b)
        # MalayalamNormalizer.normalize
        text = text.replace("\u0d64", "\u0964").replace("\u0d65", "\u0965")  # script dandas -> generic
        text = text.replace("\u0d46\u0d3e", "\u0d4a").replace("\u0d47\u0d3e", "\u0d4b")  # two-part vowels
        text = text.replace("\u0d46\u0d57", "\u0d4c").replace("\u0d57", "\u0d4c")  # au forms
        return _VISARGA.sub("\\1\u0d03", text)
