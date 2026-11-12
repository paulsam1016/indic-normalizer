"""The built-in port of indic-nlp-library's Malayalam normaliser (_indicnlp_ml).

Two layers:
- pinned outputs, which need nothing extra and always run;
- an equivalence check against the real indic-nlp-library 0.92
  (IndicNormalizerFactory().get_normalizer("ml")), which skips unless the `dev` extra
  (indic-nlp-library) is installed.
"""

import ast
import pathlib
import unicodedata

import pytest

from indic_normalizer._indicnlp_ml import MalayalamNormalizer
from indic_normalizer.languages import malayalam as ml

port = MalayalamNormalizer()

# Edge cases also used when the port was checked against indic-nlp-library in stt.
EDGE_CASES = [
    "ൻ്റ", "ന്‍", "ണ്‍ ന്‍ ര്‍ ല്‍ ള്‍ ക്‍", "ൺ ൻ ർ ൽ ൾ ൿ", "കൊ കോ കൌ കൗ",
    "കൊ കോ കൌ കൗ", "ദു:ഖം", "ആണ്: ശരി", "a:b 10:30",
    "﻿￾⁠­ x", "a​b c", "a‌b‍c",
    "„q“ ”x– y—z ´a‘b‚c’d ''e ´´f …", "ഽ ൄ ൢ ൣ ൦൧൨", "൤ ൥",
    "ഩ ഺ ൎ ൔ ൕ ൖ ൘ ൟ", "ഁ ഀ ഄ ഼", "tamil க் hindi क़ ड़ ँ", "ട്ട ഺ്ഺ",
    "ൿ ൺ‍", "", "  ", "ABC abc 123", "ൌ ൗ",
]


# --- pinned outputs (no indic-nlp needed) ---------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    # old (Unicode <= 5.0) chillu: consonant + virama + ZWJ -> atomic chillu
    ("ന്‍", "ൻ"),
    ("ണ്‍ ന്‍ ര്‍ ല്‍ ള്‍ ക്‍",
     "ൺ ൻ ർ ൽ ൾ ൿ"),
    ("ൺ ൻ ർ ൽ ൾ ൿ", "ൺ ൻ ർ ൽ ൾ ൿ"),                     # atomic chillu untouched
    ("ൿ ൺ‍", "ൿ ൺ"),                              # stray ZWJ after a chillu dropped
    # two-part vowel signs -> single code point
    ("കൊ", "കൊ"),                       # െ + ാ -> ൊ
    ("കോ", "കോ"),                       # േ + ാ -> ോ
    ("കൊ കോ", "കൊ കോ"),
    # au forms: െ + ൗ and the bare AU length mark both -> ൌ
    ("കൌ", "കൌ"),
    ("കൗ", "കൌ"),
    ("ൌ ൗ", "ൌ ൌ"),
    # colon -> visarga after a Malayalam letter only
    ("ദു:ഖം", "ദുഃഖം"),
    ("ആണ്: ശരി", "ആണ്ഃ ശരി"),          # raw normaliser; normalize() shields this colon
    ("a:b 10:30", "a:b 10:30"),
    # format / space characters
    ("﻿￾⁠­ x", " x"),               # BOM, U+FFFE, WJ, soft hyphen removed
    ("a​b c", "a b c"),                       # ZWSP, NBSP -> space
    ("a‌b‍c", "abc"),                         # ZWNJ, ZWJ removed
    # punctuation map (order matters: ´´ -> '' -> ")
    ("„q“ ”x– y—z ´a‘b‚c’d ''e ´´f …", '"q" "x- y - z \'a\'b\'c\'d "e "f ...'),
    # Malayalam-block dandas -> generic Devanagari dandas
    ("൤ ൥", "। ॥"),
    # nothing to do
    ("", ""),
    ("  ", "  "),
    ("ABC abc 123", "ABC abc 123"),
    ("tamil க் hindi क़ ड़ ँ", "tamil க் hindi क़ ड़ ँ"),
])
def test_pinned(text, expected):
    assert port.normalize(text) == expected


def test_is_the_normaliser_malayalam_uses():
    assert isinstance(ml._IND_NORM, MalayalamNormalizer)
    assert ml.INDIC_NLP_AVAILABLE is True                 # backward-compatibility flag


def test_colon_guard_still_applies_in_normalize():
    assert ml.normalize("ആണ്: ശരി", mode="inference") == "ആണ്: ശരി"
    assert ml.normalize("ദു:ഖം", mode="inference") == "ദുഃഖം"


# --- equivalence with the real indic-nlp-library ---------------------------------------------
def _existing_malayalam_strings() -> list[str]:
    """Every string literal containing Malayalam in this test suite's files."""
    out = set()
    for p in pathlib.Path(__file__).parent.glob("test_*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and any("ഀ" <= c <= "ൿ" for c in node.value)):
                out.add(node.value)
    return sorted(out)


def test_equivalent_to_indic_nlp_library():
    factory = pytest.importorskip("indicnlp.normalize.indic_normalize").IndicNormalizerFactory
    ref = factory().get_normalizer("ml")
    texts = EDGE_CASES + _existing_malayalam_strings()
    assert len(texts) > 100
    diffs = []
    for t in texts:
        for v in {t, unicodedata.normalize("NFC", t), unicodedata.normalize("NFD", t)}:
            if port.normalize(v) != ref.normalize(v):
                diffs.append((v, port.normalize(v), ref.normalize(v)))
    assert not diffs, diffs[:10]
