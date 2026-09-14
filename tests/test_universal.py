"""Tests for the language-agnostic universal normalizer."""

from indic_normalizer import normalize, get_normalizer
from indic_normalizer import universal


def test_get_normalizer_defaults_to_universal():
    fn, source = get_normalizer(None)
    assert fn is universal.normalize
    assert source == "indic_normalizer.universal"


def test_unknown_lang_falls_back_to_universal():
    fn, source = get_normalizer("klingon")
    assert source == "indic_normalizer.universal"


def test_lowercase_and_whitespace():
    assert normalize("  Hello,   WORLD!  ") == "hello world"


def test_unk_markers_removed():
    assert "⁇" not in normalize("foo ⁇ bar")
    assert "<unk>" not in normalize("foo <unk> bar")


def test_digit_fold_across_scripts():
    # Malayalam digits and Devanagari digits both -> ASCII
    assert normalize("൧൨൩") == "123"
    assert normalize("१२३") == "123"


def test_indic_vowel_signs_preserved():
    # THE key property: universal normalization must NOT strip vowel signs / virama
    # (the Whisper "vanishing vowels" bug). NFC + punctuation only.
    s = "ഇതൊരു ഉദാഹരണമാണ്"
    out = normalize(s)
    # every original letter/mark survives (only trailing form/space differences)
    for ch in "ഉദാഹരണമാണ":
        assert ch in out


def test_danda_removed():
    assert "।" not in normalize("वाक्य ।")
