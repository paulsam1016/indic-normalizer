"""Tests for the language-agnostic universal normalizer."""

import functools

import pytest

from indic_normalizer import normalize, get_normalizer
from indic_normalizer import universal


def test_get_normalizer_defaults_to_universal():
    fn, source = get_normalizer(None)
    # fn is a mode-bound partial around universal.normalize (default mode="eval")
    assert isinstance(fn, functools.partial) and fn.func is universal.normalize
    assert source == "indic_normalizer.universal"
    assert fn("  Hello  ") == "hello"          # behaves as the eval universal normalizer


def test_unknown_lang_falls_back_to_universal():
    fn, source = get_normalizer("klingon")
    assert source == "indic_normalizer.universal"


# --- mode selector on the universal normalizer (only E5 punctuation + E10 case toggle) ---
def test_universal_inference_keeps_punct_and_case():
    assert normalize("Hello, World!", mode="inference") == "Hello, World!"
    assert normalize("Hello, World!") == "hello world"        # eval default


def test_universal_inference_still_folds_digits_and_whitespace():
    # shared base still runs in inference: Indic digits -> ASCII, whitespace collapsed
    assert normalize("  ൧൨൩  ", mode="inference") == "123"


def test_universal_mode_aliases():
    for m in ("train", "training", "fine-tune", "evaluation"):
        assert normalize("Hello!", mode=m) == "hello"         # all behave like eval


def test_universal_bad_mode_raises():
    with pytest.raises(ValueError):
        get_normalizer(None, mode="bogus")


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


def test_invisible_chars():
    assert normalize("a​b") == "a b"                       # zero-width space = word break
    assert normalize("a­b") == normalize("ab") == "ab"      # soft hyphen
    assert normalize("a⁠b") == "ab"                         # word joiner
    assert normalize("﻿ab￾") == "ab"


def test_group_commas():
    assert normalize("1,000 रुपये") == normalize("1000 रुपये") == "1000 रुपये"
    assert normalize("1,75,000") == "175000"                     # Indian lakh grouping
    assert normalize("1,2,3") == "1 2 3"                         # a list, not a number
    assert normalize("1,000", mode="inference") == "1,000"       # reader-facing: as written


@pytest.mark.parametrize("text, expected", [
    ("It's fine, I can't.", "its fine i cant"),          # joined, not "it s" / "can t"
    ("it’s", "its"),                                     # curly apostrophe
    ("can`t", "cant"),
    ("The students' books", "the students books"),       # not inside a word: punctuation
    ("'quoted' text", "quoted text"),
])
def test_apostrophe_inside_a_word_joins(text, expected):
    # train (and the universal normalizer) join; eval expands contractions first (test_english)
    assert normalize(text) == expected
    assert normalize(text, lang="ml", mode="train") == expected
    assert normalize(text, lang="en", mode="train") == expected
    assert "'" in normalize("It's", lang="ml", mode="inference")   # inference keeps punctuation
