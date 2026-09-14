"""Tests for the Malayalam normalizer — including the crown-jewel regression pins
(E12 conjunct fix, E8/E9 number canonicalization) and the vanishing-vowels guardrail.
"""

import pytest

from indic_normalizer import get_normalizer

norm, source = get_normalizer("ml")


def test_registered():
    assert source == "indic_normalizer.malayalam"
    for alias in ("ml", "mal", "malayalam", "Malayalam", " ML "):
        fn, src = get_normalizer(alias)
        assert src == "indic_normalizer.malayalam"


# --- E12: mis-encoded chillu conjunct (ൻ്റ -> ന്റ), NFC-invisible ---------------------
def test_chillu_conjunct_fix():
    # chillu-n + virama + rra  vs  na + virama + rra — same word, must normalize equal
    wrong = "വിവേകാനന്ദൻ്റെ"   # contains ൻ + ് + റ
    right = "വിവേകാനന്ദന്റെ"    # contains ന + ് + റ
    assert norm(wrong) == norm(right)


def test_genuine_word_final_chillu_untouched():
    # a real word-final chillu (no following virama) must survive, not be rewritten
    out = norm("അവൻ")
    assert "ൻ" in out  # chillu-n preserved


# --- E8: native digit verbalization --------------------------------------------------
@pytest.mark.parametrize("digits,expected", [
    ("24", "ഇരുപത്തിനാല്"),
    ("500", "അഞ്ഞൂറ്"),
    ("21", "ഇരുപത്തിയൊന്ന്"),   # sandhi glide യ
])
def test_digit_verbalization(digits, expected):
    assert norm(digits) == expected


def test_id_like_digits_read_digit_by_digit():
    # zero-padded / long runs -> digit by digit
    assert norm("007") == "പൂജ്യം പൂജ്യം ഏഴ്"


def test_indic_digits_folded_then_verbalized():
    assert norm("൨൪") == norm("24")  # Malayalam 24 -> same as ASCII 24


# --- E9: English-style number words -> canonical Malayalam ----------------------------
def test_english_number_canonicalization():
    # "24", English-in-Malayalam-script, and native Malayalam all collapse to one form
    canonical = norm("24")
    assert norm("ട്വൻറ്റി ഫോർ") == canonical
    assert norm("ഇരുപത്തിനാല്") == canonical


def test_group_comma_collapsed():
    assert norm("70,000") == norm("70000")


def test_ambiguous_to_for_left_alone():
    # bare ടു / ഫോർ (English to/for) with no unambiguous number word must not convert
    out = norm("പ്ലസ് ടു")
    assert "ടു" in out          # unchanged, not turned into രണ്ട്


# --- Guardrail: vanishing-vowels must NOT happen (ACL-2024 identity check) ------------
def test_no_vanishing_vowels():
    # a plain Malayalam sentence keeps its word count and its vowel signs
    s = "ഇതൊരു ഉദാഹരണമാണ്"
    out = norm(s)
    assert len(out.split()) == len(s.split())   # word count preserved (no vowel-split)
    for vowel_sign in ("ൊ", "ാ"):
        assert vowel_sign in out                # mātras survive


def test_punctuation_and_unk_removed():
    assert "!" not in norm("പൗരൻ!!")
    assert "⁇" not in norm("foo ⁇ bar")
