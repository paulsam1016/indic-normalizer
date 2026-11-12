"""indic_normalizer.languages.malayalam.itn_numbers: number words -> digits, nothing else."""

import pytest

from indic_normalizer.languages.malayalam import itn_numbers, normalize


@pytest.mark.parametrize("text, expected", [
    ("ഇരുനൂറ്റമ്പത് രൂപ", "250 രൂപ"),
    ("ഒന്ന് നോക്ക്", "ഒന്ന് നോക്ക്"),                     # lone ഒന്ന് = "just", not 1
    ("ഒന്ന് രൂപ", "1 രൂപ"),                              # ... but 1 before a unit
    ("ഡബിൾ ടു", "22"),
    ("Hello, ഇരുപത്തിനാല് പേർ! OK?", "Hello, 24 പേർ! OK?"),   # punctuation + case kept
    ("ദു:ഖം, ആണ്: ശരി", "ദുഃഖം, ആണ്: ശരി"),              # colon guard as in normalize()
    ("  a   b ", "a b"),
    ("", ""),
])
def test_itn_numbers(text, expected):
    assert itn_numbers(text) == expected


def test_only_numbers_change():
    # normalize(mode="inference") also drops tags / ⁇ and folds Indic digits and group commas;
    # itn_numbers does not.
    text = "<noise> ⁇ ൨൪ 70,000"
    assert itn_numbers(text) == text
    assert normalize(text, mode="inference") == "24 70000"
