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
    # normalize(mode="inference") also drops tags / ⁇ and folds Indic digits; itn_numbers does
    # not. Neither drops group commas (digits are left as written).
    text = "<noise> ⁇ ൨൪ 70,000"
    assert itn_numbers(text) == text
    assert normalize(text, mode="inference") == "24 70,000"


# --- digit-by-digit runs: a code (4+ digits, or a zero, or ഡബിൾ / ട്രിപ്പിൾ) is one number;
# --- a short run is separate digits (counting, "two or three").
@pytest.mark.parametrize("text, expected", [
    ("ഒന്ന് രണ്ട് മൂന്ന്", "1 2 3"),
    ("രണ്ട് മൂന്ന് പേർ", "2 3 പേർ"),
    ("ഒന്ന് രണ്ട് മൂന്ന് നാല്", "1234"),
    ("ഒന്ന് പൂജ്യം രണ്ട്", "102"),
    ("പൂജ്യം ഏഴ്", "07"),
    ("വൺ ടു ത്രീ", "1 2 3"),
    ("വൺ ടു ത്രീ ഫോർ", "1234"),
    ("സീറോ വൺ", "01"),
    ("ഡബിൾ ടു", "22"),
    ("നൈൻ ഡബിൾ അഞ്ച്", "955"),
    ("ഒൻപത് എട്ട് ഏഴ് ആറ് അഞ്ച് നാല് മൂന്ന് രണ്ട് ഒന്ന് പൂജ്യം", "9876543210"),
])
def test_digit_runs(text, expected):
    assert itn_numbers(text) == expected


# --- spellings the digits generator does not write, read back by ITN ----------------------
@pytest.mark.parametrize("text, expected", [
    ("ഒൻപത്", "9"), ("അൻപത് രൂപ", "50 രൂപ"), ("ഇരുപത്തൊൻപത്", "29"), ("ഇരുപത്തി ഒൻപത്", "29"),
    ("നാല്പത്", "40"), ("നാൽപത്", "40"), ("എമ്പത്", "80"), ("ഇരുന്നൂറ്", "200"),
    ("അൻപതിനായിരം", "50000"),
    ("ഒന്ന് ഒൻപത്", "1 9"),                          # ഒന്ന് beside ഒൻപത് is a number, not "just"
    ("ടൂ ലാക്", "2,00,000"),                         # ടൂ = two (beside a number word)
    ("ലാക്", "ലാക്"),                                # a bare lakh word has no count: a word
    ("ഹൺഡ്രഡ് പെർസെന്റ്", "100%"),
])
def test_spelling_variants(text, expected):
    assert itn_numbers(text) == expected


def test_itn_flag_off_in_inference():
    text = "Hello, ഇരുപത്തിനാല് പേർ! ദു:ഖം"
    assert normalize(text, mode="inference") == "Hello, 24 പേർ! ദുഃഖം"
    assert normalize(text, mode="inference", itn=False) == "Hello, ഇരുപത്തിനാല് പേർ! ദുഃഖം"
    assert normalize(text, mode="eval", itn=False) == normalize(text, mode="eval")   # no effect
    from indic_normalizer import get_normalizer, normalize as top
    fn, _ = get_normalizer("ml", mode="inference", itn=False)
    assert fn(text) == top(text, lang="ml", mode="inference", itn=False) == "Hello, ഇരുപത്തിനാല് പേർ! ദുഃഖം"
    assert top("ഇരുപത്തിനാല്", mode="inference", itn=False) == "ഇരുപത്തിനാല്"     # universal: accepted
