"""English number words in Latin script -> digits (languages.english_itn), alone and inside
Malayalam inference ITN, and consistent with the Malayalam-script English reading."""

import pytest

from indic_normalizer import normalize
from indic_normalizer.base import indian_grouping
from indic_normalizer.digits import latin as L
from indic_normalizer.digits import ml as M
from indic_normalizer.languages.english_itn import itn
from indic_normalizer.languages.malayalam import itn_numbers


@pytest.mark.parametrize("text, expected", [
    # cardinals; Indian grouping from one lakh up; "and", "a hundred", bare hundred / thousand
    ("twenty four", "24"),
    ("I have twenty four apples", "I have 24 apples"),
    ("five hundred rupees", "500 rupees"),             # currency stays a word
    ("one hundred and five", "105"),
    ("hundred and five", "105"),
    ("one thousand two hundred", "1200"),
    ("thousand six hundred", "1600"),
    ("twelve hundred", "1200"),
    ("hundred", "100"),
    ("a hundred people", "100 people"),
    ("two lakh", "2,00,000"),
    ("a lakh", "1,00,000"),
    ("lakh fifty thousand", "1,50,000"),
    ("three lakh fifty thousand", "3,50,000"),
    ("two million", "20,00,000"),
    ("twenty-five thousand", "25000"),
    ("Twenty-Four", "24"),
    ("twenty k", "20k"),
    ("five k", "5k"),
    # a repeated scale or hundred starts a new number
    ("ten thousand fifteen thousand", "10000 15000"),
    ("two hundred three hundred", "200 300"),
    # ordinals, decimals, percent, years
    ("twenty first", "21st"),
    ("Twenty-First", "21st"),
    ("one hundred and first", "101st"),
    ("twelfth", "12th"),
    ("third", "3rd"),
    ("three point one four", "3.14"),
    ("two point five percent", "2.5%"),
    ("five percent", "5%"),
    ("fifty per cent.", "50%."),
    ("nineteen ninety five", "1995"),
    ("twenty twenty four", "2024"),
    ("twenty thirty", "2030"),
    # o'clock
    ("one o clock", "1 O’Clock"),
    ("five o'clock", "5 O’Clock"),
    ("eleven o’clock.", "11 O’Clock."),
    ("one oclock", "1 O’Clock"),
    # digits one by one: a code is one number, a short run separate digits
    ("nine eight four seven", "9847"),
    ("nine eight seven six five four three two one zero", "9876543210"),
    ("double oh seven", "007"),
    ("double five", "55"),
    ("one two double zero", "1200"),
    ("two three", "2 3"),
    ("one two three", "1 2 3"),
    # lone small numbers convert (like the Malayalam ITN), except "one" without numeric context
    # and a lone first / second
    ("I have five", "I have 5"),
    ("zero", "0"),
    ("one of them", "one of them"),
    ("one", "one"),
    ("one day", "1 day"),
    ("only one person five days", "only 1 person 5 days"),
    ("for five to six days", "for 5 to 6 days"),
    ("one to one meeting", "one to one meeting"),
    ("the first time", "the first time"),
    ("first, second and third", "first, second and 3rd"),
    # ambiguous pairs (time / price / range) stay words
    ("by twelve thirty", "by twelve thirty"),
    ("four twenty eight", "four twenty eight"),
    ("twenty four seven", "twenty four seven"),
    ("twenty twenty", "twenty twenty"),
    # not numbers
    ("I went to school for two years", "I went to school for 2 years"),
    ("hundreds of people", "hundreds of people"),
    ("lakhs of people", "lakhs of people"),
    ("lakh", "lakh"),
    ("wait a second", "wait a second"),
    ("500 rupees", "500 rupees"),                      # digits already written are left alone
    ("", ""),
])
def test_english_itn(text, expected):
    assert itn(text) == expected
    assert itn_numbers(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("I have five", "I have five"),
    ("one of them", "one of them"),
    ("the third time", "the third time"),
    ("five days", "5 days"),
    ("twenty four", "24"),
])
def test_spell_small_english_style(text, expected):
    assert itn(text, spell_small=True) == expected


def test_code_mixed_with_malayalam():
    assert itn_numbers("ഞാൻ twenty four മണിക്കൂർ ജോലി ചെയ്തു") == "ഞാൻ 24 മണിക്കൂർ ജോലി ചെയ്തു"
    assert itn_numbers("ട്വന്റി ഫോർ and twenty four") == "24 and 24"
    assert itn_numbers("ഇരുപത്തിനാല് പേർ, twenty five people!") == "24 പേർ, 25 people!"
    assert itn_numbers("ടൂ ലാക് or two lakh") == "2,00,000 or 2,00,000"
    assert itn_numbers("one രൂപ") == "1 രൂപ"            # a Malayalam unit word is context too
    assert itn_numbers("one കാര്യം") == "one കാര്യം"


def test_inference_mode_and_switch():
    text = "Hello, twenty four PEOPLE!"
    assert normalize(text, lang="ml", mode="inference") == "Hello, 24 PEOPLE!"
    assert normalize(text, lang="ml", mode="inference", itn=False) == text
    assert normalize("twenty four", lang="ml") == "twenty four"         # eval: no ITN


# --- the same number in Latin and in Malayalam-script English reads back the same (lone 1 / 2 / 4
# --- differ on purpose: Latin "one" is also a pronoun, ടു / ഫോർ are also "to" / "for") ----------
@pytest.mark.parametrize("n", list(range(3, 1000, 7)) + [1000, 1600, 12500, 99999, 100000, 175000,
                                                         2500000, 10000000, 123456789])
def test_latin_and_malayalam_script_agree(n):
    assert itn_numbers(L.words(n)) == itn_numbers(M.en_words(n)) == indian_grouping(n)


@pytest.mark.parametrize("latin, malayalam, expected", [
    ("nineteen ninety five", "നൈന്റീൻ നൈന്റി ഫൈവ്", "1995"),
    ("twenty twenty four", "ട്വന്റി ട്വന്റി ഫോർ", "2024"),
    ("twelve thirty", "ട്വൽവ് തേർട്ടി", None),                 # stays words
    ("four twenty eight", "ഫോർ ട്വന്റി എയ്റ്റ്", None),
    ("ten thousand fifteen thousand", "ടെൻ തൗസന്റ് ഫിഫ്റ്റീൻ തൗസന്റ്", "10000 15000"),
    ("thousand six hundred", "തൗസന്റ് സിക്സ് ഹൺഡ്രഡ്", "1600"),
    ("twenty k", "ട്വന്റി കെ", "20k"),
    ("two point five", "ടു പോയിന്റ് ഫൈവ്", "2.5"),
    ("double five", "ഡബിൾ ഫൈവ്", "55"),
    ("third", "തേർഡ്", "3rd"),
    ("nineteen oh five", "നൈന്റീൻ ഓ ഫൈവ്", "1905"),          # oh / ഓ said as zero
    ("four oh five", "ഫോർ ഓ ഫൈവ്", "405"),
    ("double oh seven", "ഡബിൾ ഓ സെവൻ", "007"),
    ("nine eight oh seven", "നൈൻ എയ്റ്റ് ഓ സെവൻ", "9807"),
])
def test_same_rules_in_both_scripts(latin, malayalam, expected):
    assert itn_numbers(latin) == (expected or latin)
    assert itn_numbers(malayalam) == (expected or malayalam)


def test_malayalam_o_is_a_word_elsewhere():
    assert itn_numbers("ഓ ശരി") == "ഓ ശരി"
    assert itn_numbers("ഫൈവ് ഓ ക്ലോക്ക്") == "5 ഓ ക്ലോക്ക്"
