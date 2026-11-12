"""Code-mixed handlers (indic_normalizer.mixed) and the eval itn switch."""

import pytest

from indic_normalizer import get_normalizer, mixed, normalize


def ml(text, **kw):
    return normalize(text, lang="ml", **kw)


# --- options --------------------------------------------------------------------------------
def test_resolve():
    assert set(mixed.resolve(True)) == {"en"}
    assert mixed.resolve(False) == {} and mixed.resolve(None) == {}
    assert mixed.resolve({"en": False}) == {}
    assert mixed.resolve({"en": {"spell_small": True}})["en"].spell_small is True
    assert mixed.resolve(True, spell_small=True)["en"].spell_small is True     # the shortcut
    assert mixed.resolve(False, spell_small=True) == {}                        # nothing to set


@pytest.mark.parametrize("bad", [{"fr": True}, {"en": {"nope": 1}}, {"en": 3}, "yes"])
def test_bad_options_fail_early(bad):
    with pytest.raises(ValueError):
        get_normalizer("ml", mixed=bad)


# --- inference: the English handler's ITN ---------------------------------------------------
def test_inference_mixed_switch():
    assert ml("twenty four people", mode="inference") == "24 people"
    assert ml("twenty four people", mode="inference", mixed=False) == "twenty four people"
    assert ml("twenty four people", mode="inference", mixed={"en": False}) == "twenty four people"
    assert ml("ഇരുപത്തിനാല് പേർ", mode="inference", mixed=False) == "24 പേർ"   # main ITN still runs
    assert ml("I have five", mode="inference", mixed={"en": {"spell_small": True}}) == "I have five"
    assert ml("I have five", mode="inference", spell_small=True) == "I have five"


# --- eval: itn switch (off by default: numbers are scored as words) -------------------------
@pytest.mark.parametrize("ref, hyp", [
    ("ഞാൻ 24 പേർക്ക് കൊടുത്തു", "ഞാൻ ഇരുപത്തിനാല് പേർക്ക് കൊടുത്തു"),
    ("ഞാൻ 24 hours ജോലി ചെയ്തു", "ഞാൻ twenty four hours ജോലി ചെയ്തു"),
    ("2,50,000 രൂപ", "രണ്ടര ലക്ഷം രൂപ"),
    ("call 9847012345", "call nine eight four seven zero one two three four five"),
])
def test_eval_itn_scores_in_digits(ref, hyp):
    assert ml(ref, itn=True) == ml(hyp, itn=True)


def test_eval_itn_details():
    assert ml("ഞാൻ 24 പേർക്ക് കൊടുത്തു", itn=True) == "ഞാൻ 24 പേർക്ക് കൊടുത്ത്"
    assert ml("രണ്ടര ലക്ഷം രൂപ", itn=True) == "250000 രൂപ"            # no group commas
    assert ml("ഒന്ന് നോക്ക്", itn=True) == "ഒന്ന് നോക്ക്"                 # the I1 guard holds
    assert ml("24 പേർ") == "ഇരുപത്തിനാല് പേർ"                             # default: words
    assert ml("24", mode="train", itn=True) == "24"                       # no effect in train


# --- eval: digits in an English sentence are spelled out in English -------------------------
@pytest.mark.parametrize("ref, hyp", [
    ("I worked 24 hours", "I worked twenty four hours"),
    ("call 9847012345", "call nine eight four seven zero one two three four five"),
    ("Room 101.", "room one hundred one"),
])
def test_eval_english_sentence_digits_meet_english_words(ref, hyp):
    assert ml(ref) == ml(hyp)


def test_eval_english_digit_words():
    assert ml("I worked 24 hours") == "i worked twenty four hours"
    assert ml("Pay 2,50,000 now") == "pay two lakh fifty thousand now"       # group commas first
    assert ml("PIN 0471") == "pin zero four seven one"                        # leading zero: digit by digit
    # per sentence: a Malayalam sentence keeps Malayalam words, even beside an English one
    assert ml("ഞാൻ 24 hours ജോലി ചെയ്തു. I slept 8 hours.") == \
        "ഞാൻ ഇരുപത്തിനാല് hours ജോലി ചെയ്ത് i slept eight hours"
    assert ml("24 പേർ") == "ഇരുപത്തിനാല് പേർ"
    assert ml("24") == "ഇരുപത്തിനാല്"                                         # no letters: main language
    # off with mixed=False / {"en": False}; eval itn=True scores in digits instead
    assert ml("I worked 24 hours", mixed=False) == "i worked ഇരുപത്തിനാല് hours"
    assert ml("I worked 24 hours", mixed={"en": False}) == "i worked ഇരുപത്തിനാല് hours"
    assert ml("I worked twenty four hours", itn=True) == "i worked 24 hours"


# --- eval: currency signs and percent become words, in the sentence's language ---------------
@pytest.mark.parametrize("ref, hyp", [
    ("₹500", "അഞ്ഞൂറ് രൂപ"),
    ("ഞാൻ ₹500 കൊടുത്തു", "ഞാൻ അഞ്ഞൂറ് രൂപ കൊടുത്തു"),
    ("ഞാൻ Rs.500 കൊടുത്തു", "ഞാൻ 500 രൂപ കൊടുത്തു"),
    ("500₹", "അഞ്ഞൂറ് രൂപ"),
    ("₹ 70,000", "എഴുപതിനായിരം രൂപ"),
    ("ഞാൻ $24 കൊടുത്തു", "ഞാൻ ഇരുപത്തിനാല് ഡോളർ കൊടുത്തു"),
    ("ഞാൻ 20% കൊടുത്തു", "ഞാൻ ഇരുപത് ശതമാനം കൊടുത്തു"),
    ("ഞാൻ 20 % കൊടുത്തു", "ഞാൻ 20 ശതമാനം കൊടുത്തു"),
    ("I paid ₹500", "I paid five hundred rupees"),
    ("It costs $24.", "It costs twenty four dollars."),
    ("That is $1 only", "That is one dollar only"),            # singular for 1
    ("€10 ticket", "ten euros ticket"),
    ("I got 20% off", "I got twenty percent off"),
    ("ഞാൻ ₹500 കൊടുത്തു. I paid 20% tax.", "ഞാൻ അഞ്ഞൂറ് രൂപ കൊടുത്തു. I paid twenty percent tax."),
])
def test_eval_signs_meet_spoken_words(ref, hyp):
    assert ml(ref) == ml(hyp)
    assert ml(ref, itn=True) == ml(hyp, itn=True)               # also when scoring in digits


def test_signs_details():
    assert ml("ഞാൻ ₹500 കൊടുത്തു") == "ഞാൻ അഞ്ഞൂറ് രൂപ കൊടുത്ത്"
    assert ml("I got 20% off") == "i got twenty percent off"
    assert ml("ഞാൻ 20% കൊടുത്തു", itn=True) == "ഞാൻ 20 ശതമാനം കൊടുത്ത്"
    assert ml("ഞാൻ ₹ കൊടുത്തു") == "ഞാൻ കൊടുത്ത്"                # a lone sign is dropped
    assert ml("hours 500") == "hours five hundred"              # rs only as a word of its own
    assert ml("ഞാൻ ₹500 കൊടുത്തു", mode="train") == "ഞാൻ ₹500 കൊടുത്തു"        # train keeps the sign
    assert ml("ഞാൻ 20% കൊടുത്തു", mode="inference") == "ഞാൻ 20% കൊടുത്തു"
    assert ml("version 2.0.1") == "version two point zero one"   # a dot inside a number ends no sentence
