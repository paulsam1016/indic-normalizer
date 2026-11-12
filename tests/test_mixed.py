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
