"""English as the main language (lang="en") and the English eval rules shared with code-mixed
English: canonical forms, number readings, currency / percent."""

import pytest

from indic_normalizer import get_normalizer, normalize


def en(text, **kw):
    return normalize(text, lang="en", **kw)


def test_registered():
    for alias in ("en", "eng", "english", " English "):
        assert get_normalizer(alias)[1] == "indic_normalizer.english"


# --- eval: reference and hypothesis meet ----------------------------------------------------
@pytest.mark.parametrize("ref, hyp", [
    ("I paid 24 dollars on the 21st.", "I paid twenty four dollars on the twenty first."),
    ("It's 2.5% at 4:30 PM.", "it is two point five percent at four thirty pm"),
    ("Meet at 4:05", "meet at four oh five"),
    ("Meet at 10:00", "meet at ten o clock"),
    ("in 1995", "in nineteen ninety five"),
    ("I paid ₹500 & $24", "I paid five hundred rupees and twenty four dollars"),
    ("Call 9847012345", "call nine eight four seven zero one two three four five"),
    # canonical forms (eval only)
    ("I can't go", "I cannot go"),
    ("we'll see", "we will see"),
    ("it's done", "it is done"),
    ("it's been long", "it has been long"),
    ("let's go", "let us go"),
    ("I'm gonna wait", "I am going to wait"),
    ("Um, I think so", "I think so"),
    ("hmm OK", "okay"),
    ("salt & pepper", "salt and pepper"),
])
def test_eval_ref_meets_hyp(ref, hyp):
    assert en(ref) == en(hyp)


def test_eval_details():
    assert en("Um, I can't do it, OK?") == "i can not do it okay"
    assert en("John's car") == "johns car"                     # possessive: apostrophe joined
    assert en("uh huh yes") == "uh huh yes"                    # uh huh = yes: kept
    assert en("mhm") == "mhm"
    assert en("book") == "book"


@pytest.mark.parametrize("ref, hyp", [
    ("I paid 24 dollars on the 21st.", "I paid twenty four dollars on the twenty first."),
    ("It's at 4:30 PM", "it is at four thirty pm"),
    ("I paid ₹500", "I paid five hundred rupees"),
    ("I got 20% off", "I got twenty percent off"),
    ("in 1995", "in nineteen ninety five"),
])
def test_eval_itn_scores_in_digits(ref, hyp):
    assert en(ref, itn=True) == en(hyp, itn=True)


# --- train / inference ----------------------------------------------------------------------
def test_train_keeps_what_was_said():
    s = "Um, I can't pay twenty four dollars, I have 5."
    assert en(s, mode="train") == "um i cant pay twenty four dollars i have 5"


def test_inference():
    s = "Um, I can't pay twenty four dollars, I have five."
    assert en(s, mode="inference") == "Um, I can't pay 24 dollars, I have five."      # English style
    assert en(s, mode="inference", spell_small=False) == "Um, I can't pay 24 dollars, I have 5."
    assert en(s, mode="inference", itn=False) == s
    assert en("Meet at four thirty PM.", mode="inference") == "Meet at 4:30 PM."


# --- the same eval rules for English inside Malayalam ---------------------------------------
@pytest.mark.parametrize("ref, hyp", [
    ("I came on the 21st", "I came on the twenty first"),
    ("I scored 2.5 points", "I scored two point five points"),
    ("ഞാൻ can't പറഞ്ഞു", "ഞാൻ can not പറഞ്ഞു"),
    ("ഞാൻ 2.5 കിലോ വാങ്ങി", "ഞാൻ രണ്ടര കിലോ വാങ്ങി"),              # Malayalam: the digits module's reading
    ("21st ന് വരാം", "ഇരുപത്തിയൊന്നാം ന് വരാം"),             # glide spelling, as E8 writes integers
    ("4:30 ന് വരാം", "നാലര ന് വരാം"),
    ("ഞാൻ 3.14 എഴുതി", "ഞാൻ മൂന്ന് പോയിന്റ് ഒന്ന് നാല് എഴുതി"),
])
def test_code_mixed_eval(ref, hyp):
    assert normalize(ref, lang="ml") == normalize(hyp, lang="ml")


def test_canonical_off():
    assert normalize("I can't go", lang="ml", mixed={"en": {"canonical": False}}) == "i cant go"
    assert normalize("I can't go", lang="ml", mixed=False) == "i cant go"
