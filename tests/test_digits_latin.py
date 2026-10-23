import itertools
import re

import pytest

from indic_normalizer.digits import candidates, find, latin, ml, verbalize
from indic_normalizer.languages.malayalam import normalize


def V(t):
    return verbalize(normalize(t, mode="train"))


def test_english_sentences_get_latin_readings():
    assert V("I was looking for a room for 5 to 6 days.") == "i was looking for a room for five to six days"
    assert V("by 12:30 or 1 o clock") == "by twelve thirty or one o clock"
    assert V("charge 1200 rupees") == "charge one thousand two hundred rupees"
    assert V("Tax rate 5%.") == "tax rate five percent"
    assert V("A1 table page") == "a one table page"
    assert V("ഫീസ് 25 ആണ്") == "ഫീസ് ഇരുപത്തഞ്ച് ആണ്"                     # Malayalam stays Malayalam
    c = {x.text for x in candidates("1200", latin=True)}
    assert {"one thousand two hundred", "twelve hundred", "one two double zero"} <= c
    assert all(x.lang == "en" and x.form.startswith("latin") for x in candidates("12:30", latin=True))
    assert {x.text for x in candidates("11:00", latin=True)} >= {"eleven o clock", "eleven"}
    assert latin.ordinal(20) == "twentieth" and latin.ordinal(12) == "twelfth" and latin.ordinal(21) == "twenty first"
    assert latin.is_english("I was looking 5") and not latin.is_english("ഫീസ് 25 ആണ് ok")


def test_latin_never_leaves_digits():
    shapes = ["{n}", "{n}%", "₹{n}", "{n}.5", "{n}:30", "{n}-{n}", "{n}th", "{n}/4", "{n}K", "a{n}", "+91 {n}"]
    for n, s in itertools.product(["0", "5", "12", "25", "100", "1200", "15000", "800000", "9876543210"], shapes):
        t = s.format(n=n)
        for sp in find(t):
            for c in ml.candidates(sp.core, sp.cls, latin=True):
                assert not re.search(r"\d", c.text), (t, c)


def test_new_malayalam_readings():
    def texts(t):
        return {x.form: x.text for x in ml.candidates(t)}
    assert texts("3-ഇയർ")["english"] == "ത്രീ ഇയർ" and texts("4-സ്റ്റാർ")["english"] == "ഫോർ സ്റ്റാർ"
    assert texts("1-ഓൺ")["english"] == "വൺ ഓൺ" and texts("2-ഇയറിന്റെ")["english"] == "ടു ഇയറിന്റെ"
    assert texts("11ക്ക്")["oclock_manikku"] == "പതിനൊന്ന് മണിക്ക്" and texts("1-ന്")["oclock_manikku"] == "ഒരു മണിക്ക്"
    assert "oclock_manikku" not in texts("13-ന്") and "oclock_manikku" not in texts("9-ൽ")
    assert texts("325000-ന്")["drop_thousand"] == "മൂന്ന് ലക്ഷത്തി ഇരുപത്തഞ്ചിന്"
    assert texts("24500")["drop_thousand_e"] == "ഇരുപത്തിനാലേ അഞ്ഞൂറ്"
    assert texts("15000")["english_k"] == "ഫിഫ്റ്റീൻ കെ" and "english_k" not in texts("325000")
    assert texts("800000")["english_lakhs"] == "എയ്റ്റ് ലാക്സ്"
    assert {"ഒരു ലക്ഷത്തി എഴുപതിനായിരം", "വൺ ലാക് സെവന്റി തൗസന്റ്"} <= set(x.text for x in ml.candidates("1.70000"))
    assert verbalize(normalize("ഇത് 3+2 ആണ്", mode="train")) == "ഇത് മൂന്ന് പ്ലസ് രണ്ട് ആണ്"


def test_spelling_families():
    from indic_normalizer.digits.scoring.pipeline import family
    c = {(x.lang, x.form): x for x in ml.candidates("24500")}
    assert family(c[("ml", "drop_thousand_e")]) == family(c[("ml", "drop_thousand")])
    c = {x.form: x for x in ml.candidates("800000")}
    assert family(c["english_lakhs"]) == family(c["english"])


def test_pipeline_uses_latin_for_english_sentences():
    from indic_normalizer.digits.scoring import load_config
    from indic_normalizer.digits.scoring.pipeline import plan_text, same_document
    doc = {"sentences": [{"text": "Only 1 person, 5 days.", "start_time": 0, "end_time": 2},
                         {"text": "ഫീസ് 25 ആണ്", "start_time": 2, "end_time": 3}], "words": []}
    cfg = load_config(None)
    plans = plan_text(doc, cfg)
    assert plans[0]["latin"] and [n.cands[0].text for n in plans[0]["numbers"]] == ["one", "five"]
    assert not plans[1]["latin"] and plans[1]["numbers"][0].cands[0].text == "ഇരുപത്തഞ്ച്"
    out, _, _ = same_document(doc, plans, cfg, choices={0: ["one", "five"], 1: ["ഇരുപത്തഞ്ച്"]})
    assert out["sentences"][0]["text"] == "only one person five days"


def test_lakh_shorthand_and_missing_point():
    c = {(x.lang, x.form, x.text) for x in ml.candidates("155")}
    assert ("ml", "lakh_e", "ഒന്നേ അമ്പത്തഞ്ച്") in c and ("en", "lakh_short", "വൺ ലാക് ഫിഫ്റ്റി ഫൈവ്") in c
    assert ("ml", "lakh_short", "ഒരു ലക്ഷത്തി അമ്പത്തഞ്ച്") in c and ("ml", "lakh_e_glide", "ഒന്നേ അമ്പത്തിയഞ്ച്") in c
    assert ml.candidates("155")[0].text == "നൂറ്റമ്പത്തഞ്ച്" and not any("lakh" in x.form for x in ml.candidates("400"))
    from indic_normalizer.digits.scoring.pipeline import family
    f = {x.form: x for x in ml.candidates("485")}
    assert family(f["lakh_e"]) == family(f["lakh_e_glide"]) == family(f["lakh_e_spaced"])
    fs = [x.form for x in ml.candidates("225-ൽ")][:12]
    assert "contracted+nosuffix" in fs and "lakh_e" in fs
    assert "രണ്ടര" in {x.text for x in candidates("25", next_word="ലക്ഷം")}
    assert "രണ്ടര" not in {x.text for x in candidates("25")}
    assert "സിക്സ് പോയിന്റ് ടു ഫൈവ്" in {x.text for x in candidates("625", next_word="ലാക്സ്")}


def test_digits_only_sentence_takes_document_language():
    from indic_normalizer.digits.scoring import load_config
    from indic_normalizer.digits.scoring.pipeline import plan_text
    doc = {"sentences": [{"text": "My code is 021."}, {"text": "0031."}, {"text": "Please call me."}], "words": []}
    plans = plan_text(doc, load_config(None))
    assert plans[1]["latin"] and plans[1]["numbers"][0].cands[0].text == "zero zero three one"
    doc = {"sentences": [{"text": "ഫീസ് എത്രയാണ്?"}, {"text": "155."}], "words": []}
    p = plan_text(doc, load_config(None))[1]
    assert not p["latin"] and "ഒന്നേ അമ്പത്തഞ്ച്" in {c.text for c in p["numbers"][0].cands}


def test_percent_has_no_digit_split():
    for t in ("55%", "12%", "100%"):
        for latin_ in (False, True):
            forms = [c.form for c in ml.candidates(t, latin=latin_)]
            assert forms and not any(("digit_by_digit" in f or "doubled" in f or "lakh" in f) for f in forms), (t, forms)


def test_sentence_languages_rules():
    from indic_normalizer.digits.latin import sentence_languages as L
    ML, EN = "ഇത് ഒരു നീണ്ട മലയാളം വാക്യം ആണ് എന്ന് പറയാം", "an english line"     # Malayalam-heavy file
    # 1. one language above 80% of the file: everything in it, no review
    assert L([ML, ML, ML, ML, "ok 5"]) == [("ml", None)] * 5
    assert L([EN, EN, EN, "ഫീസ് 5"]) == [("en", None)] * 4
    # 2. mixed file (about half and half): stretches keep their language, digits-only follow the last
    #    lettered sentence (the next one at the start)
    M, E = "മലയാളം വാക്യം", "english line"
    out = L([M, M, E, E, "155.", M])
    assert [o[0] for o in out] == ["ml", "ml", "en", "en", "en", "ml"]
    assert [o[1] is None for o in out] == [True, True, True, True, False, True]     # no letters -> review
    assert "without letters" in out[4][1] and "previous" in out[4][1]
    first = L(["155.", E, E, M, M])[0]
    assert first[0] == "en" and "next" in first[1]
    # an isolated switch -> its own language, but reviewed
    out = L([M, M, E, M, "25.", E, E])
    assert out[2][0] == "en" and "isolated English" in out[2][1] and out[3] == ("ml", None)
    assert out[4][0] == "ml" and "without letters" in out[4][1] and out[5] == ("en", None)
    out = L([M, "नमस्ते", M, E, E])
    assert out[1][0] == "ml" and "script" in out[1][1]

def test_language_issue_sends_numbers_to_review():
    from indic_normalizer.digits.scoring import load_config
    from indic_normalizer.digits.scoring.pipeline import decide, plan_text
    ML, EN = "മലയാളം വാക്യം 5", "english line 6"
    doc = {"sentences": [{"text": ML}, {"text": ML}, {"text": EN}, {"text": ML}, {"text": EN}, {"text": EN}],
           "words": []}
    cfg = load_config(None)
    plans = plan_text(doc, cfg)
    n = plans[2]["numbers"][0]
    assert plans[2]["latin"] and n.cands[0].text == "six" and "isolated English" in n.text_issue
    n.scores[("segment", "tdt")] = [0.0] + [50.0] * (len(n.cands) - 1)
    assert decide(n, "segment", "tdt", cfg)["needs_review"]
    assert plans[0]["numbers"][0].text_issue is None


def test_thousand_without_one():
    c = {x.form: x.text for x in ml.candidates("1600")}
    assert c["english_no_one"] == "തൗസന്റ് സിക്സ് ഹൺഡ്രഡ്"
    assert "thousand six hundred" in {x.text for x in ml.candidates("1600", latin=True)}
    assert "english_no_one" not in {x.form for x in ml.candidates("2600")}


def test_currency_sign_apart_or_joined():
    from indic_normalizer.digits.detect import join_currency
    assert join_currency("ഫീസ് ₹ 70000 ആണ്") == "ഫീസ് ₹70000 ആണ്" and join_currency("rs 500") == "rs500"
    assert join_currency("500 ₹ ആണ്") == "500₹ ആണ്" and join_currency("ആ rs ഇല്ല") == "ആ rs ഇല്ല"
    assert V("ഫീസ് ₹ 70,000 ആണ്") == "ഫീസ് എഴുപതിനായിരം രൂപ ആണ്"
    assert V("ഫീസ് Rs. 500 ആണ്") == "ഫീസ് അഞ്ഞൂറ് രൂപ ആണ്" and V("INR 500 only") == "five hundred rupees only"
    c = [x.text for x in ml.candidates("₹70000")]
    assert c[:4] == ["എഴുപതിനായിരം രൂപ", "സെവന്റി തൗസന്റ് റുപ്പീസ്", "എഴുപതിനായിരം", "സെവന്റി തൗസന്റ്"]
    assert not any("പൂജ്യം" in t or "ഡബിൾ" in t for t in c)


def test_currency_numbers_reviewed_and_sign_dropped_from_words():
    from indic_normalizer.digits.scoring import load_config
    from indic_normalizer.digits.scoring.pipeline import plan_text, same_document
    doc = {"sentences": [{"text": "ഫീസ് ₹ 70,000 ആണ്", "start_time": 0, "end_time": 3},
                         {"text": "ഫീസ് Rs.500 ആണ്", "start_time": 3, "end_time": 5}],
           "words": [{"word": "ഫീസ്", "start_time": 0, "end_time": .5}, {"word": "₹", "start_time": .5, "end_time": .7},
                     {"word": "70,000", "start_time": .7, "end_time": 2}, {"word": "ആണ്", "start_time": 2, "end_time": 3},
                     {"word": "ഫീസ്", "start_time": 3, "end_time": 3.5}, {"word": "Rs.500", "start_time": 3.5, "end_time": 4.5},
                     {"word": "ആണ്", "start_time": 4.5, "end_time": 5}]}
    cfg = load_config(None)
    plans = plan_text(doc, cfg)
    assert all("currency sign" in p["numbers"][0].text_issue for p in plans)
    out, stt, _ = same_document(doc, plans, cfg, choices={0: ["എഴുപതിനായിരം രൂപ"], 1: ["അഞ്ഞൂറ്"]})
    assert [s["text"] for s in out["sentences"]] == ["ഫീസ് എഴുപതിനായിരം രൂപ ആണ്", "ഫീസ് അഞ്ഞൂറ് ആണ്"]
    assert [w["word"] for w in out["words"]] == ["ഫീസ്", "എഴുപതിനായിരം", "രൂപ", "ആണ്", "ഫീസ്", "അഞ്ഞൂറ്", "ആണ്"]
    assert stt["word_fallbacks"] == 0 and stt["word_text_mismatch"] == 0
