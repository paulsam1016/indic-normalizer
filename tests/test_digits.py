import re

import pytest

from indic_normalizer import digits
from indic_normalizer.digits import ml


def first(core, style="contracted"):
    return digits.candidates(core, style=style)[0].text


@pytest.mark.parametrize("n,contracted,glide", [
    (21, "ഇരുപത്തൊന്ന്", "ഇരുപത്തിയൊന്ന്"),
    (25, "ഇരുപത്തഞ്ച്", "ഇരുപത്തിയഞ്ച്"),
    (27, "ഇരുപത്തേഴ്", "ഇരുപത്തിയേഴ്"),
    (22, "ഇരുപത്തിരണ്ട്", "ഇരുപത്തിരണ്ട്"),
    (150, "നൂറ്റമ്പത്", "നൂറ്റിയമ്പത്"),
    (1500, "ആയിരത്തഞ്ഞൂറ്", "ആയിരത്തിയഞ്ഞൂറ്"),
    (25000, "ഇരുപത്തയ്യായിരം", "ഇരുപത്തിയയ്യായിരം"),
])
def test_compound_styles(n, contracted, glide):
    assert ml.words(n, "contracted") == contracted
    assert ml.words(n, "glide") == glide


@pytest.mark.parametrize("n,w", [
    (0, "പൂജ്യം"), (10, "പത്ത്"), (100, "നൂറ്"), (1000, "ആയിരം"), (5000, "അയ്യായിരം"),
    (10000, "പതിനായിരം"), (12000, "പന്ത്രണ്ടായിരം"), (15000, "പതിനയ്യായിരം"),
    (20000, "ഇരുപതിനായിരം"), (50000, "അമ്പതിനായിരം"), (90000, "തൊണ്ണൂറായിരം"),
    (85000, "എൺപത്തയ്യായിരം"), (100000, "ഒരു ലക്ഷം"), (200000, "രണ്ട് ലക്ഷം"),
    (175000, "ഒരു ലക്ഷത്തി എഴുപത്തയ്യായിരം"), (10_000_000, "ഒരു കോടി"),
])
def test_native_words(n, w):
    assert ml.words(n) == w


def test_english_words():
    assert ml.en_words(25) == "ട്വന്റി ഫൈവ്"
    assert ml.en_words(50000) == "ഫിഫ്റ്റി തൗസന്റ്"
    assert ml.en_words(175000) == "വൺ ലാക് സെവന്റി ഫൈവ് തൗസന്റ്"
    assert ml.en_words(2500) == "ടു തൗസന്റ് ഫൈവ് ഹൺഡ്രഡ്"


@pytest.mark.parametrize("core,cls,default", [
    ("1,75,000", "grouped", "ഒരു ലക്ഷത്തി എഴുപത്തയ്യായിരം"),
    ("2.5", "decimal", "രണ്ടര"),
    ("0.5", "decimal", "അര"),
    ("3.75", "decimal", "മൂന്നേമുക്കാൽ"),
    ("70%", "percent", "എഴുപത് ശതമാനം"),
    ("9:30", "time", "ഒമ്പതര"),
    ("6:45", "time", "ആറേമുക്കാൽ"),
    ("1:00", "time", "ഒരു മണി"),
    ("4:00-5:00", "time_range", "നാല് മണി അഞ്ച് മണി"),
    ("8-9", "range", "എട്ട് ഒമ്പത്"),
    ("17/5/1988", "date", "പതിനേഴ് അഞ്ച് ആയിരത്തിതൊള്ളായിരത്തെൺപത്തെട്ട്"),
    ("15-ാം", "ordinal", "പതിനഞ്ചാം"),
    ("21st", "ordinal", "ഇരുപത്തൊന്നാം"),
    ("23-ന്", "num_suffix", "ഇരുപത്തിമൂന്നിന്"),
    ("250യിൽ", "num_suffix", "ഇരുനൂറ്റമ്പതിൽ"),
    ("4-ൽ", "num_suffix", "നാലിൽ"),
    ("50ക്ക്", "num_suffix", "അമ്പതിന്"),
    ("9876543210", "phone", "ഒമ്പത് എട്ട് ഏഴ് ആറ് അഞ്ച് നാല് മൂന്ന് രണ്ട് ഒന്ന് പൂജ്യം"),
    ("0123", "id_code", "പൂജ്യം ഒന്ന് രണ്ട് മൂന്ന്"),
    ("₹500", "currency", "അഞ്ഞൂറ് രൂപ"),
    ("1/2", "slash", "അര"),
    ("10K", "alnum", "പത്ത് കെ"),
    ("ഡി3യും", "alnum", "ഡി മൂന്നും"),
    ("3:30-ന്", "other", "മൂന്നരയ്ക്ക്"),
    ("26-27-ഓ", "other", "ഇരുപത്താറ് ഇരുപത്തേഴോ"),
])
def test_classes_and_default_reading(core, cls, default):
    assert digits.classify(core) == cls
    assert first(core) == default


def test_candidates_include_english_and_glide():
    texts = [c.text for c in digits.candidates("25") if c.form not in ("ordinal_th", "digit_by_digit")]
    assert texts == ["ഇരുപത്തഞ്ച്", "ഇരുപത്തിയഞ്ച്", "ട്വന്റി ഫൈവ്", "ഇരുപത്തി അഞ്ച്"]
    langs = {c.lang for c in digits.candidates("50000")}
    assert langs == {"ml", "en"}
    assert "ടു തൗസന്റ് നൈന്റീൻ" in [c.text for c in digits.candidates("2019")] or \
        "ട്വന്റി നൈന്റീൻ" in [c.text for c in digits.candidates("2019")]


def test_glide_style_default():
    assert first("25", style="glide") == "ഇരുപത്തിയഞ്ച്"


def test_verbalize_keeps_punctuation_and_text():
    assert digits.verbalize("ഫീസ് 1,75,000 ആണ്, 4:00-5:00 വരെ.") == \
        "ഫീസ് ഒരു ലക്ഷത്തി എഴുപത്തയ്യായിരം ആണ്, നാല് മണി അഞ്ച് മണി വരെ."
    assert digits.verbalize("ശരി, 100.") == "ശരി, നൂറ്."
    assert digits.verbalize("no numbers here") == "no numbers here"
    assert digits.verbalize("") == ""


def test_verbalize_malayalam_digits():
    assert digits.verbalize("൨൪ രൂപ") == "ഇരുപത്തിനാല് രൂപ"


def test_operator_between_numbers_only():
    assert digits.verbalize("2950 × 18%") == "രണ്ടായിരത്തിതൊള്ളായിരത്തമ്പത് ഇന്റു പതിനെട്ട് ശതമാനം"
    assert digits.find("a × b") == []


def test_chooser_receives_span_and_candidates():
    seen = []

    def pick_english(span, cands):
        seen.append((span.token_index, span.cls))
        return next(c for c in cands if c.lang == "en")

    out = digits.verbalize("ഒരു 25 രൂപ, 4 പേര്", chooser=pick_english)
    assert out == "ഒരു ട്വന്റി ഫൈവ് രൂപ, ഫോർ പേര്"
    assert seen == [(1, "integer"), (3, "integer")]
    assert digits.verbalize("25", chooser=lambda s, c: 1) == "ഇരുപത്തിയഞ്ച്"
    assert digits.verbalize("25", chooser=lambda s, c: "XX") == "XX"
    assert digits.verbalize("25", chooser=lambda s, c: None) == "ഇരുപത്തഞ്ച്"


def test_find_spans():
    spans = digits.find("ആ, 70% ആയി (2.5)")
    assert [(s.core, s.cls, s.token_index) for s in spans] == [("70%", "percent", 1), ("2.5", "decimal", 3)]
    text = "ആ, 70% ആയി (2.5)"
    assert all(text[s.start:s.end] == s.core for s in spans)


@pytest.mark.parametrize("core", ["1,75,000", "4:00-5:00", "2.5", "70%", "23-ന്", "8-9", "17/5/1988",
                                  "071:30", "1/5-നാണ്", "50-50%", "2,65", "12th-ന്റെയും", "5G",
                                  "123456789012", "99,99,99,999", "1,00,00,00,000"])
def test_no_digits_or_number_punct_in_any_candidate(core):
    for c in digits.candidates(core):
        assert not re.search(r"[0-9:%/,.]", c.text), (core, c)
        assert "-" not in c.text, (core, c)


def test_unknown_language():
    with pytest.raises(ValueError):
        digits.verbalize("25", lang="xx")
    with pytest.raises(ValueError):
        digits.candidates("25", style="bogus")


def test_spaced_and_dropped_thousand_candidates():
    c = {(x.form, x.lang): x.text for x in digits.candidates("38500")}
    assert c[("spaced", "ml")] == "മുപ്പത്തി എണ്ണായിരത്തി അഞ്ഞൂറ്"
    assert c[("drop_thousand", "ml")] == "മുപ്പത്തെട്ട് അഞ്ഞൂറ്"
    assert c[("drop_thousand_spaced", "ml")] == "മുപ്പത്തി എട്ട് അഞ്ഞൂറ്"
    assert c[("drop_thousand", "en")] == "തേർട്ടി എയ്റ്റ് ഫൈവ് ഹൺഡ്രഡ്"
    assert first("38500") == "മുപ്പത്തെണ്ണായിരത്തഞ്ഞൂറ്"          # default unchanged
    assert "ആയിരത്തി എഴുനൂറ്" in [x.text for x in digits.candidates("1700")]
    assert not any(x.form.startswith("drop") for x in digits.candidates("1700"))   # 1 thousand: no drop
    assert not any(x.form.startswith("drop") for x in digits.candidates("5000"))   # no hundreds part


def test_english_suffix_sandhi():
    assert [x.text for x in digits.candidates("23-ന്") if x.form == "english"] == ["ട്വന്റി ത്രീക്ക്"]
    assert [x.text for x in digits.candidates("20-ൽ") if x.form == "english"] == ["ട്വന്റിയിൽ"]
    assert [x.text for x in digits.candidates("3-ന്റെ") if x.form == "english"] == ["ത്രീയുടെ"]


def test_english_ordinal_th_for_dates():
    c = lambda core: {x.form: x.text for x in digits.candidates(core)}
    assert c("20-ലോട്ടൊക്കെ")["ordinal_th"] == "ട്വന്റിത്തിലോട്ടൊക്കെ"
    assert c("27-ലോട്ട്")["ordinal_th"] == "ട്വന്റി സെവൻത്തിലോട്ട്"
    assert c("14-ലോട്ടും")["ordinal_th"] == "ഫോർട്ടീൻത്തിലോട്ടും"
    assert c("6-ലോട്ടും")["ordinal_th"] == "സിക്സ്ത്തിലോട്ടും"
    assert c("20-ലോട്ട്")["english"] == "ട്വന്റിയിലോട്ട്"
    assert c("27")["ordinal_th"] == "ട്വന്റി സെവൻത്ത്"
    assert "ordinal_th" not in c("45") and "ordinal_th" not in c("05")
    assert first("27") == "ഇരുപത്തേഴ്"                         # default unchanged
    assert [x.text for x in digits.candidates("10th") if x.lang == "en"] == ["ടെൻത്ത്"]


def test_letter_abbreviations_are_words_written_joined():
    from indic_normalizer.digits import candidates
    for core in ("3ml", "3എംഎൽ", "3-എംഎൽ"):
        assert [c.text for c in candidates(core)][:2] == ["മൂന്ന് എംഎൽ", "ത്രീ എംഎൽ"]
    assert [c.text for c in candidates("5ജിബി")][:2] == ["അഞ്ച് ജിബി", "ഫൈവ് ജിബി"]
    assert candidates("zg00")[0].text == "സെഡ്ജി പൂജ്യം പൂജ്യം"
    assert candidates("3-ഇൽ")[0].text == "മൂന്നിൽ"            # case endings still glue on
    assert candidates("10-ആണ്")[0].text == "പത്താണ്"


def test_suffix_joining_after_am_and_english_words():
    from indic_normalizer.digits import candidates
    first = lambda core: [c.text for c in candidates(core)][:2]
    assert first("200000യിൽ") == ["രണ്ട് ലക്ഷത്തിൽ", "ടു ലാകിൽ"]          # not ലക്ഷമിൽ
    assert first("250000യിൽ")[0] == "രണ്ട് ലക്ഷത്തി അമ്പതിനായിരത്തിൽ"
    assert first("10000-ത്തിലധികം") == ["പതിനായിരത്തിലധികം", "ടെൻ തൗസന്റിലധികം"]   # no doubled ത്തി
    assert first("16000ത്തോളം") == ["പതിനാറായിരത്തോളം", "സിക്സ്റ്റീൻ തൗസന്റോളം"]
    assert first("1000വും") == ["ആയിരവും", "വൺ തൗസന്റും"]
    assert first("1000-ഉം")[0] == "ആയിരവും" and first("1000-ഓ")[0] == "ആയിരമോ"
    assert first("100000-ഉടെ")[0] == "ഒരു ലക്ഷത്തിന്റെ"
    assert first("50-ത്തോളം") == ["അമ്പതോളം", "ഫിഫ്റ്റിയോളം"]
    assert first("15-ാമത്തെ")[0] == "പതിനഞ്ചാമത്തെ"
    # English words after a number are separate words
    assert "ട്വന്റി വൺ ഇയേഴ്സ്" in [c.text for c in candidates("21-ഇയേഴ്സ്")]
    assert first("3-ഹവർ") == ["മൂന്ന് ഹവർ", "ത്രീ ഹവർ"]
    assert first("3-അവറിന്")[1] == "ത്രീ അവറിന്" and first("2-വേ")[1] == "ടു വേ"


def test_new_candidates_from_v2_typed_readings():
    from indic_normalizer import digits
    texts = lambda core, **kw: [c.text for c in digits.candidates(core, **kw)][:16]   # what gets scored
    assert "പത്തായിരം" in texts("10000") and "ഒരു ലക്ഷത്തി പത്തായിരം" in texts("110000")
    assert not any("പത്തായിര" in t for t in texts("20000") + texts("50000"))   # ഇരുപതിനായിരം stays
    assert "രണ്ടായിരത്തി ഒരു നൂറ്" in texts("2100")
    assert "ത്രീ ലാക് ഫിഫ്റ്റി ഫൈവിന്" in texts("355000-ന്") and "വൺ ലാക് നൈന്റി ഫൈവിൽ" in texts("195000ൽ")
    assert "ത്രീ സീറോ" in texts("30")
    # suffix said without its ന്
    assert {"പതിനാലും", "ഫോർട്ടീൻത്തും"} <= set(texts("14-നും"))
    assert {"ആറാണ്", "സിക്സ്ത്താണ്"} <= set(texts("6-നാണ്"))
    assert {"ട്വന്റി സിക്സ്ത്താണ്", "ഇരുപത്താറല്ലേ"} <= set(texts("26-നാണ്") + texts("26-നല്ലേ"))
    assert "പതിനഞ്ചാന്തി" in texts("15-ന്") and not any("ന്തി" in t for t in texts("45-ന്"))
    assert not any(c.form.endswith("+no_n") for c in digits.candidates("23-നുള്ളിൽ"))
    # a number alone in its sentence: trailing -ഏ
    assert "എൺപത്തൊമ്പതേ" in texts("89", alone=True) and "എൺപത്തൊമ്പതേ" not in texts("89")
    assert "പൂജ്യം ഏഴേ" in texts("07", alone=True)
    assert digits.alone("89.") and digits.alone("83, 73.") and not digits.alone("ഇത് 89 ആണ്")
    assert digits.verbalize("89.") == "എൺപത്തൊമ്പത്."          # the default reading is unchanged


def test_new_candidates_from_v1_typed_readings():
    from indic_normalizer import digits
    from indic_normalizer.digits.ml import attach
    texts = lambda core, **kw: [c.text for c in digits.candidates(core, **kw)][:16]
    # a lost decimal point: before a lakh word, alone in the sentence, or before a unit
    assert "ഫോർ പോയിന്റ് എയ്റ്റ് ഫൈവ്" in texts("485", alone=True)
    assert "ഒന്നര" in texts("15", next_word="അവേഴ്സിന്") and "മൂന്നര" in texts("35", next_word="ലാക്കിൽ")
    assert "രണ്ടര" not in texts("25") and "രണ്ടര" not in texts("25", next_word="ആണ്")
    # half lakh
    assert "രണ്ടര ലക്ഷം" in texts("250000") and "രണ്ടര ലക്ഷത്തിൽ" in texts("250000യിൽ")
    assert not any("ലക്ഷം" in t and "ര " in t for t in texts("200000"))
    # "one lakh" written 1000 / 10000
    assert {"വൺ ലാക്", "ഒരു ലക്ഷം"} <= set(texts("1000")) and "വൺ ലാകിന്" in texts("10000-ന്")
    assert "വൺ ലാക്" not in texts("2000")
    # genitive written, dative said
    assert "നാൽപ്പതിന്" in texts("40-ന്റെ") and "സിക്സ്റ്റി ടുവിന്" in texts("62-ന്റെ")
    # ടു takes വ
    assert [attach("ടു", s) for s in ("ന്", "ിൽ", "ന്റെ", "ും", "ൽ")] == ["ടുവിന്", "ടുവിൽ", "ടുവിന്റെ", "ടുവും", "ടുവിൽ"]
    assert "നൈന്റി ടുവിൽ" in texts("92-ൽ") and attach("ത്രീ", "ന്") == "ത്രീക്ക്"
