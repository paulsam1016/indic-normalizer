"""Signs before / after a number: kept by train mode for the digits module (strip_number_signs
removes them), read by the digits module, and the train-mode count of clips with digits."""

import pytest

from indic_normalizer import digits, normalize
from indic_normalizer.analysis import analyze, digits_left_message


def train(text, lang="ml", **kw):
    return normalize(text, lang=lang, mode="train", **kw)


# --- train keeps the signs (default) ------------------------------------------------------------
@pytest.mark.parametrize("raw, out", [
    ("ഫീസ് ₹500 ആണ്.", "ഫീസ് ₹500 ആണ്"),
    ("ഫീസ് Rs.500 ആണ്", "ഫീസ് rs.500 ആണ്"),
    ("ഫീസ് 500/- ആണ്", "ഫീസ് 500/- ആണ്"),
    ("ഫീസ് 500₹, ആണ്", "ഫീസ് 500₹ ആണ്"),
    ("ഞാൻ $24, €10, £5, ¥100 കൊടുത്തു", "ഞാൻ $24 €10 £5 ¥100 കൊടുത്തു"),
    ("ഞാൻ $ 24 കൊടുത്തു", "ഞാൻ $ 24 കൊടുത്തു"),
    ("20% കിഴിവ്, 2.5 % പലിശ", "20% കിഴിവ് 2.5% പലിശ"),        # 2.5 % glued for the reader
    ("റൂം #5 ആണ്", "റൂം #5 ആണ്"),
    ("നമ്പർ +91 98765 ആണ്", "നമ്പർ +91 98765 ആണ്"),
    ("താപനില -5 ആണ്, 5° ആണ്", "താപനില -5 ആണ് 5° ആണ്"),
    ("ഏകദേശം ~50 പേർ", "ഏകദേശം ~50 പേർ"),
])
def test_train_keeps_signs(raw, out):
    assert train(raw) == out


def test_train_spaced_dash_is_still_punctuation():
    assert train("4:00 - 5:00 വരെ") == "4:00 5:00 വരെ"
    assert train("വന്നു - പോയി") == "വന്നു പോയി"


def test_train_inside_marks_unchanged():
    assert train("4:00-5:00, 8-9, 23-ന്, 1,75,000.") == "4:00-5:00 8-9 23-ന് 1,75,000"


@pytest.mark.parametrize("lang, raw, out", [
    ("en", "I paid $2,000,000, ₹500 and 20% at 4:30, #5.", "i paid $2,000,000 ₹500 and 20% at 4:30 #5"),
    (None, "hello $24, 20%, 2.5", "hello $24 20% 2.5"),
])
def test_train_keeps_signs_english_and_universal(lang, raw, out):
    assert train(raw, lang=lang) == out


# --- strip_number_signs=True removes them -------------------------------------------------------
@pytest.mark.parametrize("lang, raw, out", [
    ("ml", "ഫീസ് ₹500, Rs.500, Rs 500, 500₹, 500/-, Rs.500/- ആണ്.", "ഫീസ് 500 500 500 500 500 500 ആണ്"),
    ("ml", "ഞാൻ $24, $ 24, €10, £5, ¥100 കൊടുത്തു", "ഞാൻ 24 24 10 5 100 കൊടുത്തു"),
    ("ml", "20%, 2.5 %, #5, +91, -5, 5°, ~50.", "20 2.5 5 91 5 5 50"),
    ("ml", "4:00-5:00, 2.5, 1,75,000, 23-ന്", "4:00-5:00 2.5 1,75,000 23-ന്"),   # inside kept
    ("en", "I paid $24 and 20% at 4:30.", "i paid 24 and 20 at 4:30"),
    (None, "hello $24, 20%", "hello 24 20"),
])
def test_strip_number_signs(lang, raw, out):
    assert train(raw, lang=lang, strip_number_signs=True) == out


def test_strip_number_signs_only_affects_train():
    t = "ഫീസ് ₹500 ആണ്"
    assert normalize(t, lang="ml", strip_number_signs=True) == normalize(t, lang="ml")
    assert normalize(t, lang="ml", mode="inference", strip_number_signs=True) == t


# --- the digits module reads them ---------------------------------------------------------------
@pytest.mark.parametrize("raw, out", [
    ("ഫീസ് ₹500 ആണ്", "ഫീസ് അഞ്ഞൂറ് രൂപ ആണ്"),
    ("ഫീസ് rs.500 ആണ്", "ഫീസ് അഞ്ഞൂറ് രൂപ ആണ്"),
    ("ഫീസ് rs 500 ആണ്", "ഫീസ് അഞ്ഞൂറ് രൂപ ആണ്"),
    ("ഫീസ് 500₹ ആണ്", "ഫീസ് അഞ്ഞൂറ് രൂപ ആണ്"),
    ("ഫീസ് 500/- ആണ്", "ഫീസ് അഞ്ഞൂറ് രൂപ ആണ്"),
    ("ഫീസ് rs.500/- ആണ്", "ഫീസ് അഞ്ഞൂറ് രൂപ ആണ്"),
    ("ഫീസ് ₹70,000 ആണ്", "ഫീസ് എഴുപതിനായിരം രൂപ ആണ്"),
    ("ഞാൻ $24 കൊടുത്തു", "ഞാൻ ഇരുപത്തിനാല് ഡോളർ കൊടുത്തു"),
    ("ഞാൻ $ 24 കൊടുത്തു", "ഞാൻ ഇരുപത്തിനാല് ഡോളർ കൊടുത്തു"),
    ("ഞാൻ €10 കൊടുത്തു", "ഞാൻ പത്ത് യൂറോ കൊടുത്തു"),
    ("ഞാൻ £5 കൊടുത്തു", "ഞാൻ അഞ്ച് പൗണ്ട് കൊടുത്തു"),
    ("20% കിഴിവ്", "ഇരുപത് ശതമാനം കിഴിവ്"),
    ("2.5 % പലിശ", "രണ്ടര ശതമാനം പലിശ"),
    ("റൂം #5 ആണ്", "റൂം നമ്പർ അഞ്ച് ആണ്"),
    ("താപനില 5° ആണ്", "താപനില അഞ്ച് ഡിഗ്രി ആണ്"),
    ("താപനില -5 ആണ്", "താപനില മൈനസ് അഞ്ച് ആണ്"),
])
def test_digits_reads_signs(raw, out):
    assert digits.verbalize(raw) == out


def test_digits_sign_variants_are_candidates():
    texts = [c.text for c in digits.candidates("₹500")]
    assert texts[:4] == ["അഞ്ഞൂറ് രൂപ", "ഫൈവ് ഹൺഡ്രഡ് റുപ്പീസ്", "അഞ്ഞൂറ്", "ഫൈവ് ഹൺഡ്രഡ്"]
    assert "ട്വന്റി ഫോർ ഡോളേഴ്സ്" in [c.text for c in digits.candidates("$24")]
    assert "ഇരുപത്തിനാല്" in [c.text for c in digits.candidates("$24")]        # sign not said
    assert "അഞ്ച്" in [c.text for c in digits.candidates("#5")]
    for t in ("#5", "5°", "-5"):                                              # no "fifth"
        assert not any("ഫിഫ്ത്ത്" in c.text for c in digits.candidates(t))


@pytest.mark.parametrize("raw, out", [
    ("I paid $24.", "I paid twenty four dollars."),
    ("I paid $1.", "I paid one dollar."),
    ("It costs ₹500/- only.", "It costs five hundred rupees only."),
    ("Room #5 is 20% off at 5°C.", "Room number five is twenty percent off at five degrees celsius."),
    ("Pay € 10 now.", "Pay ten euros now."),
])
def test_digits_reads_signs_english(raw, out):
    assert digits.verbalize(raw) == out


def test_train_then_digits_round_trip():
    t = train("ഫീസ് ₹500, $24, 20%, #5, 500/- ആണ്.")
    assert digits.verbalize(t) == ("ഫീസ് അഞ്ഞൂറ് രൂപ ഇരുപത്തിനാല് ഡോളർ ഇരുപത് ശതമാനം "
                                   "നമ്പർ അഞ്ച് അഞ്ഞൂറ് രൂപ ആണ്")


# --- train mode: clips with digits counted ------------------------------------------------------
def test_analyze_counts_clips_with_digits(tmp_path):
    f = tmp_path / "clips.txt"
    f.write_text("ഫീസ് ₹500 ആണ്.\nഅവൻ വന്നു.\nI paid $24.\n", encoding="utf-8")
    r = analyze(str(f), norm_mode="train")
    assert (r.lines_with_digits, r.n_lines) == (2, 3)
    assert analyze(str(f), norm_mode="eval").lines_with_digits == 0    # eval spells them out


def test_digits_left_message():
    msg = digits_left_message(2, 3)
    assert "Clips with digits: 2 of 3" in msg and "indic-digits verbalize" in msg
    assert "no numbers left" in digits_left_message(0, 3)
