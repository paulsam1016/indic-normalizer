from indic_normalizer import digits
from indic_normalizer.digits import hi


def texts(core, **kw):
    return [c.text for c in digits.candidates(core, lang="hi", **kw)]


def test_english_words_in_devanagari():
    assert hi.en_words(0) == "ज़ीरो"
    assert hi.en_words(13) == "थर्टीन"
    assert hi.en_words(25) == "ट्वेंटी फ़ाइव"
    assert hi.en_words(428) == "फ़ोर हंड्रेड ट्वेंटी एट"
    assert hi.en_words(175000) == "वन लाख सेवेंटी फ़ाइव थाउज़ेंड"
    assert hi.en_words(20_000_000) == "टू करोड़"


def test_integer_candidates():
    t = texts("428")
    assert t[0] == "फ़ोर हंड्रेड ट्वेंटी एट"
    assert "फ़ोर ट्वेंटी एट" in t and "फ़ोर टू एट" in t
    assert texts("100")[:2] == ["वन हंड्रेड", "हंड्रेड"]
    assert "डबल फ़ाइव" in texts("55")
    assert texts("1,75,000")[0] == "वन लाख सेवेंटी फ़ाइव थाउज़ेंड"
    assert texts("405")[1] == "फ़ोर ओ फ़ाइव"


def test_other_classes():
    assert texts("2.5")[0] == "टू पॉइंट फ़ाइव"
    assert texts("1998")[0] == "नाइनटीन नाइनटी एट"
    assert "थाउज़ेंड सिक्स हंड्रेड" in texts("1600")
    assert texts("₹500")[0] == "फ़ाइव हंड्रेड रुपीज़" and "फ़ाइव हंड्रेड" in texts("₹500")
    assert texts("12%")[0] == "ट्वेल्व परसेंट" and "वन टू परसेंट" not in texts("12%")
    assert texts("0123") == ["ज़ीरो वन टू थ्री"]
    assert all(not any(ch.isdigit() for ch in x) for x in texts("9876543210"))


def test_verbalize_hindi_sentence():
    assert digits.verbalize("मेरे पास 25 रुपये हैं", lang="hi") == "मेरे पास ट्वेंटी फ़ाइव रुपये हैं"


def test_devanagari_is_not_counted_as_english():
    from indic_normalizer.digits.latin import is_english
    assert not is_english("मेरे पास 25 OK रुपये हैं")


def test_verifier_candidate_generator_hindi(tmp_path):
    from indic_normalizer.verifier.server import api_candidates
    st = {"run_dir": str(tmp_path)}
    r = api_candidates(st, "428 रुपये")
    assert r["lang"] == "hi" and r["groups"][0]["candidates"][0]["text"] == "फ़ोर हंड्रेड ट्वेंटी एट"
    assert r["groups"][0]["candidates"][0]["phrase"] == "फ़ोर हंड्रेड ट्वेंटी एट रुपये"
    r = api_candidates(st, "428", lang="hi")
    assert r["lang"] == "hi"
    assert api_candidates(st, "428")["lang"] == "ml"
