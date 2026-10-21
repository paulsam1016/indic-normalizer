import json

from indic_normalizer.digits.latin_stats import analyse, latin_part, token_kind, token_shape


def test_token_helpers():
    assert token_kind("e-mail") == "english" and token_kind("A&B") == "english"
    assert token_kind("ATM") == "english" and token_kind("ATM-ൽ") == "mixed" and token_kind("5G") == "alnum"
    assert [token_shape(t) for t in ("ATM", "Kerala", "ok", "iPhone", "A")] == \
        ["acronym", "title", "lower", "camel", "letter"]
    assert latin_part("Wi-Fiയിൽ") == "Wi-Fi" and latin_part("ATM-ൽ") == "ATM"


def test_analyse_writes_outputs(tmp_path):
    d = tmp_path / "mini"
    d.mkdir()
    doc = {"sentences": [{"text": "ഇന്ന് ATM-ൽ പോയി.", "start_time": 0, "end_time": 2, "speaker": "s1"},
                         {"text": "ശരി OK", "start_time": 2, "end_time": 3, "speaker": "s2"},
                         {"text": "ഒന്നും ഇല്ല", "start_time": 3, "end_time": 4, "speaker": "s1"}],
           "words": [{"word": "ATM-ൽ", "start_time": 0.5, "end_time": 1.0}, {"word": "ശരി", "start_time": 2, "end_time": 2.4}]}
    (d / "a.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (d / "b.json").write_text(json.dumps({"transcription": "ഒന്നും ഇല്ല"}, ensure_ascii=False), encoding="utf-8")
    out, m = analyse(str(d), str(tmp_path / "outputs"))
    l = m["latin"]
    assert l["files_with_latin"] == 1 and l["sentences_with_latin"] == 2 and l["tokens_with_latin"] == 2
    assert l["latin_chars"] == 5 and m["kinds"]["mixed"]["tokens"] == 1 and m["shapes"]["acronym"]["tokens"] == 2
    assert {w["word"] for w in m["top_words_casefolded"]} == {"atm", "ok"}
    for f in ("latin_metrics.json", "latin_metrics.html", "latin_chars.tsv", "latin_words.tsv",
              "latin_tokens.tsv", "files.tsv"):
        assert (tmp_path / "outputs" / "mini" / "latin" / f).exists(), f
