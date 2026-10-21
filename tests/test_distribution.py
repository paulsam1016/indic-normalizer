import json

from indic_normalizer.digits.stats import analyse, classify_numeric, sc_group


def test_classify_numeric():
    cases = {
        "17/5/1988": "date", "4:00-5:00": "time_range", "9:30": "time", "70%": "percent",
        "8-9": "range", "1,75,000": "grouped", "2.5": "decimal", "15-ാം": "ordinal",
        "21st": "ordinal", "2019": "year", "0123": "id_code", "12345678": "id_code",
        "9876543210": "phone", "23-ന്": "num_suffix", "250യിൽ": "num_suffix",
        "5G": "alnum", "ഡി3": "alnum", "6/14": "slash", "100.": "integer", "൨൪": "integer",
    }
    for tok, cls in cases.items():
        assert classify_numeric(tok) == cls, tok


def test_sc_group():
    assert sc_group(".") == "punct" and sc_group("—") == "punct"
    assert sc_group("%") == "punct" or sc_group("%") == "symbol"
    assert sc_group("×") == "symbol" and sc_group("₹") == "symbol"
    assert sc_group("‌") == "format"
    assert sc_group("ക") is None and sc_group("്") is None and sc_group("5") is None


def test_analyse_writes_outputs(tmp_path):
    src = tmp_path / "mini"
    src.mkdir()
    doc = {
        "transcription": "ഫീസ് 1,75,000 ആണ്. ശരി.",
        "words": [
            {"word": "ഫീസ്", "start_time": 0.0, "end_time": 0.4, "speaker": "1"},
            {"word": "1,75,000", "start_time": 0.4, "end_time": 1.4, "speaker": "1"},
            {"word": "ആണ്.", "start_time": 1.4, "end_time": 1.7, "speaker": "1"},
            {"word": "ശരി.", "start_time": 2.0, "end_time": 2.3, "speaker": "2"},
        ],
        "sentences": [
            {"text": "ഫീസ് 1,75,000 ആണ്.", "start_time": 0.0, "end_time": 1.7, "speaker": "1"},
            {"text": "ശരി.", "start_time": 2.0, "end_time": 2.3, "speaker": "2"},
        ],
    }
    (src / "a.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (src / "b.json").write_text(json.dumps({"words": [], "sentences": [
        {"text": "ഇല്ല.", "start_time": 0.0, "end_time": 0.5, "speaker": "1"}]}), encoding="utf-8")

    out, m = analyse(str(src), str(tmp_path / "outputs"))
    assert out.endswith("mini")
    d = m["digits"]
    assert d["files_with_digits"] == 1 and d["files_with_digits_pct"] == 50.0
    assert d["words_with_digits"] == 1 and d["sentences_with_digits"] == 1
    assert d["word_duration_s_with_digits"]["sum"] == 1.0
    assert d["sentence_duration_s_with_digits"]["sum"] == 1.7
    assert m["numeric_classes"]["grouped"]["count"] == 1
    for f in ("metrics.json", "metrics.html", "files.tsv", "sentences.tsv",
              "numeric_expressions.tsv", "sc_chars.tsv", "digits/a.tsv", "sc/a.tsv"):
        assert (tmp_path / "outputs" / "mini" / f).exists(), f
    assert not (tmp_path / "outputs" / "mini" / "digits" / "b.tsv").exists()
