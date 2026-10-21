"""Smoke + unit tests for the corpus analyzer (indic_normalizer.analysis).

These build a tiny plain-text fixture file in a temp dir (so they don't depend on the
real data/ folder) and check the file loader, the Item detectors, the guardrail's
virama exclusion, and the HTML renderer.
"""

import pytest

from indic_normalizer import analysis as A


@pytest.fixture
def fixture_file(tmp_path):
    """A minimal plain-text corpus: one transcript per line, incl. an E12 conjunct,
    an English number run, and a digit run."""
    p = tmp_path / "corpus.txt"
    p.write_text(
        "വിവേകാനന്ദൻ്റെ\n"  # E12: ൻ്റ
        "ട്വൻറ്റി ഫോർ\n"                     # E9: English number words
        "24\n"                                                                                       # E8: digit run
        "ഹലോ, കിട്ടി.\n"                               # E5: punctuation
        "നാല്\n",                                                                # C1: word-final virama (samvruthokaram)
        encoding="utf-8",
    )
    return str(p)


def test_load_lines(fixture_file):
    lines = A.load_lines(fixture_file)
    assert len(lines) == 5                       # five non-empty lines


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        A.load_lines(str(tmp_path / "does_not_exist.txt"))


def test_load_lines_coerces_legacy_json(tmp_path):
    # an older JSON-per-line file still yields its text
    p = tmp_path / "legacy.jsonl"
    p.write_text('{"text": "അ"}\n{"transcription": "ആ"}\n', encoding="utf-8")
    lines = A.load_lines(str(p))
    assert lines == ["അ", "ആ"]


def test_analyze_structure(fixture_file):
    r = A.analyze(fixture_file)
    assert r.n_lines == 5
    assert r.label == "corpus.txt"
    codes = {it.code for it in r.items}
    assert {"E1", "E5", "E8", "E9", "E12", "C1", "C2", "W1"} <= codes
    # E12 (chillu conjunct) fires on the fixture's first line
    e12 = next(it for it in r.items if it.code == "E12")
    assert e12.lines_hit >= 1
    # E8 (digit verbalization, eval mode) fires on the "24" line
    e8 = next(it for it in r.items if it.code == "E8")
    assert e8.lines_hit >= 1


def test_handled_examples_show_before_after(fixture_file):
    r = A.analyze(fixture_file)
    e12 = next(it for it in r.items if it.code == "E12")
    # the affected word and its fix, as "before → after"
    assert e12.examples, "E12 should have an example"
    assert " → " in e12.examples[0]
    before, after = e12.examples[0].split(" → ", 1)
    assert "ൻ" in before          # mis-encoded chillu ൻ in the before form
    assert before != after            # it actually changed


def test_gap_examples_show_word_and_marker(fixture_file):
    r = A.analyze(fixture_file)
    c1 = next(it for it in r.items if it.code == "C1")
    assert c1.examples, "C1 should have an example"
    # gap items show "word (marker)" — a marker in parens, no arrow
    assert "(" in c1.examples[0] and ")" in c1.examples[0]
    assert " → " not in c1.examples[0]


def test_hit_dataclass_as_example():
    assert A.Hit("24", "ഇ").as_example() == "24 → ഇ"
    assert A.Hit("നാല്", None, "final-vowel").as_example() == \
        "നാല് (final-vowel)"


def test_limit(fixture_file):
    r = A.analyze(fixture_file, limit=2)
    assert r.n_lines == 2


def test_handled_items_sorted_before_gap(fixture_file):
    r = A.analyze(fixture_file)
    kinds = [it.kind for it in r.items]
    assert kinds == sorted(kinds, key=lambda k: k != "handled")


def test_guardrail_excludes_virama():
    # A word whose chillu canonicalizes loses a virama but NO true vowel sign — the
    # guardrail must NOT flag it as vowel loss.
    from indic_normalizer import get_normalizer
    norm, _ = get_normalizer("ml")
    raw = "അവൻ"          # അവൻ
    flags = A._guardrail_flags(raw, norm(raw))
    assert "vowel_loss" not in flags


def test_render_html(fixture_file, tmp_path):
    r = A.analyze(fixture_file)
    out = tmp_path / "report.html"
    A.render_html(r, str(out))
    html = out.read_text(encoding="utf-8")
    assert "<title>" in html and "Normalization analysis" in html
    assert "corpus.txt" in html                  # file label shown
    assert "{{" not in html                      # no leftover format-brace artifacts
    assert "Handled Items" in html and "Gap Items" in html


# --- compare mode --------------------------------------------------------------------

@pytest.fixture
def ref_hyp(tmp_path):
    """Two aligned files: line 0 differs only by an E12 chillu mis-encoding (recoverable
    by normalization); line 1 is a genuine word difference (not recoverable)."""
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("വിവേകാനന്ദന്റെ\n"  # ന്റ (correct)
                   "ഒന്ന്\n",                                                        # ഒന്ന്
                   encoding="utf-8")
    hyp.write_text("വിവേകാനന്ദൻ്റെ\n"  # ൻ്റ (mis-encoded, E12)
                   "രണ്ട്\n",                                                        # രണ്ട് (genuinely different)
                   encoding="utf-8")
    return str(ref), str(hyp)


def test_edit_distance():
    assert A._lev(list("kitten"), list("sitting")) == 3
    assert A._lev(["a", "b", "c"], ["a", "c"]) == 1        # one deletion
    assert A._lev([], ["a", "b"]) == 2                     # two insertions
    assert A._lev(["x"], ["x"]) == 0


def test_compare_wer_cer_drops_after_norm(ref_hyp):
    # line 0 differs only by an E12 chillu mis-encoding -> normalization makes it match,
    # so WER/CER must be LOWER after normalization; line 1 is a real word substitution.
    ref, hyp = ref_hyp
    r = A.compare(ref, hyp)
    assert r.n_pairs == 2
    assert r.length_mismatch is None
    assert r.metrics_norm["wer"] < r.metrics_raw["wer"]    # normalization recovered line 0
    assert r.metrics_norm["cer"] <= r.metrics_raw["cer"]
    assert 0.0 < r.metrics_norm["wer"] <= 100.0            # line 1 still an error
    # metrics dict carries wer and cer — but no 'combined'
    m = r.metrics_norm
    assert {"wer", "cer"} <= set(m)
    assert "combined" not in m


def test_compare_identical_files_zero_error(fixture_file):
    r = A.compare(fixture_file, fixture_file)
    assert r.metrics_raw["wer"] == 0.0 and r.metrics_raw["cer"] == 0.0
    assert r.metrics_norm["wer"] == 0.0 and r.metrics_norm["cer"] == 0.0


def test_compare_keeps_blank_alignment(tmp_path):
    # Same line count, but one file has blank lines (empty hypothesis). Blanks kept so
    # line N still pairs with line N — no false length mismatch, and the non-blank lines
    # still align (so the identical ones contribute zero error).
    ref = tmp_path / "ref.txt"; hyp = tmp_path / "hyp.txt"
    ref.write_text("അ\nആ\nഇ\nഈ\nഉ\n", encoding="utf-8")
    hyp.write_text("അ\n\nഇ\n\nഉ\n", encoding="utf-8")   # blanks at lines 1, 3
    r = A.compare(str(ref), str(hyp))
    assert r.length_mismatch is None              # same line count, not a mismatch
    assert r.n_pairs == 5
    # 3 of 5 words correct (lines 0,2,4), 2 deletions (blank lines) -> WER 40%
    assert abs(r.metrics_raw["wer"] - 40.0) < 0.01


def test_compare_both_blank_skipped(tmp_path):
    ref = tmp_path / "r.txt"; hyp = tmp_path / "h.txt"
    ref.write_text("അ\n\nഇ\n", encoding="utf-8")
    hyp.write_text("അ\n\nഇ\n", encoding="utf-8")
    r = A.compare(str(ref), str(hyp))
    assert r.blank_pairs == 1                      # the both-blank line skipped
    assert r.n_pairs == 2                          # only the two content pairs scored
    assert r.metrics_raw["wer"] == 0.0


def test_load_lines_keep_blanks(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("അ\n\nഇ\n", encoding="utf-8")
    assert len(A.load_lines(str(p))) == 2                    # default drops the blank
    assert len(A.load_lines(str(p), keep_blanks=True)) == 3  # kept for alignment


def test_compare_length_mismatch(tmp_path):
    a = tmp_path / "a.txt"; b = tmp_path / "b.txt"
    a.write_text("അ\nആ\n", encoding="utf-8")
    b.write_text("അ\nആ\nഇ\n", encoding="utf-8")
    r = A.compare(str(a), str(b))
    assert r.n_pairs == 2                          # min length scored
    assert r.length_mismatch == (2, 3)


def test_render_compare_html(ref_hyp, tmp_path):
    ref, hyp = ref_hyp
    r = A.compare(ref, hyp)
    out = tmp_path / "compare.html"
    A.render_compare_html(r, str(out))
    html = out.read_text(encoding="utf-8")
    assert "<title>" in html and "WER / CER compare" in html
    assert "ref.txt" in html and "hyp.txt" in html
    assert "{{" not in html
    assert "WER" in html and "CER" in html
    # normalization breakdown tables present, WER/CER sub/del/ins breakdown tables gone
    assert "Handled Items" in html and "Gap Items" in html
    assert "WER breakdown" not in html and "CER breakdown" not in html


def test_compare_has_item_breakdown(ref_hyp):
    ref, hyp = ref_hyp
    r = A.compare(ref, hyp)
    # breakdown runs over both files combined (2 ref + 2 hyp lines)
    assert r.n_lines == 4
    codes = {it.code for it in r.items}
    assert {"E12", "C1", "C2"} <= codes                # handled + gap Items detected
    e12 = next(it for it in r.items if it.code == "E12")
    assert e12.lines_hit >= 1                           # the ൻ്റ line fires on the hyp side


def test_compare_breakdown_split_by_outcome(tmp_path):
    # pair 0: differs only by E12 chillu -> recovered; pair 1: real word sub -> still
    # different; pair 2: identical -> already equal.
    ref = tmp_path / "ref.txt"; hyp = tmp_path / "hyp.txt"
    ref.write_text("വിവേകാനന്ദന്റെ\n"  # ന്റ
                   "ഒന്ന്\n"                                                        # ഒന്ന്
                   "ഹലോ\n", encoding="utf-8")                                                # ഹലോ
    hyp.write_text("വിവേകാനന്ദൻ്റെ\n"  # ൻ്റ (E12)
                   "രണ്ട്\n"                                                        # രണ്ട് (differs)
                   "ഹലോ\n", encoding="utf-8")                                                # ഹലോ (identical)
    r = A.compare(str(ref), str(hyp))
    assert (r.pairs_recovered, r.pairs_still, r.pairs_equal) == (1, 1, 1)
    # Recovered bucket is CAUSAL: E12 credited as the recovering Item, and only E12
    rec_codes = {it.code for it in r.recovered_attr}
    assert rec_codes == {"E12"}
    assert r.recovery_unattributed == 0
    # E12 does NOT appear in the still-different breakdown (that pair is a real word sub)
    still_codes = {it.code for it in r.items_still if it.lines_hit}
    assert "E12" not in still_codes


def test_compare_recovery_excludes_whitespace_lowercase(tmp_path):
    # a punctuation-only difference is recovered by E5; E11 (whitespace) is jointly
    # required but must NOT be credited as a cause.
    ref = tmp_path / "ref.txt"; hyp = tmp_path / "hyp.txt"
    ref.write_text("ഹലോ കിട്ടി\n", encoding="utf-8")   # ഹലോ കിട്ടി
    hyp.write_text("ഹലോ, കിട്ടി.\n", encoding="utf-8")  # ഹലോ, കിട്ടി.
    r = A.compare(str(ref), str(hyp))
    assert r.pairs_recovered == 1
    codes = {it.code for it in r.recovered_attr}
    assert "E5" in codes                 # punctuation removal did it
    assert "E11" not in codes and "E10" not in codes
