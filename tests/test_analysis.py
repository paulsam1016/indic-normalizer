"""Smoke + unit tests for the corpus analyzer (indic_normalizer.analysis).

These build a tiny fixture corpus in a temp dir (so they don't depend on the real
data/ folder) and check the loader selector, the Item detectors, the guardrail's
virama exclusion, and the HTML renderer.
"""

import os

import pytest

from indic_normalizer import analysis as A


@pytest.fixture
def fixture_root(tmp_path):
    """A minimal data/<lang> tree with both a dataset.txt and a calls/*.txt file."""
    ml = tmp_path / "malayalam"
    (ml / "calls").mkdir(parents=True)
    # dataset.txt — one transcript per line; includes an E12 conjunct and an English number
    (ml / "dataset.txt").write_text(
        "വിവേകാനന്ദൻ്റെ\n"  # E12: ൻ്റ
        "ട്വൻറ്റി ഫോർ\n"                     # E9: English number words
        "24\n",                                                                                       # E8: digit run
        encoding="utf-8",
    )
    (ml / "calls" / "c1.txt").write_text(
        "ഹലോ, കിട്ടി. 100 വെക്കാം.",
        encoding="utf-8",
    )
    return str(tmp_path)


def test_load_corpus_selector(fixture_root):
    ds = A.load_corpus("dataset", data_root=fixture_root)
    assert len(ds) == 3                      # three non-empty lines
    calls = A.load_corpus("calls", data_root=fixture_root)
    assert len(calls) == 1                   # one call file


def test_bad_source_rejected(fixture_root):
    with pytest.raises(ValueError):
        A.load_corpus("nope", data_root=fixture_root)


def test_analyze_structure(fixture_root):
    r = A.analyze("dataset", data_root=fixture_root)
    assert r.n_lines == 3
    codes = {it.code for it in r.items}
    # all handled + gap items present
    assert {"E1", "E5", "E8", "E9", "E12", "C1", "C2", "W1"} <= codes
    # E12 (chillu conjunct) must fire on the fixture's first line
    e12 = next(it for it in r.items if it.code == "E12")
    assert e12.lines_hit >= 1
    # E8 (digit verbalization) fires on the "24" line
    e8 = next(it for it in r.items if it.code == "E8")
    assert e8.lines_hit >= 1


def test_handled_items_sorted_before_gap(fixture_root):
    r = A.analyze("calls", data_root=fixture_root)
    kinds = [it.kind for it in r.items]
    # every "handled" comes before every "gap"
    assert kinds == sorted(kinds, key=lambda k: k != "handled")


def test_guardrail_excludes_virama():
    # A word whose consonant+virama+ZWJ chillu canonicalizes to atomic chillu loses a
    # virama but NO true vowel sign — the guardrail must NOT flag it as vowel loss.
    raw = "അവൻ"          # അവൻ (already atomic) — sanity: no flag
    from indic_normalizer import get_normalizer
    norm, _ = get_normalizer("ml")
    flags = A._guardrail_flags(raw, norm(raw))
    assert "vowel_loss" not in flags


def test_render_html(fixture_root, tmp_path):
    r = A.analyze("dataset", data_root=fixture_root)
    out = tmp_path / "report.html"
    A.render_html(r, str(out))
    html = out.read_text(encoding="utf-8")
    assert "<title>" in html and "Normalization analysis" in html
    assert "{{" not in html                  # no leftover format-brace artifacts
    assert "Handled Items" in html and "Gap Items" in html
