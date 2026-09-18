"""Corpus analyzer for the normalization Items — find, quantify, and prioritize.

Runs any plain-text corpus (one transcript per line) through the pipeline and reports,
per normalization Item:

  * **firing frequency** — how often each *handled* rule (E1-E12) actually changes
    text on this corpus, so the banked wins are evidence-backed rather than assumed;
  * **gap detection** — how often each *unhandled* pattern from NORMALIZATION.md
    (W1/W3/W4 segmentation, C1/C2/C5/C6 residue) appears, with real examples, so the
    backlog is prioritized by measured prevalence;
  * **before/after deltas** — characters and tokens the pipeline adds or removes, split
    into a WER-side (token-count) and CER-side (character) view;
  * **guardrail** — a corpus-wide vanishing-vowels tripwire (ACL-2024 identity check):
    any line where normalization drops a vowel sign / virama or splits a word at one.

The input is a single plain-text file, one transcript (or segment) per line — the shape
every corpus file now has (dataset.txt, dataset_hyp.txt, call_ref.txt, call_hyp.txt, …),
so there is no dataset/calls distinction to make. Legacy JSON lines (``{"text": ...}`` /
``{"transcription": ...}``) are still coerced to their text, so an older file also works.

Nothing here mutates the corpus; detectors are read-only probes. The heavy detectors
reuse the library's own primitives (``base``, the Malayalam module's regexes) so the
report tracks the real pipeline, not a re-implementation.

Compare mode (two files) line-aligns a reference and a hypothesis and reports corpus
WER and CER — both on the raw text and after normalizing both sides, so the drop shows
how much normalization recovers. WER/CER are computed with jiwer (stdlib edit-distance
fallback if absent), matching AetherTranscribe's eval_metrics.score_text exactly.

Usage
-----
    from indic_normalizer.analysis import analyze, render_html, compare, render_compare_html
    report = analyze("data/malayalam/call_ref.txt", lang="ml")
    render_html(report, "report.html")

    cmp = compare("data/malayalam/dataset_ref.txt", "data/malayalam/dataset_hyp.txt")
    render_compare_html(cmp, "compare.html")

CLI
---
    python -m indic_normalizer.analysis call_ref.txt --out report.html          # analyze one
    python -m indic_normalizer.analysis dataset_ref.txt dataset_hyp.txt --out compare.html
"""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import base, get_normalizer
from .languages import malayalam as _ml

# --------------------------------------------------------------------------------------
# Corpus loading — a single plain-text file, one transcript per line
# --------------------------------------------------------------------------------------


def _coerce_text(raw: str) -> str:
    """Return the transcript text of one line.

    Lines are plain text now, so this is usually a strip(). A legacy JSON line
    (``{"text": ...}`` or ``{"transcription": ...}``) is still coerced to its text so an
    older corpus file keeps working.
    """
    s = raw.strip()
    if not s:
        return ""
    if s[0] in "{[":
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return s
        if isinstance(obj, dict):
            return str(obj.get("text") or obj.get("transcription")
                       or obj.get("hypothesis_raw") or obj.get("reference_raw") or "")
        return s
    return s


def load_lines(input_path: str, keep_blanks: bool = False) -> list[str]:
    """Load transcript strings from a plain-text file.

    By default (``keep_blanks=False``) empty lines are dropped — right for per-file
    analysis. For line-aligned COMPARE, pass ``keep_blanks=True`` so a blank line stays in
    place as ``""`` and line N of one file still pairs with line N of the other; otherwise
    a blank on one side would shift every following pair out of alignment.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input file not found: {input_path}")
    texts: list[str] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            t = _coerce_text(line)
            if t or keep_blanks:
                texts.append(t)
    return texts


# --------------------------------------------------------------------------------------
# Item detectors
# --------------------------------------------------------------------------------------
#
# Each detector answers "which words does Item X affect on this RAW line, and how does
# each change?" It returns a list of Hit objects (empty = the item does not fire). A
# HANDLED item (E*) reports each affected word together with the fixed form it becomes,
# by re-running that item's exact pipeline sub-step on the single word; a GAP item (W*/C*)
# reports the word that matches the surface pattern NORMALIZATION.md describes, with a
# short marker naming the pattern (no fix — the item is unhandled).
#
# total occurrences = len(hits); lines_hit counts lines with >= 1 hit. The Examples
# column shows each hit as "before → after" (handled) or "word (marker)" (gap).

_VIRAMA = "്"
_ZWJ = "‍"
_ZWNJ = "‌"


@dataclass
class Hit:
    """One affected word. ``after`` is the fixed form (handled items) or None (gap
    items); ``marker`` names the gap pattern for gap items."""
    before: str
    after: "str | None" = None
    marker: str = ""

    def as_example(self) -> str:
        if self.after is not None:
            return f"{self.before} → {self.after}"
        return f"{self.before} ({self.marker})" if self.marker else self.before


def _words_with_spans(t: str):
    """Yield (word, start, end) for whitespace-split tokens, preserving offsets."""
    for m in re.finditer(r"\S+", t):
        yield m.group(0), m.start(), m.end()

# --- handled-item helpers (mirror the real pipeline; report before → after) -----------

_INDIC_DIGIT = re.compile(r"[०-९൦-൯]")   # sample Devanagari + Malayalam
_ANY_DECIMAL = re.compile(r"\d")
_ASCII_DIGIT_RUN = re.compile(r"[0-9]+")
_GROUP_COMMA = _ml._GROUP_COMMA
_CHILLU_CONJ = _ml._CHILLU_CONJ
_EN_ANY = _ml._EN_ANY
_EN_STRONG = _ml._EN_STRONG
_ZW_ALL = re.compile("[" + _ZWJ + _ZWNJ + "﻿]")


def _hits_by_word(t: str, matches, fix) -> "list[Hit]":
    """Build one Hit per word that ``matches`` (word -> bool), with ``fix`` (word -> str)
    as its 'after'. Only words whose fix actually differs are reported."""
    out = []
    for w, _s, _e in _words_with_spans(t):
        if matches(w):
            after = fix(w)
            if after != w:
                out.append(Hit(w, after))
    return out


def _hits_nfc(t: str) -> "list[Hit]":
    return _hits_by_word(t, lambda w: unicodedata.normalize("NFC", w) != w,
                         lambda w: unicodedata.normalize("NFC", w))


def _hits_indicnlp(t: str) -> "list[Hit]":
    return _hits_by_word(t, lambda w: _ml._IND_NORM.normalize(w) != w,
                         lambda w: _ml._IND_NORM.normalize(w))


def _hits_unk(t: str) -> "list[Hit]":
    # the ⁇ / <unk> marker itself is the affected token; it becomes empty
    return [Hit(m.group(0), "∅") for m in base.UNK.finditer(t)]


def _hits_zero_width(t: str) -> "list[Hit]":
    return _hits_by_word(t, lambda w: bool(_ZW_ALL.search(w)),
                         lambda w: _ZW_ALL.sub("", w))


def _hits_punct(t: str) -> "list[Hit]":
    return _hits_by_word(t, lambda w: bool(base.PUNCT.search(w)),
                         lambda w: base.PUNCT.sub(" ", w).strip())


def _hits_indic_digit(t: str) -> "list[Hit]":
    return _hits_by_word(t, lambda w: bool(_INDIC_DIGIT.search(w)),
                         lambda w: base.fold_digits(w))


def _hits_group_comma(t: str) -> "list[Hit]":
    return _hits_by_word(t, lambda w: bool(_GROUP_COMMA.search(w)),
                         lambda w: _GROUP_COMMA.sub("", w))


def _hits_digit_verbalize(t: str) -> "list[Hit]":
    # words containing an ASCII digit run (after fold + comma strip) that E8 verbalizes
    def fix(w):
        w2 = _GROUP_COMMA.sub("", base.fold_digits(w))
        return _ml._NUM_RE.sub(_ml._verbalize_digits, w2)
    return _hits_by_word(
        t,
        lambda w: bool(_ASCII_DIGIT_RUN.search(_GROUP_COMMA.sub("", base.fold_digits(w)))),
        fix,
    )


def _hits_en_numbers(t: str) -> "list[Hit]":
    """Each run of English-in-Malayalam-script number words E9 converts -> one Hit
    (the whole run before → its canonical Malayalam)."""
    words = t.split()
    out, i = [], 0
    while i < len(words):
        j = i
        while j < len(words) and words[j] in _EN_ANY:
            j += 1
        if j > i and any(words[k] in _EN_STRONG for k in range(i, j)):
            run = " ".join(words[i:j])
            after = _ml._canonicalize_en_numbers(run)
            if after != run:
                out.append(Hit(run, after))
            i = j
        else:
            i = max(j, i + 1)
    return out


def _hits_lower(t: str) -> "list[Hit]":
    return _hits_by_word(t, lambda w: w.lower() != w, lambda w: w.lower())


def _hits_chillu_conj(t: str) -> "list[Hit]":
    return _hits_by_word(
        t, lambda w: bool(_CHILLU_CONJ.search(w)), _ml._fix_chillu_conjuncts)


# --- gap-item helpers (surface patterns; see NORMALIZATION.md) -------------------------

# C1 samvruthokaram / final-vowel drift: word-final chandrakkala/samvruthokaram forms.
#   ...ു്  (samvruthokaram)  |  ...്  (bare virama)  |  ...ു  (final u)  at a word end.
_C1_FINAL = re.compile(r"(?:ു്|്|ു)(?=\s|$)")
# C2 anusvara vs nasal: word-final ം  vs  ന്/മ് — count both realizations word-finally.
_C2_ANUSVARA = re.compile(r"ം(?=\s|$)")
_C2_NASAL = re.compile(r"(?:ന|മ)്(?=\s|$)")
# W3 acronym / spaced single-letter run: >=2 single Malayalam letters separated by space
#   (e.g. എ ടി എം). A "letter" here = a base consonant/vowel optionally + one matra,
#   length 1-2, standing alone.
_W3_TOKEN = re.compile(r"^[ഀ-ൿ]{1,2}$")
# W4 intra-word hyphen (covid-19 / വാക്ക്-വാക്ക്): hyphen with non-space on both sides.
_W4_HYPHEN = re.compile(r"(?<=\S)[-‐-―](?=\S)")
# C5 foreign-word transliteration variants: Latin token embedded in Malayalam text
#   (code-mixed borrowings whose spelling drifts). Count Latin alphabetic runs.
_C5_LATIN = re.compile(r"[A-Za-z]{2,}")
# C6 visual-lookalike: ASCII 0/O or Malayalam ഠ used where a digit is expected — a weak
#   proxy: a lone 'O'/'0' adjacent to Malayalam.
_C6_LOOKALIKE = re.compile(r"(?:(?<=[ഀ-ൿ])[O0]|[O0](?=[ഀ-ൿ]))")


def _hits_word_pattern(t: str, pattern, marker: str) -> "list[Hit]":
    """One Hit per word that contains ``pattern`` (a compiled regex), marked ``marker``."""
    return [Hit(w, None, marker) for w, _s, _e in _words_with_spans(t) if pattern.search(w)]


def _hits_c1(t: str) -> "list[Hit]":
    return _hits_word_pattern(t, _C1_FINAL, "final-vowel")


def _hits_c2(t: str) -> "list[Hit]":
    out = []
    for w, _s, _e in _words_with_spans(t):
        if _C2_ANUSVARA.search(w):
            out.append(Hit(w, None, "anusvara ം"))
        elif _C2_NASAL.search(w):
            out.append(Hit(w, None, "nasal ന്/മ്"))
    return out


def _hits_w4(t: str) -> "list[Hit]":
    return _hits_word_pattern(t, _W4_HYPHEN, "intra-word hyphen")


def _hits_c5(t: str) -> "list[Hit]":
    return _hits_word_pattern(t, _C5_LATIN, "Latin token")


def _hits_c6(t: str) -> "list[Hit]":
    return _hits_word_pattern(t, _C6_LOOKALIKE, "lookalike O/0")


def _hits_w3(t: str) -> "list[Hit]":
    """Spaced single-letter runs of length >= 2 (acronym-style): report the joined run."""
    words = t.split()
    out, run = [], []
    for w in words:
        if _W3_TOKEN.match(w):
            run.append(w)
        else:
            if len(run) >= 2:
                out.append(Hit(" ".join(run), None, "spaced acronym"))
            run = []
    if len(run) >= 2:
        out.append(Hit(" ".join(run), None, "spaced acronym"))
    return out


# --------------------------------------------------------------------------------------
# Item catalogue
# --------------------------------------------------------------------------------------

@dataclass
class Item:
    code: str
    kind: str          # "handled" | "gap"
    metric: str        # "WER" | "CER" | "CER+WER"
    title: str
    detect: object     # callable(raw_text) -> list[Hit]


def _hits_whitespace(t: str) -> "list[Hit]":
    """E11 is line-level (no single affected word); report a whole-line before → after
    only when collapsing actually changes the line."""
    fixed = base.SPACE.sub(" ", t).strip()
    return [Hit(_snippet(t, 36), _snippet(fixed, 36))] if fixed != t else []


def _handled_items() -> list[Item]:
    return [
        Item("E1", "handled", "CER", "Unicode NFC", _hits_nfc),
        Item("E2", "handled", "CER", "Chillu / ZWJ-ZWNJ canonicalization", _hits_indicnlp),
        Item("E3", "handled", "CER+WER", "⁇ / <unk> removal", _hits_unk),
        Item("E4", "handled", "CER", "Zero-width (ZWNJ/ZWJ/BOM) drop", _hits_zero_width),
        Item("E5", "handled", "CER+WER", "ASCII + Indic punctuation removal", _hits_punct),
        Item("E6", "handled", "CER", "Indic digit → ASCII", _hits_indic_digit),
        Item("E7", "handled", "CER+WER", "Digit-grouping commas (70,000→70000)",
             _hits_group_comma),
        Item("E8", "handled", "WER+CER", "Native digit verbalization (24→ഇരുപത്തിനാല്)",
             _hits_digit_verbalize),
        Item("E9", "handled", "WER+CER", "English↔Malayalam number canonicalization",
             _hits_en_numbers),
        Item("E10", "handled", "CER", "Lowercasing (code-mixed Latin)", _hits_lower),
        Item("E11", "handled", "WER", "Whitespace collapse + trim", _hits_whitespace),
        Item("E12", "handled", "CER", "Mis-encoded chillu conjunct (ൻ്റ→ന്റ)",
             _hits_chillu_conj),
    ]


def _gap_items() -> list[Item]:
    return [
        Item("C1", "gap", "CER", "Samvruthokaram / final-vowel drift", _hits_c1),
        Item("C2", "gap", "CER", "Anusvara vs nasal spelling (word-final)", _hits_c2),
        Item("W1", "gap", "WER", "Sandhi / postposition split-merge (candidates)",
             _hits_w1),
        Item("W3", "gap", "WER", "Acronym / spaced single-letter runs", _hits_w3),
        Item("W4", "gap", "WER", "Intra-word hyphen", _hits_w4),
        Item("C5", "gap", "CER", "Foreign-word (Latin) tokens — transliteration variants",
             _hits_c5),
        Item("C6", "gap", "CER", "Visual-lookalike letter/digit (O/0 by Malayalam)",
             _hits_c6),
    ]


# W1 postposition split candidate: a short function-morpheme token that commonly appears
# both fused and spaced. We can't know the reference form, so we report *occurrences of the
# spaced free-standing postposition*, which is the split side of the W1 split-merge pair.
_W1_POSTPOS = {
    "ലേക്ക്",  # ലേക്ക്  (to/towards)
    "ക്ക്",              # ക്ക്
    "നിന്ന്",  # നിന്ന്  (from)
    "ഓട്",                    # ഓട്     (with)
    "ക്കാൻ",        # ക്കാൻ
    "വേണ്ടി",  # വേണ്ടി  (for)
}


def _hits_w1(t: str) -> "list[Hit]":
    return [Hit(w, None, "spaced postposition") for w in t.split() if w in _W1_POSTPOS]


# --------------------------------------------------------------------------------------
# Guardrail: vanishing-vowels tripwire (ACL-2024 self-normalization identity)
# --------------------------------------------------------------------------------------

# Malayalam DEPENDENT VOWEL SIGNS (matras) only — the Mark-class chars a bad
# normalizer (Devanagari-style category stripping) would silently delete. The virama
# (U+0D4D) is deliberately EXCLUDED: it legitimately disappears during chillu
# canonicalization (E2/E12 fold consonant+virama+ZWJ into an atomic chillu), so
# counting it here would false-positive on every correctly-normalized chillu line.
_MATRA = re.compile(r"[ാ-ൌൗൢൣ]")   # dependent vowel signs only — NOT the virama (U+0D4D), which chillu canonicalization legitimately removes


def _guardrail_flags(raw: str, norm_out: str) -> dict:
    """Flag a line where normalization may have dropped vowel signs or split words at one.

    Two heuristics, matched to the ACL-2024 findings:
      * **vowel-sign loss** — the normalized text has fewer matras than the raw text by
        more than digit verbalization can explain (verbalization legitimately rewrites
        digit runs), i.e. raw has no digits but matras dropped;
      * **word-count inflation** — normalized token count exceeds raw by a wide margin on
        a digit-free line (splitting at a vowel sign inflates the count).
    Returns {} when clean.
    """
    if _ANY_DECIMAL.search(raw) or any(w in _EN_ANY for w in raw.split()):
        return {}  # digit / number lines legitimately change counts — skip
    raw_matras = len(_MATRA.findall(raw))
    out_matras = len(_MATRA.findall(norm_out))
    raw_wc = len(raw.split())
    out_wc = len(norm_out.split())
    flags = {}
    if raw_matras and out_matras < raw_matras:
        flags["vowel_loss"] = raw_matras - out_matras
    if raw_wc and out_wc > raw_wc:
        flags["word_inflation"] = out_wc - raw_wc
    return flags


# --------------------------------------------------------------------------------------
# Analysis driver
# --------------------------------------------------------------------------------------

@dataclass
class ItemResult:
    code: str
    kind: str
    metric: str
    title: str
    lines_hit: int = 0        # lines where the item fires at least once
    total_hits: int = 0       # total occurrences across the corpus
    examples: list = field(default_factory=list)


@dataclass
class Report:
    input_path: str                   # the analyzed file
    label: str                        # display name (the file's basename)
    lang: str
    n_lines: int
    n_chars: int
    n_tokens: int
    items: list                       # list[ItemResult]
    char_delta: int                   # sum(len(norm) - len(raw)), chars
    char_abs_delta: int               # sum(|per-char edit|) approximated by len change magnitude
    token_delta: int                  # sum(norm_tokens - raw_tokens)
    lines_changed: int                # lines where norm(raw) != raw
    guardrail_flags: list             # list of {line_no, snippet, flags}
    norm_mode: str = "eval"           # normalizer mode used: "eval" (verbalize) or "inference" (ITN)
    generated_at: str = ""


def _item_breakdown(texts: list, examples_per_item: int = 5) -> list:
    """Per-Item firing frequency + examples over ``texts`` -> ordered list[ItemResult]
    (handled Items first, then gap; each sorted by lines hit). Shared by the analyze and
    compare reports."""
    catalogue = _handled_items() + _gap_items()
    results = {it.code: ItemResult(it.code, it.kind, it.metric, it.title) for it in catalogue}
    for raw in texts:
        for it in catalogue:
            try:
                hits = it.detect(raw) or []
            except Exception:
                hits = []
            if hits:
                r = results[it.code]
                r.lines_hit += 1
                r.total_hits += len(hits)
                for h in hits:
                    if len(r.examples) >= examples_per_item:
                        break
                    ex = _snippet(h.as_example(), 70)
                    if ex not in r.examples:
                        r.examples.append(ex)
    return sorted(results.values(),
                  key=lambda r: (r.kind != "handled", -r.lines_hit, r.code))


def analyze(input_path: str, lang: str = "ml",
            limit: int | None = None, examples_per_item: int = 5,
            norm_mode: str = "eval", segmentation: bool = False) -> Report:
    """Analyze the plain-text corpus at ``input_path`` and return a :class:`Report`.

    ``input_path`` is a file with one transcript (or segment) per line. ``limit`` caps
    the number of lines (handy for a quick look); ``None`` = all. ``norm_mode`` is the
    normalizer mode to inspect: "eval" (default; model-facing verbalization) or "inference"
    (reader-facing ITN — numbers to digits, punctuation/case preserved).
    """
    texts = load_lines(input_path)
    if limit is not None:
        texts = texts[:limit]

    norm_fn, _ = get_normalizer(lang, mode=norm_mode, segmentation=segmentation)

    n_chars = n_tokens = 0
    char_delta = char_abs_delta = token_delta = lines_changed = 0
    guardrail = []

    for idx, raw in enumerate(texts):
        n_chars += len(raw)
        raw_tokens = len(raw.split())
        n_tokens += raw_tokens

        # before/after deltas + guardrail (needs the normalized form)
        out = norm_fn(raw)
        if out != raw:
            lines_changed += 1
        d = len(out) - len(raw)
        char_delta += d
        char_abs_delta += abs(d)
        token_delta += len(out.split()) - raw_tokens

        flags = _guardrail_flags(raw, out)
        if flags:
            guardrail.append({
                "line_no": idx,
                "snippet": _snippet(raw),
                "out": _snippet(out),
                "flags": flags,
            })

    ordered = _item_breakdown(texts, examples_per_item)
    return Report(
        input_path=input_path, label=os.path.basename(input_path), lang=lang,
        n_lines=len(texts), n_chars=n_chars, n_tokens=n_tokens,
        items=ordered, char_delta=char_delta, char_abs_delta=char_abs_delta,
        token_delta=token_delta, lines_changed=lines_changed,
        guardrail_flags=guardrail, norm_mode=norm_mode,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def _snippet(t: str, width: int = 80) -> str:
    t = t.strip()
    return t if len(t) <= width else t[:width - 1] + "…"


# --------------------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "—"


def _item_tables_html(items: list, n_lines: int) -> str:
    """The Handled + Gap normalization-breakdown tables (shared by the analyze and compare
    reports). ``items`` is an ordered list[ItemResult]; ``n_lines`` the denominator for the
    %-lines column."""
    e = html.escape
    # "% items" = each Item's share of total Item occurrences within its kind.
    tot = {"handled": 0, "gap": 0}
    for it in items:
        tot[it.kind] = tot.get(it.kind, 0) + it.total_hits

    def rows(kind: str) -> str:
        out = []
        for it in items:
            if it.kind != kind:
                continue
            ex = ('<span class="sep">·</span>'.join(
                      f'<span class="chg">{e(x)}</span>' for x in it.examples[:3])
                  if it.examples else "")
            out.append(f"""
      <tr>
        <td class="code">{e(it.code)}</td>
        <td><span class="metric metric-{it.metric.split('+')[0]}">{e(it.metric)}</span></td>
        <td class="title">{e(it.title)}</td>
        <td class="num">{it.total_hits:,}</td>
        <td class="num">{_pct(it.total_hits, tot.get(kind, 0))}</td>
        <td class="ex" dir="auto">{ex}</td>
      </tr>""")
        return "".join(out)

    return f"""
  <h2>Handled Items — firing frequency (E1–E12)</h2>
  <p class="note">How often each rule the pipeline already applies actually fires on this
     corpus (Items = total occurrences). High-frequency rows are the banked wins; near-zero
     rows are cheap insurance that rarely triggers here.</p>
  <div class="scroll"><table>
    <thead><tr><th>Item</th><th>Metric</th><th>Rule</th><th>Items</th><th>% items</th>
      <th>Affected word → fix</th></tr></thead>
    <tbody>{rows("handled")}</tbody>
  </table></div>

  <h2>Gap Items — measured prevalence (backlog)</h2>
  <p class="note">Unhandled patterns from <code>NORMALIZATION.md</code> (Items = total
     occurrences, an upper bound on how often the rule <em>could</em> fire), for
     prioritizing which backlog item to build next — most-frequent first.</p>
  <div class="scroll"><table>
    <thead><tr><th>Item</th><th>Metric</th><th>Pattern</th><th>Items</th><th>% items</th>
      <th>Affected word (pattern)</th></tr></thead>
    <tbody>{rows("gap")}</tbody>
  </table></div>"""


def render_html(report: Report, out_path: str) -> str:
    """Write a self-contained, theme-aware HTML report; return ``out_path``."""
    r = report
    e = html.escape

    gl = r.guardrail_flags
    guard_class = "ok" if not gl else "warn"
    guard_msg = ("No vanishing-vowel flags — normalization preserved vowel signs and "
                 "word counts across every line." if not gl else
                 f"{len(gl):,} line(s) flagged — review below.")
    guard_rows = "".join(f"""
      <tr>
        <td class="num">{g['line_no']:,}</td>
        <td dir="auto">{e(g['snippet'])}</td>
        <td dir="auto">{e(g['out'])}</td>
        <td class="ex">{e(', '.join(f'{k}:{v}' for k, v in g['flags'].items()))}</td>
      </tr>""" for g in gl[:50])

    tok_dir = "added" if r.token_delta >= 0 else "removed"
    chr_dir = "added" if r.char_delta >= 0 else "removed"

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Normalization analysis — {e(r.label)} ({e(r.lang)})</title>
<style>
  :root {{
    --bg:#f7f7f5; --card:#fff; --ink:#1c1b19; --muted:#6b6a67; --line:#e6e4df;
    --accent:#7c5cff; --ok:#1a7f5a; --warn:#b4531a; --chip:#efece6;
    --wer:#2563a8; --cer:#8a5a1a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#17161a; --card:#201f25; --ink:#eceaf0; --muted:#a3a1ab; --line:#312f3a;
      --accent:#a58cff; --ok:#5fd3a6; --warn:#f0a56a; --chip:#2a2932;
      --wer:#7fb0e6; --cer:#d9a05a;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#17161a; --card:#201f25; --ink:#eceaf0; --muted:#a3a1ab; --line:#312f3a;
    --accent:#a58cff; --ok:#5fd3a6; --warn:#f0a56a; --chip:#2a2932;
    --wer:#7fb0e6; --cer:#d9a05a;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding-block:2.4rem; padding-inline:20px; }}
  h1 {{ font-size:1.55rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
  h2 {{ font-size:1.12rem; margin:2.2rem 0 .7rem; letter-spacing:-.01em; }}
  h3 {{ font-size:1rem; margin:1.8rem 0 .4rem; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); margin:0 0 1.6rem; font-size:.92rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin:1.2rem 0 .4rem; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:14px 16px; }}
  .kpi .v {{ font-size:1.5rem; font-weight:650; letter-spacing:-.02em; }}
  .kpi .l {{ color:var(--muted); font-size:.8rem; margin-top:2px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
    border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:.9rem; }}
  .scroll {{ overflow-x:auto; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line);
    vertical-align:top; }}
  th {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.04em;
    color:var(--muted); font-weight:600; background:var(--chip); }}
  tr:last-child td {{ border-bottom:none; }}
  td.code {{ font-weight:700; font-variant-numeric:tabular-nums; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  td.title {{ min-width:180px; }}
  td.ex {{ color:var(--muted); font-size:.86rem; max-width:420px; }}
  td.ex .chg {{ color:var(--ink); }}
  td.ex .sep {{ color:var(--line); margin:0 .4em; }}
  .metric {{ font-size:.72rem; font-weight:700; padding:1px 7px; border-radius:999px;
    background:var(--chip); white-space:nowrap; }}
  .metric-WER {{ color:var(--wer); }} .metric-CER {{ color:var(--cer); }}
  .note {{ color:var(--muted); font-size:.86rem; margin:.5rem 0 0; }}
  .banner {{ border-radius:12px; padding:12px 16px; margin:.4rem 0 0;
    border:1px solid var(--line); background:var(--card); }}
  .banner.ok {{ border-color:color-mix(in srgb,var(--ok) 40%,var(--line)); }}
  .banner.warn {{ border-color:color-mix(in srgb,var(--warn) 45%,var(--line)); }}
  .banner .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
    margin-right:8px; vertical-align:middle; }}
  .banner.ok .dot {{ background:var(--ok); }} .banner.warn .dot {{ background:var(--warn); }}
  code {{ background:var(--chip); padding:1px 5px; border-radius:5px; font-size:.85em; }}
  footer {{ color:var(--muted); font-size:.8rem; margin-top:2.6rem;
    border-top:1px solid var(--line); padding-top:1rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Normalization analysis</h1>
  <p class="sub">file <code>{e(r.label)}</code> &middot; language <code>{e(r.lang)}</code>
     &middot; mode <code>{e(r.norm_mode)}</code>
     &middot; {r.n_lines:,} lines &middot; generated {e(r.generated_at)}</p>

  <div class="cards">
    <div class="kpi"><div class="v">{r.n_lines:,}</div><div class="l">lines</div></div>
    <div class="kpi"><div class="v">{r.n_tokens:,}</div><div class="l">tokens (raw)</div></div>
    <div class="kpi"><div class="v">{r.n_chars:,}</div><div class="l">characters (raw)</div></div>
    <div class="kpi"><div class="v">{_pct(r.lines_changed, r.n_lines)}</div>
      <div class="l">lines changed by normalize()</div></div>
    <div class="kpi"><div class="v">{abs(r.token_delta):,}</div>
      <div class="l">tokens {tok_dir} (WER-side)</div></div>
    <div class="kpi"><div class="v">{abs(r.char_delta):,}</div>
      <div class="l">chars {chr_dir} net (CER-side)</div></div>
  </div>

  {_item_tables_html(r.items, r.n_lines)}

  <h2>Guardrail — vanishing-vowels tripwire</h2>
  <div class="banner {guard_class}"><span class="dot"></span>{e(guard_msg)}</div>
  {"" if not gl else f'''<div class="scroll" style="margin-top:.8rem"><table>
    <thead><tr><th>Line</th><th>Raw</th><th>Normalized</th><th>Flags</th></tr></thead>
    <tbody>{guard_rows}</tbody></table></div>'''}

  <footer>
    Detectors probe the raw text; handled-item counts re-run the library's own pipeline
    sub-steps, so this report tracks the real normalizer. Gap-item counts are
    surface-pattern heuristics (upper bounds). Generated by
    <code>indic_normalizer.analysis</code>.
  </footer>
</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path


# --------------------------------------------------------------------------------------
# Compare mode — two files (e.g. reference vs hypothesis), paired by line order
# --------------------------------------------------------------------------------------
#
# The point of this library: you normalize BOTH reference and hypothesis so a WER/CER
# score penalizes real mis-transcriptions, not convention/encoding differences. Compare
# mode measures exactly that payoff — how many line-pairs that DIFFER raw become EQUAL
# after normalization (recovered), and which normalization Items did the recovering.


def _score_text(hyps: list, refs: list) -> dict:
    """Corpus WER/CER (%), via jiwer — the SAME computation AetherTranscribe's
    ``eval_metrics.score_text`` uses, so the analyzer's numbers match the official scores.
    Reference length = substitutions+deletions+hits.

    Falls back to a pure-stdlib edit-distance implementation if jiwer is not installed,
    which gives identical WER/CER totals (only the sub/del/ins split can tie-break
    differently); the fallback marks itself in the returned dict via ``engine``.
    """
    try:
        import jiwer
        w = jiwer.process_words(refs, hyps)
        c = jiwer.process_characters(refs, hyps)
        w_tot = w.substitutions + w.deletions + w.hits
        c_tot = c.substitutions + c.deletions + c.hits
        wpct = (lambda v: round(100.0 * v / w_tot, 4)) if w_tot else (lambda v: 0.0)
        cpct = (lambda v: round(100.0 * v / c_tot, 4)) if c_tot else (lambda v: 0.0)
        wer, cer = w.wer * 100.0, c.cer * 100.0
        return {
            "wer": round(wer, 4), "wer_sub": wpct(w.substitutions),
            "wer_delete": wpct(w.deletions), "wer_insert": wpct(w.insertions),
            "cer": round(cer, 4), "cer_sub": cpct(c.substitutions),
            "cer_delete": cpct(c.deletions), "cer_insert": cpct(c.insertions),
            "engine": "jiwer",
        }
    except Exception:
        return _score_text_stdlib(hyps, refs)


def _lev(ref: list, hyp: list) -> int:
    """Levenshtein edit distance (sub+ins+del), two-row DP. Fallback for _score_text."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ri = ref[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ri == hyp[j - 1] else 1))
        prev = cur
    return prev[n]


def _score_text_stdlib(hyps: list, refs: list) -> dict:
    """Corpus WER/CER via stdlib edit distance — same totals as jiwer (no sub/del/ins
    split). Used only when jiwer is unavailable."""
    we = wt = ce = ct = 0
    for ref, hyp in zip(refs, hyps):
        rw, hw = ref.split(), hyp.split()
        we += _lev(rw, hw); wt += len(rw)
        ce += _lev(list(ref), list(hyp)); ct += len(ref)
    wer = 100.0 * we / wt if wt else 0.0
    cer = 100.0 * ce / ct if ct else 0.0
    return {
        "wer": round(wer, 4), "wer_sub": 0.0, "wer_delete": 0.0, "wer_insert": 0.0,
        "cer": round(cer, 4), "cer_sub": 0.0, "cer_delete": 0.0, "cer_insert": 0.0,
        "engine": "stdlib",
    }


# --------------------------------------------------------------------------------------
# Causal attribution — which handled Item actually recovered a pair
# --------------------------------------------------------------------------------------
#
# The Malayalam normalize() pipeline is a fixed linear sequence of steps, each mapping to
# one handled Item. To attribute a recovery, we re-run the pipeline with ONE step disabled
# and see whether the pair still matches: if disabling Item X breaks the match, X was
# necessary for the recovery, so X is credited. This mirrors the real normalize() (steps
# reference the malayalam module's own functions), so it can't drift from it.

def _ml_steps(segmentation: bool = False):
    """Ordered (E-code, step_fn) list mirroring malayalam.normalize(). Each step_fn maps
    str->str. Built lazily so importing analysis never forces the malayalam import. The CER
    folds C6/C2/C1 are always present (they run in every eval normalize); when ``segmentation``
    is True the opt-in W4/W3/W5 boundary rules are also included, so recovery attribution
    matches the pipeline actually run."""
    _m = _ml
    steps = [
        ("E12", _m._fix_chillu_conjuncts),
        ("E2",  _m._IND_NORM.normalize),
        ("E1",  lambda t: unicodedata.normalize("NFC", t)),
        ("E3",  lambda t: _m._UNK.sub(" ", t)),
        ("E4",  lambda t: _m._ZW.sub("", t)),
        ("E6",  lambda t: "".join(_m._DIGITS.get(c, c) for c in t)),
        ("C6",  _m._fix_homoglyphs),          # ഠ->0 in a number (shared base; both modes)
        ("C2",  _m._fold_final_nasal),        # word-final മ് -> ം (shared base; both modes)
        ("E7",  lambda t: _m._GROUP_COMMA.sub("", t)),
    ]
    if segmentation:
        steps.append(("W4", _m._join_intra_word_hyphen))   # before E5 (which eats the hyphen)
    steps += [
        ("E5",  lambda t: _m._PUNCT.sub(" ", t)),
        ("E9",  _m._canonicalize_en_numbers),
        ("E8",  lambda t: _m._NUM_RE.sub(_m._verbalize_digits, t)),
        ("E10", lambda t: t.lower()),
        ("C1",  _m._fold_samvruthokaram),     # word-final ...ു്/...ു -> ...് (eval only)
    ]
    if segmentation:
        steps.append(("W3", _m._collapse_letter_runs))     # after E5/E10 (split tokens)
        steps.append(("W5", _m._collapse_reduplication))
    steps.append(("E11", lambda t: _m._SPACE.sub(" ", t).strip()))
    return steps


def _normalize_ablated(text: str, skip: "str | None", steps) -> str:
    """Run the pipeline over ``text`` skipping the step whose E-code == ``skip``."""
    for code, fn in steps:
        if code == skip:
            continue
        text = fn(text or "")
    return text


# Mechanical cleanup steps that are jointly required alongside almost any real fix
# (lowercase, whitespace collapse). Excluded from recovery attribution so the credited
# Item is the substantive cause, not the follow-on tidy-up.
_ATTR_EXCLUDE = {"E10", "E11"}


def _attribute_recovery(ref_raw: str, hyp_raw: str, steps) -> list:
    """For a recovered pair (unequal raw, equal after full normalization), return the
    E-codes of the SUBSTANTIVE handled Items that were necessary: disabling the step leaves
    the pair unequal. Ablation runs over the full pipeline (so dependencies are respected),
    but E10/E11 (lowercase, whitespace) are not reported as causes."""
    necessary = []
    for code, _fn in steps:
        if code in _ATTR_EXCLUDE:
            continue
        r = _normalize_ablated(ref_raw, code, steps)
        h = _normalize_ablated(hyp_raw, code, steps)
        if r != h:                      # without this step the pair no longer matches
            necessary.append(code)
    return necessary


@dataclass
class CompareReport:
    """Corpus WER/CER for a hypothesis file scored against a reference file, RAW and after
    normalization. Rates are corpus-level (total edit distance / total reference length) —
    the same quantity AetherTranscribe's eval_metrics computes (via jiwer). ``metrics_raw``
    / ``metrics_norm`` are the full dicts from :func:`_score_text` (wer, cer, their
    sub/delete/insert splits, engine)."""
    label_a: str                                   # reference file
    label_b: str                                   # hypothesis file
    lang: str
    n_pairs: int                                   # scored pairs (non-blank)
    length_mismatch: "tuple[int, int] | None"      # (len_ref, len_hyp) if files differ
    blank_pairs: int                               # pairs blank on both sides (skipped)
    metrics_raw: dict                              # WER/CER/... on raw text
    metrics_norm: dict                             # WER/CER/... after normalization
    n_lines: int = 0                               # ref+hyp lines behind the Item breakdown
    items: list = field(default_factory=list)      # normalization Item breakdown (all ref+hyp)
    # per-outcome pair counts and their Item breakdowns (over both ref+hyp lines of the
    # pairs in each bucket). equal = matched raw; recovered = differed raw, matched after
    # normalization; still = differed even after normalization.
    pairs_equal: int = 0
    pairs_recovered: int = 0
    pairs_still: int = 0
    n_equal: int = 0
    n_still: int = 0
    # Recovered bucket is CAUSAL: recovered_attr credits the handled Item(s) that actually
    # made each pair match (verified by pipeline ablation). recovery_unattributed counts
    # recovered pairs with no single necessary step (e.g. non-Malayalam, or joint effects).
    recovered_attr: list = field(default_factory=list)
    recovery_unattributed: int = 0
    # Still-different / already-equal keep the line-level "what fired" breakdown.
    items_equal: list = field(default_factory=list)
    items_still: list = field(default_factory=list)
    norm_mode: str = "eval"                         # normalizer mode used (eval | inference)
    generated_at: str = ""


def compare(file_a: str, file_b: str, lang: str = "ml",
            limit: int | None = None, norm_mode: str = "eval",
            segmentation: bool = False) -> CompareReport:
    """Score hypothesis ``file_b`` against reference ``file_a`` — corpus WER and CER, both
    on the RAW text and AFTER normalizing both sides, so the drop shows how much
    normalization recovers. WER/CER come from :func:`_score_text` (jiwer), matching
    AetherTranscribe's ``eval_metrics``.

    Pairs by line order (line N of A with line N of B). Blank lines are KEPT (as ``""``)
    so alignment holds when one side has empty entries; a pair blank on BOTH sides is
    skipped. If the files differ in length the extra lines of the longer file are ignored
    and ``length_mismatch`` records both counts.
    """
    a = load_lines(file_a, keep_blanks=True)
    b = load_lines(file_b, keep_blanks=True)
    length_mismatch = (len(a), len(b)) if len(a) != len(b) else None
    n = min(len(a), len(b))
    if limit is not None:
        n = min(n, limit)

    norm_fn, _ = get_normalizer(lang, mode=norm_mode, segmentation=segmentation)
    # Recovery attribution ablates the EVAL pipeline (WER/CER convergence is an eval-mode
    # concept); it is not meaningful on inference/ITN output, so it only runs in eval mode.
    # The step list matches the segmentation flag so W3/W4/W5 are attributable when enabled.
    steps = (_ml_steps(segmentation=segmentation)
             if (norm_mode == "eval" and lang in ("ml", "mal", "malayalam")) else None)
    refs_raw, hyps_raw, refs_norm, hyps_norm = [], [], [], []
    blank_pairs = 0
    lines_equal, lines_still = [], []               # raw ref+hyp lines, by outcome
    recovery_credit = {}                            # E-code -> # pairs it recovered
    pairs_recovered = pairs_equal = 0
    unattributed = 0                                 # recovered pairs with no clear cause

    for i in range(n):
        ref_raw, hyp_raw = a[i], b[i]
        if not ref_raw and not hyp_raw:            # nothing to score
            blank_pairs += 1
            continue
        ref_n, hyp_n = norm_fn(ref_raw), norm_fn(hyp_raw)
        refs_raw.append(ref_raw); hyps_raw.append(hyp_raw)
        refs_norm.append(ref_n); hyps_norm.append(hyp_n)

        if ref_raw == hyp_raw:                      # already equal before normalization
            pairs_equal += 1
            lines_equal += [ref_raw, hyp_raw]
        elif ref_n == hyp_n:                        # recovered by normalization
            pairs_recovered += 1
            causes = _attribute_recovery(ref_raw, hyp_raw, steps) if steps else []
            if causes:
                for c in causes:
                    recovery_credit[c] = recovery_credit.get(c, 0) + 1
            else:
                unattributed += 1
        else:                                       # still different after normalization
            lines_still += [ref_raw, hyp_raw]

    pairs_still = len(lines_still) // 2
    metrics_raw = _score_text(hyps_raw, refs_raw)
    metrics_norm = _score_text(hyps_norm, refs_norm)

    all_lines = refs_raw + hyps_raw
    items = _item_breakdown(all_lines)              # overall (all pairs)
    items_still = _item_breakdown(lines_still)      # what fired in the residual errors
    items_equal = _item_breakdown(lines_equal)      # cosmetic (already-matching pairs)

    # recovery attribution -> ItemResult-like rows (code, title, lines_hit = pairs recovered)
    _title = {it.code: it.title for it in _handled_items()}
    _metric = {it.code: it.metric for it in _handled_items()}
    recovered_attr = [
        ItemResult(code, "handled", _metric.get(code, ""), _title.get(code, code),
                   lines_hit=cnt, total_hits=cnt)
        for code, cnt in sorted(recovery_credit.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    return CompareReport(
        label_a=os.path.basename(file_a), label_b=os.path.basename(file_b), lang=lang,
        n_pairs=len(refs_raw), length_mismatch=length_mismatch,
        blank_pairs=blank_pairs, metrics_raw=metrics_raw, metrics_norm=metrics_norm,
        n_lines=len(all_lines), items=items,
        pairs_equal=pairs_equal, pairs_recovered=pairs_recovered, pairs_still=pairs_still,
        recovered_attr=recovered_attr, recovery_unattributed=unattributed,
        items_still=items_still, items_equal=items_equal,
        n_still=len(lines_still), n_equal=len(lines_equal), norm_mode=norm_mode,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def render_compare_html(report: CompareReport, out_path: str) -> str:
    """Write a self-contained, theme-aware WER/CER comparison report; return out_path."""
    r = report
    e = html.escape
    mr, mn = r.metrics_raw, r.metrics_norm

    def delta(raw, norm):
        d = norm - raw
        cls = "down" if d < 0 else ("up" if d > 0 else "flat")
        sign = "−" if d < 0 else ("+" if d > 0 else "")
        return f'<span class="delta {cls}">{sign}{abs(d):.2f}</span>'

    warn = ""
    if r.length_mismatch:
        la, lb = r.length_mismatch
        warn = (f'<div class="banner warn" style="margin-bottom:1rem"><span class="dot">'
                f'</span>File lengths differ ({la:,} vs {lb:,} lines) — scored the first '
                f'{r.n_pairs + r.blank_pairs:,} pairs by line order; extra lines ignored.</div>')

    engine = mn.get("engine", "")
    engine_note = ("" if engine == "jiwer" else
                   '<p class="note">jiwer not installed — WER/CER totals computed with the '
                   'stdlib fallback (identical rates; sub/delete/insert split omitted).</p>')

    # Recovered bucket: causal attribution table (Item -> pairs it recovered).
    def recovery_table() -> str:
        rows = "".join(f"""
      <tr>
        <td class="code">{e(it.code)}</td>
        <td><span class="metric metric-{it.metric.split('+')[0]}">{e(it.metric)}</span></td>
        <td class="title">{e(it.title)}</td>
        <td class="num">{it.lines_hit:,}</td>
        <td class="num">{_pct(it.lines_hit, r.pairs_recovered)}</td>
      </tr>""" for it in r.recovered_attr)
        unattr = (f'<tr><td class="code">—</td><td></td>'
                  f'<td class="title">unattributed (joint / non-Malayalam)</td>'
                  f'<td class="num">{r.recovery_unattributed:,}</td>'
                  f'<td class="num">{_pct(r.recovery_unattributed, r.pairs_recovered)}</td></tr>'
                  if r.recovery_unattributed else "")
        return f"""<div class="scroll"><table>
    <thead><tr><th>Item</th><th>Metric</th><th>Rule</th><th>Pairs recovered</th>
      <th>% of recovered</th></tr></thead>
    <tbody>{rows}{unattr}</tbody></table></div>"""

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WER/CER compare — {e(r.label_a)} vs {e(r.label_b)}</title>
<style>
  :root {{
    --bg:#f7f7f5; --card:#fff; --ink:#1c1b19; --muted:#6b6a67; --line:#e6e4df;
    --ok:#1a7f5a; --warn:#b4531a; --chip:#efece6; --up:#b4531a; --down:#1a7f5a;
    --wer:#2563a8; --cer:#8a5a1a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#17161a; --card:#201f25; --ink:#eceaf0; --muted:#a3a1ab; --line:#312f3a;
      --ok:#5fd3a6; --warn:#f0a56a; --chip:#2a2932; --up:#f0a56a; --down:#5fd3a6;
      --wer:#7fb0e6; --cer:#d9a05a;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#17161a; --card:#201f25; --ink:#eceaf0; --muted:#a3a1ab; --line:#312f3a;
    --ok:#5fd3a6; --warn:#f0a56a; --chip:#2a2932; --up:#f0a56a; --down:#5fd3a6;
    --wer:#7fb0e6; --cer:#d9a05a;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:900px; margin:0 auto; padding-block:2.4rem; padding-inline:20px; }}
  h1 {{ font-size:1.55rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
  h2 {{ font-size:1.12rem; margin:2.2rem 0 .7rem; letter-spacing:-.01em; }}
  h3 {{ font-size:1rem; margin:1.8rem 0 .4rem; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); margin:0 0 1.6rem; font-size:.92rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin:1.2rem 0 1rem; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:14px 16px; }}
  .kpi .v {{ font-size:1.7rem; font-weight:650; letter-spacing:-.02em; }}
  .kpi .l {{ color:var(--muted); font-size:.8rem; margin-top:2px; }}
  .kpi .raw {{ color:var(--muted); font-size:.82rem; margin-top:4px; }}
  .delta {{ font-size:.9rem; font-weight:700; }}
  .delta.down {{ color:var(--down); }} .delta.up {{ color:var(--up); }}
  .delta.flat {{ color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
    border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:.9rem; }}
  .scroll {{ overflow-x:auto; }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); }}
  th {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.04em;
    color:var(--muted); font-weight:600; background:var(--chip); }}
  tr:last-child td {{ border-bottom:none; }}
  td.code {{ font-weight:700; font-variant-numeric:tabular-nums; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  td.title {{ min-width:180px; }}
  td.ex {{ color:var(--muted); font-size:.86rem; max-width:420px; }}
  td.ex .chg {{ color:var(--ink); }}
  td.ex .sep {{ color:var(--line); margin:0 .4em; }}
  .metric {{ font-size:.72rem; font-weight:700; padding:1px 7px; border-radius:999px;
    background:var(--chip); white-space:nowrap; }}
  .metric-WER {{ color:var(--wer); }} .metric-CER {{ color:var(--cer); }}
  .note {{ color:var(--muted); font-size:.86rem; margin:.5rem 0 0; }}
  .banner {{ border-radius:12px; padding:12px 16px; border:1px solid var(--line);
    background:var(--card); }}
  .banner.warn {{ border-color:color-mix(in srgb,var(--warn) 45%,var(--line)); }}
  .banner .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
    margin-right:8px; vertical-align:middle; background:var(--warn); }}
  code {{ background:var(--chip); padding:1px 5px; border-radius:5px; font-size:.85em; }}
  footer {{ color:var(--muted); font-size:.8rem; margin-top:2.6rem;
    border-top:1px solid var(--line); padding-top:1rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>WER / CER compare</h1>
  <p class="sub">reference <code>{e(r.label_a)}</code> &middot; hypothesis
     <code>{e(r.label_b)}</code> &middot; language <code>{e(r.lang)}</code>
     &middot; mode <code>{e(r.norm_mode)}</code>
     &middot; {r.n_pairs:,} scored pairs
     &middot; generated {e(r.generated_at)}</p>

  {warn}

  <div class="cards">
    <div class="kpi"><div class="v">{mn['wer']:.2f}%</div>
      <div class="l">WER (normalized)</div>
      <div class="raw">raw {mr['wer']:.2f}% &middot; {delta(mr['wer'], mn['wer'])}</div></div>
    <div class="kpi"><div class="v">{mn['cer']:.2f}%</div>
      <div class="l">CER (normalized)</div>
      <div class="raw">raw {mr['cer']:.2f}% &middot; {delta(mr['cer'], mn['cer'])}</div></div>
  </div>

  <p class="note">Corpus-level WER/CER (total edit distance ÷ total reference length),
     the same quantity AetherTranscribe's <code>eval_metrics</code> reports. "Normalized"
     runs both reference and hypothesis through the <code>{e(r.lang)}</code> normalizer
     before scoring; the delta is how much normalization moves the rate (lower is better).</p>
  {engine_note}

  {f'<p class="note">{r.blank_pairs:,} blank-on-both-sides pairs were skipped (no reference content to score).</p>' if r.blank_pairs else ''}

  <h2 style="margin-top:2.6rem">Normalization breakdown by pair outcome</h2>
  <p class="note">Split by what happened to each ref↔hyp pair. The <em>Recovered</em> section
     is causal — it credits the handled Item(s) that actually made each pair match, verified
     by re-running normalization with that step disabled. The other two show which Items
     fired on the lines in that bucket.</p>

  <h3 style="margin-top:1.8rem">① Recovered — differed raw, matched after normalization
     <span class="note">({r.pairs_recovered:,} pairs)</span></h3>
  <p class="note">What normalization actually fixed. Each Item is credited only where
     disabling it would leave the pair unequal — so only handled Items appear.</p>
  {recovery_table() if r.pairs_recovered else '<p class="note">No pairs were recovered by normalization.</p>'}

  <h3 style="margin-top:2rem">② Still different — not equal even after normalization
     <span class="note">({r.pairs_still:,} pairs · {r.n_still:,} lines)</span></h3>
  <p class="note">Genuine differences normalization can't close. Handled Items here fired
     but weren't enough; gap Items are the unhandled patterns present in the residual
     errors — candidate levers to build next.</p>
  {_item_tables_html(r.items_still, r.n_still) if r.n_still else '<p class="note">Every scored pair matched after normalization.</p>'}

  <h3 style="margin-top:2rem">③ Already equal — matched before normalization
     <span class="note">({r.pairs_equal:,} pairs · {r.n_equal:,} lines)</span></h3>
  <p class="note">Pairs that already matched raw. Items firing here are cosmetic — no
     recovery was needed.</p>
  {_item_tables_html(r.items_equal, r.n_equal) if r.n_equal else '<p class="note">No pairs matched before normalization.</p>'}

  <footer>
    Pairs aligned by line order; blank lines kept so alignment holds. WER/CER via jiwer
    when available (else a stdlib edit-distance fallback with identical rates), matching
    <code>eval_metrics.score_text</code>. Generated by <code>indic_normalizer.analysis</code>.
  </footer>
</div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m indic_normalizer.analysis",
        description="Analyze one corpus file, or compare two (reference vs hypothesis).")
    p.add_argument("inputs", nargs="+", metavar="FILE",
                   help="one file -> per-file Item analysis; two files (reference then "
                        "hypothesis, e.g. dataset_ref.txt dataset_hyp.txt) -> WER/CER "
                        "compare, paired by line order.")
    p.add_argument("--lang", default="ml", help="language code (default: ml).")
    p.add_argument("--norm-mode", default="eval",
                   choices=["eval", "training", "fine-tune", "inference"],
                   help="normalizer mode: eval (default; model-facing verbalization, "
                        "aliases training/fine-tune) or inference (reader-facing ITN — "
                        "numbers to digits, punctuation/case preserved).")
    p.add_argument("--segmentation", action="store_true",
                   help="enable the opt-in W3/W4/W5 boundary rules (eval only; OFF by "
                        "default — measured net-neutral-to-negative on the sample corpora).")
    p.add_argument("--out", default=None,
                   help="HTML report output path (default: normalization_report.html / "
                        "wer_cer_compare.html).")
    p.add_argument("--limit", type=int, default=None,
                   help="analyze/compare only the first N lines/pairs (quick look).")
    p.add_argument("--json", default=None,
                   help="also write the raw report as JSON to this path.")
    args = p.parse_args(argv)

    if len(args.inputs) > 2:
        p.error("give one file (analyze) or two files (compare), not more.")

    if len(args.inputs) == 2:
        _run_compare(args)
    else:
        _run_analyze(args)


def _run_analyze(args):
    out = args.out or "normalization_report.html"
    report = analyze(args.inputs[0], lang=args.lang, limit=args.limit,
                     norm_mode=args.norm_mode, segmentation=args.segmentation)
    render_html(report, out)

    print(f"Analyzed {report.n_lines:,} lines from '{report.label}' "
          f"({report.n_tokens:,} tokens, {report.n_chars:,} chars) [mode={report.norm_mode}].")
    print(f"  lines changed by normalize(): {report.lines_changed:,} "
          f"({_pct(report.lines_changed, report.n_lines)})")
    print(f"  token delta (WER-side): {report.token_delta:+,}   "
          f"char delta (CER-side): {report.char_delta:+,}")
    print(f"  guardrail flags: {len(report.guardrail_flags):,}")
    print("  top firing items:", ", ".join(
        f"{it.code}={it.lines_hit}" for it in report.items if it.lines_hit)[:400] or "none")
    print(f"HTML report -> {out}")

    if args.json:
        payload = {
            "mode": "analyze",
            "norm_mode": report.norm_mode,
            "input_path": report.input_path, "label": report.label,
            "lang": report.lang, "n_lines": report.n_lines,
            "n_tokens": report.n_tokens, "n_chars": report.n_chars,
            "lines_changed": report.lines_changed, "token_delta": report.token_delta,
            "char_delta": report.char_delta, "guardrail_flags": len(report.guardrail_flags),
            "items": [{"code": it.code, "kind": it.kind, "metric": it.metric,
                       "title": it.title, "lines_hit": it.lines_hit,
                       "total_hits": it.total_hits, "examples": it.examples}
                      for it in report.items],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"JSON summary -> {args.json}")


def _run_compare(args):
    out = args.out or "wer_cer_compare.html"
    if args.norm_mode != "eval":
        print("  note: compare/WER-CER scoring uses eval-mode normalization; "
              f"--norm-mode={args.norm_mode} still runs but recovery attribution is eval-only.")
    r = compare(args.inputs[0], args.inputs[1], lang=args.lang, limit=args.limit,
                norm_mode=args.norm_mode, segmentation=args.segmentation)
    render_compare_html(r, out)

    mr, mn = r.metrics_raw, r.metrics_norm
    print(f"Scored '{r.label_b}' against '{r.label_a}' — {r.n_pairs:,} pairs"
          + (f" ({r.blank_pairs:,} blank skipped)." if r.blank_pairs else "."))
    if r.length_mismatch:
        print(f"  ! file lengths differ {r.length_mismatch} — scored first {r.n_pairs + r.blank_pairs:,} by line order.")
    print(f"  WER: raw {mr['wer']:.2f}% -> normalized {mn['wer']:.2f}%  "
          f"({mn['wer'] - mr['wer']:+.2f})")
    print(f"  CER: raw {mr['cer']:.2f}% -> normalized {mn['cer']:.2f}%  "
          f"({mn['cer'] - mr['cer']:+.2f})")
    if mn.get("engine") != "jiwer":
        print("  (jiwer not installed — used stdlib edit-distance fallback; rates identical)")
    print(f"HTML report -> {out}")

    if args.json:
        payload = {
            "mode": "compare",
            "norm_mode": r.norm_mode,
            "reference": r.label_a, "hypothesis": r.label_b, "lang": r.lang,
            "n_pairs": r.n_pairs, "length_mismatch": r.length_mismatch,
            "blank_pairs": r.blank_pairs,
            "raw": r.metrics_raw, "normalized": r.metrics_norm,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"JSON summary -> {args.json}")


if __name__ == "__main__":
    main()
