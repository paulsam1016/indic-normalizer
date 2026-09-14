"""Corpus analyzer for the normalization Items — find, quantify, and prioritize.

Runs a Malayalam ASR corpus (the ``dataset`` transcript list, or the ``calls``
transcriptions) through the pipeline and reports, per normalization Item:

  * **firing frequency** — how often each *handled* rule (E1-E12) actually changes
    text on this corpus, so the banked wins are evidence-backed rather than assumed;
  * **gap detection** — how often each *unhandled* pattern from NORMALIZATION.md
    (W1/W3/W4 segmentation, C1/C2/C5/C6 residue) appears, with real examples, so the
    backlog is prioritized by measured prevalence;
  * **before/after deltas** — characters and tokens the pipeline adds or removes, split
    into a WER-side (token-count) and CER-side (character) view;
  * **guardrail** — a corpus-wide vanishing-vowels tripwire (ACL-2024 identity check):
    any line where normalization drops a vowel sign / virama or splits a word at one.

Nothing here mutates the corpus; detectors are read-only probes. The heavy detectors
reuse the library's own primitives (``base``, the Malayalam module's regexes) so the
report tracks the real pipeline, not a re-implementation.

Usage
-----
    from indic_normalizer.analysis import analyze, render_html
    report = analyze(source="calls", lang="ml")     # or source="dataset"
    render_html(report, "report.html")

CLI
---
    python -m indic_normalizer.analysis --source calls --lang ml --out report.html
"""

from __future__ import annotations

import glob
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
# Corpus loading (the dataset / calls selector)
# --------------------------------------------------------------------------------------

# Default corpus root: the repo's data/ dir, resolved relative to this file
# (src/indic_normalizer/analysis.py -> ../../data). Override with data_root= / --data-root.
_DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
)

SOURCES = ("dataset", "calls")
# Future seam: data/<lang>/dataset_outputs.jsonl and calls_outputs/*.jsonl carry
# paired reference/hypothesis (raw + normalized) fields. A "pairs" source could load
# those to measure per-Item impact on ref vs hyp (and match/mismatch flips) directly;
# add it to SOURCES and a branch in load_corpus when wanted.


def _coerce_text(raw: str, *, prefer: str) -> str:
    """Return the transcript text from one corpus record.

    The corpus files hold **plain text** now (one transcript per dataset line; the bare
    transcription per call file), but this also accepts the legacy JSON shapes
    (``{"text": ...}`` for dataset rows, ``{"transcription": ...}`` for calls) so the
    analyzer keeps working whichever form is on disk.
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
            return str(obj.get(prefer) or obj.get("text") or obj.get("transcription") or "")
        return s
    return s


def load_corpus(source: str, lang: str = "ml", data_root: str | None = None) -> list[str]:
    """Load transcript strings for ``source`` (``"dataset"`` | ``"calls"``).

    ``dataset`` -> ``<root>/<lang_dir>/dataset.txt`` (one transcript per line).
    ``calls``   -> every ``<root>/<lang_dir>/calls/*.txt`` (one transcription each).
    ``<lang_dir>`` is the language's folder name (``malayalam`` for ``ml``), falling
    back to the bare ``<root>`` layout if the language subfolder is absent.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    root = data_root or _DEFAULT_DATA_ROOT
    lang_dir = {"ml": "malayalam"}.get(lang, lang)

    def _dir(*parts: str) -> str:
        cand = os.path.join(root, lang_dir, *parts)
        if os.path.exists(cand):
            return cand
        return os.path.join(root, *parts)   # fall back to non-language-nested layout

    texts: list[str] = []
    if source == "dataset":
        path = _dir("dataset.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset not found: {path}")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                t = _coerce_text(line, prefer="text")
                if t:
                    texts.append(t)
    else:  # calls
        calls_dir = _dir("calls")
        files = sorted(glob.glob(os.path.join(calls_dir, "*.txt")))
        if not files:
            raise FileNotFoundError(f"no call files under: {calls_dir}")
        for fp in files:
            with open(fp, encoding="utf-8") as fh:
                t = _coerce_text(fh.read(), prefer="transcription")
            if t:
                texts.append(t)
    return texts


# --------------------------------------------------------------------------------------
# Item detectors
# --------------------------------------------------------------------------------------
#
# Each detector answers "does Item X apply to this line, and how many times?" on the RAW
# input. HANDLED items (E*) are counted by re-running the exact pipeline sub-step and
# seeing whether it changes the text; GAP items (W*/C*) are counted by matching the
# surface pattern NORMALIZATION.md describes. A detector returns an int hit-count (0 =
# does not fire), plus optionally captures an example.

_VIRAMA = "്"
_ZWJ = "‍"
_ZWNJ = "‌"

# --- handled-item helpers (mirror the real pipeline) ----------------------------------

_INDIC_DIGIT = re.compile(r"[०-९൦-൯]")   # sample Devanagari + Malayalam
_ANY_DECIMAL = re.compile(r"\d")
_ASCII_DIGIT_RUN = re.compile(r"[0-9]+")
_GROUP_COMMA = _ml._GROUP_COMMA
_CHILLU_CONJ = _ml._CHILLU_CONJ
_EN_ANY = _ml._EN_ANY
_EN_STRONG = _ml._EN_STRONG


def _count_indic_digits(t: str) -> int:
    return sum(1 for _ in _INDIC_DIGIT.finditer(t))


def _count_zero_width(t: str) -> int:
    return t.count(_ZWJ) + t.count(_ZWNJ) + t.count("﻿")


def _count_en_number_runs(t: str) -> int:
    """Runs of English-in-Malayalam-script number words that E9 would convert."""
    words = t.split()
    runs, i = 0, 0
    while i < len(words):
        j = i
        while j < len(words) and words[j] in _EN_ANY:
            j += 1
        if j > i and any(words[k] in _EN_STRONG for k in range(i, j)):
            runs += 1
            i = j
        else:
            i = max(j, i + 1)
    return runs


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


def _w3_acronym_hits(t: str) -> int:
    """Count spaced single-letter runs of length >= 2 (acronym-style)."""
    words = t.split()
    hits, run = 0, 0
    for w in words:
        if _W3_TOKEN.match(w):
            run += 1
        else:
            if run >= 2:
                hits += 1
            run = 0
    if run >= 2:
        hits += 1
    return hits


# --------------------------------------------------------------------------------------
# Item catalogue
# --------------------------------------------------------------------------------------

@dataclass
class Item:
    code: str
    kind: str          # "handled" | "gap"
    metric: str        # "WER" | "CER" | "CER+WER"
    title: str
    detect: object     # callable(raw_text) -> int hit count

    def __post_init__(self):
        pass


def _handled_items() -> list[Item]:
    return [
        Item("E1", "handled", "CER", "Unicode NFC",
             lambda t: int(unicodedata.normalize("NFC", t) != t)),
        Item("E2", "handled", "CER", "Chillu / ZWJ-ZWNJ canonicalization",
             lambda t: int(_ml._IND_NORM.normalize(t) != t)),
        Item("E3", "handled", "CER+WER", "⁇ / <unk> removal",
             lambda t: len(base.UNK.findall(t))),
        Item("E4", "handled", "CER", "Zero-width (ZWNJ/ZWJ/BOM) drop",
             _count_zero_width),
        Item("E5", "handled", "CER+WER", "ASCII + Indic punctuation removal",
             lambda t: len(base.PUNCT.findall(t))),
        Item("E6", "handled", "CER", "Indic digit → ASCII",
             _count_indic_digits),
        Item("E7", "handled", "CER+WER", "Digit-grouping commas (70,000→70000)",
             lambda t: len(_GROUP_COMMA.findall(t))),
        Item("E8", "handled", "WER+CER", "Native digit verbalization (24→ഇരുപത്തിനാല്)",
             lambda t: len(_ASCII_DIGIT_RUN.findall(_GROUP_COMMA.sub("", base.fold_digits(t))))),
        Item("E9", "handled", "WER+CER", "English↔Malayalam number canonicalization",
             _count_en_number_runs),
        Item("E10", "handled", "CER", "Lowercasing (code-mixed Latin)",
             lambda t: int(t.lower() != t)),
        Item("E11", "handled", "WER", "Whitespace collapse + trim",
             lambda t: int(base.SPACE.sub(" ", t).strip() != t)),
        Item("E12", "handled", "CER", "Mis-encoded chillu conjunct (ൻ്റ→ന്റ)",
             lambda t: len(_CHILLU_CONJ.findall(t))),
    ]


def _gap_items() -> list[Item]:
    return [
        Item("C1", "gap", "CER", "Samvruthokaram / final-vowel drift",
             lambda t: len(_C1_FINAL.findall(t))),
        Item("C2", "gap", "CER", "Anusvara vs nasal spelling (word-final)",
             lambda t: len(_C2_ANUSVARA.findall(t)) + len(_C2_NASAL.findall(t))),
        Item("W1", "gap", "WER", "Sandhi / postposition split-merge (candidates)",
             _postposition_candidates),
        Item("W3", "gap", "WER", "Acronym / spaced single-letter runs",
             _w3_acronym_hits),
        Item("W4", "gap", "WER", "Intra-word hyphen",
             lambda t: len(_W4_HYPHEN.findall(t))),
        Item("C5", "gap", "CER", "Foreign-word (Latin) tokens — transliteration variants",
             lambda t: len(_C5_LATIN.findall(t))),
        Item("C6", "gap", "CER", "Visual-lookalike letter/digit (O/0 by Malayalam)",
             lambda t: len(_C6_LOOKALIKE.findall(t))),
    ]


# W1 postposition split candidate: a short function-morpheme token that commonly appears
# both fused and spaced. We can't know the reference form, so we count *occurrences of the
# spaced free-standing postposition*, which is the split side of the W1 split-merge pair.
_W1_POSTPOS = {
    "ലേക്ക്",  # ലേക്ക്  (to/towards)
    "ക്ക്",              # ക്ക്
    "നിന്ന്",  # നിന്ന്  (from)
    "ഓട്",                    # ഓട്     (with)
    "ക്കാൻ",        # ക്കാൻ
    "വേണ്ടി",  # വേണ്ടി  (for)
}


def _postposition_candidates(t: str) -> int:
    return sum(1 for w in t.split() if w in _W1_POSTPOS)


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
    source: str
    lang: str
    data_root: str
    n_lines: int
    n_chars: int
    n_tokens: int
    items: list                       # list[ItemResult]
    char_delta: int                   # sum(len(norm) - len(raw)), chars
    char_abs_delta: int               # sum(|per-char edit|) approximated by len change magnitude
    token_delta: int                  # sum(norm_tokens - raw_tokens)
    lines_changed: int                # lines where norm(raw) != raw
    guardrail_flags: list             # list of {line_no, snippet, flags}
    generated_at: str = ""


def analyze(source: str, lang: str = "ml", data_root: str | None = None,
            limit: int | None = None, examples_per_item: int = 5) -> Report:
    """Analyze ``source`` and return a :class:`Report`.

    ``limit`` caps the number of corpus lines (handy for a quick look); ``None`` = all.
    """
    texts = load_corpus(source, lang=lang, data_root=data_root)
    if limit is not None:
        texts = texts[:limit]

    norm_fn, _ = get_normalizer(lang)
    catalogue = _handled_items() + _gap_items()
    results = {it.code: ItemResult(it.code, it.kind, it.metric, it.title) for it in catalogue}

    n_chars = n_tokens = 0
    char_delta = char_abs_delta = token_delta = lines_changed = 0
    guardrail = []

    for idx, raw in enumerate(texts):
        n_chars += len(raw)
        raw_tokens = len(raw.split())
        n_tokens += raw_tokens

        # per-item detection on the raw line
        for it in catalogue:
            try:
                hits = it.detect(raw)
            except Exception:
                hits = 0
            if hits:
                r = results[it.code]
                r.lines_hit += 1
                r.total_hits += hits
                if len(r.examples) < examples_per_item:
                    r.examples.append(_snippet(raw))

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

    ordered = sorted(
        results.values(),
        key=lambda r: (r.kind != "handled", -r.lines_hit, r.code),
    )
    return Report(
        source=source, lang=lang, data_root=data_root or _DEFAULT_DATA_ROOT,
        n_lines=len(texts), n_chars=n_chars, n_tokens=n_tokens,
        items=ordered, char_delta=char_delta, char_abs_delta=char_abs_delta,
        token_delta=token_delta, lines_changed=lines_changed,
        guardrail_flags=guardrail,
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


def render_html(report: Report, out_path: str) -> str:
    """Write a self-contained, theme-aware HTML report; return ``out_path``."""
    r = report
    e = html.escape

    def item_rows(kind: str) -> str:
        rows = []
        for it in r.items:
            if it.kind != kind:
                continue
            ex = " · ".join(e(x) for x in it.examples[:3]) if it.examples else ""
            rows.append(f"""
      <tr>
        <td class="code">{e(it.code)}</td>
        <td><span class="metric metric-{it.metric.split('+')[0]}">{e(it.metric)}</span></td>
        <td class="title">{e(it.title)}</td>
        <td class="num">{it.lines_hit:,}</td>
        <td class="num">{_pct(it.lines_hit, r.n_lines)}</td>
        <td class="num">{it.total_hits:,}</td>
        <td class="ex" dir="auto">{ex}</td>
      </tr>""")
        return "".join(rows)

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
<title>Normalization analysis — {e(r.source)} ({e(r.lang)})</title>
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
  td.ex {{ color:var(--muted); font-size:.86rem; max-width:340px; }}
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
  <p class="sub">source <code>{e(r.source)}</code> &middot; language <code>{e(r.lang)}</code>
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

  <h2>Handled Items — firing frequency (E1–E12)</h2>
  <p class="note">How often each rule the pipeline already applies actually changes text
     on this corpus. High-frequency rows are the banked wins; near-zero rows are cheap
     insurance that rarely triggers here.</p>
  <div class="scroll"><table>
    <thead><tr><th>Item</th><th>Metric</th><th>Rule</th><th>Lines</th><th>% lines</th>
      <th>Occurrences</th><th>Examples</th></tr></thead>
    <tbody>{item_rows("handled")}</tbody>
  </table></div>

  <h2>Gap Items — measured prevalence (backlog)</h2>
  <p class="note">Unhandled patterns from <code>NORMALIZATION.md</code>. Counts are
     surface-pattern candidates (an upper bound on how often the rule <em>could</em>
     fire), for prioritizing which backlog item to build next — most-frequent first.</p>
  <div class="scroll"><table>
    <thead><tr><th>Item</th><th>Metric</th><th>Pattern</th><th>Lines</th><th>% lines</th>
      <th>Occurrences</th><th>Examples</th></tr></thead>
    <tbody>{item_rows("gap")}</tbody>
  </table></div>

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
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m indic_normalizer.analysis",
        description="Find, quantify, and prioritize normalization Items across a corpus.")
    p.add_argument("--source", choices=SOURCES, default="calls",
                   help="corpus selector: 'dataset' (transcript list) or 'calls' "
                        "(call transcriptions). Default: calls.")
    p.add_argument("--lang", default="ml", help="language code (default: ml).")
    p.add_argument("--out", default="normalization_report.html",
                   help="HTML report output path.")
    p.add_argument("--data-root", default=None,
                   help="corpus root dir (default: the repo's data/).")
    p.add_argument("--limit", type=int, default=None,
                   help="analyze only the first N lines (quick look).")
    p.add_argument("--json", default=None,
                   help="also write the raw report as JSON to this path.")
    args = p.parse_args(argv)

    report = analyze(args.source, lang=args.lang, data_root=args.data_root,
                     limit=args.limit)
    render_html(report, args.out)

    # concise stdout summary
    print(f"Analyzed {report.n_lines:,} lines from '{args.source}' "
          f"({report.n_tokens:,} tokens, {report.n_chars:,} chars).")
    print(f"  lines changed by normalize(): {report.lines_changed:,} "
          f"({_pct(report.lines_changed, report.n_lines)})")
    print(f"  token delta (WER-side): {report.token_delta:+,}   "
          f"char delta (CER-side): {report.char_delta:+,}")
    print(f"  guardrail flags: {len(report.guardrail_flags):,}")
    print("  top firing items:", ", ".join(
        f"{it.code}={it.lines_hit}" for it in report.items if it.lines_hit)[:400] or "none")
    print(f"HTML report -> {args.out}")

    if args.json:
        payload = {
            "source": report.source, "lang": report.lang, "n_lines": report.n_lines,
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


if __name__ == "__main__":
    main()
