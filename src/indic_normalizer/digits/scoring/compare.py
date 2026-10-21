"""Compare the scoring methods (head tdt|ctc x unit segment|word) on an audio dir.

For every number it records each method's pick, margin and review flag, then reports:
    - per method: picks by language (ml / en) and form, % needing review, median margin,
      time per unit
    - pairwise agreement between methods (same pick)
    - optional: agreement with one or more hypothesis sets (``hyps_dirs``: dirs of <stem>.jsonl
      with {start_time, end_time, text} lines, e.g. base_hypotheses-streaming and
      base_hypotheses-segment; each is reported under its dir name). A pick "matches" when its
      words appear in that hypothesis text overlapping the sentence (both eval-normalized).
      "oracle" is the share of numbers where ANY candidate appears — the ceiling; "match | oracle"
      is the pick's agreement on just those rows. Different ASR engines (streaming vs per-segment,
      other models) write numbers differently, so compare methods WITHIN one reference, not
      absolute numbers across references.
    - optional (``transcribe=True``): the same against the scoring model's own greedy transcript
      of each sentence, reported as reference ``model`` (needs a scorer with ``transcribe()``).
Outputs: <out_root>/<data dir name>/scoring_compare/{report.html, report.json, numbers.tsv}
report.html is self-contained (open it in a browser): sortable / filterable tables and a numbers
explorer (filter by method, reference, outcome, class, form, margin; search; CSV export).
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from itertools import combinations
from typing import Optional

from ...languages import malayalam as _ml
from .audio import find_audio, load_wav, slice_audio
from .config import HEADS, UNITS, ScoringConfig
from .html_report import _finite, numbers_payload, page
from .pipeline import _run, decide, iter_groups, list_sources

METHODS = [(h, u) for u in UNITS for h in HEADS]

# numbers.tsv column documentation (also written into report.html). <m> = method, <ref> = reference.
COLUMNS = [
    ("file", "transcript / audio stem (`<file>.json`, `<file>.wav`)"),
    ("sentence", "index of the sentence in the JSON `sentences` list"),
    ("sent_start, sent_end", "sentence start / end time in the audio (s) — the window the *segment* unit scores (plus `pad_s`)"),
    ("word_start, word_end", "start / end time (s) of the timestamped word holding the number; empty when it could not be matched (word unit then falls back to segment)"),
    ("core", "the number as written in the train-normalized text (`1,75,000` -> `175000`; `23-ന്`, `4:00-5:00` keep their punctuation)"),
    ("class", "number type from `digits.find`: integer, num_suffix, time, ordinal, percent, range, decimal, date, ..."),
    ("word_unit", "`word` = the word / window units scored the number's own word window (+- `context_words`); `segment(fallback)` = words could not be matched, whole sentence used for word and window"),
    ("n_cands", "how many candidate readings were scored (<= `max_candidates`)"),
    ("candidates", "all candidate readings, ` | `-separated, in candidate order (first = default)"),
    ("default", "the default reading (native Malayalam, contracted) — what you get without audio scoring"),
    ("sentence_text", "original reference sentence from the JSON"),
    ("hyp:<ref>", "hypothesis text of reference `<ref>` overlapping the sentence (+- `hyp_pad_s`, default 0.5 s), eval-normalized. `<ref>` = a `--hyps-dir` name, or `model` (`--transcribe`)"),
    ("oracle:<ref>", "True if ANY candidate appears in `hyp:<ref>` — rows where agreement is achievable at all"),
    ("<m>", "the reading method `<m>` picked (lowest NLL); `<m>` = head (tdt / ctc) / unit (segment = whole sentence; word = word window encoded alone; window = sentence encoded once, number's frames scored)"),
    ("<m>:lang", "language of the pick: `ml` (Malayalam words) or `en` (English words in Malayalam script)"),
    ("<m>:form", "how the pick reads: contracted / glide / english / half / hh_mm / ordinal / ... (see digits.ml)"),
    ("<m>:margin", "runner-up NLL minus best NLL, in nats (larger = more confident; empty if only one candidate)"),
    ("<m>:review_margin", "NLL of the best candidate from a different reading family minus best NLL (contracted / glide / spaced spellings are one family; English, drop-thousand, half, ... are others)"),
    ("<m>:review", "True if review_margin < `margin_threshold` (would be flagged needs_review)"),
    ("<m>:style_gap", "score of the best other spelling of the chosen reading (contracted / glide / spaced) minus the audio's spelling, nats; empty = no spelling variants"),
    ("<m>:text_issue", "a problem around the number its reading can't fix: a currency sign written as a separate word (₹ 500, Rs. 500, 500 ₹), or an isolated sentence in another language / script in a mixed file; always flagged for review"),
    ("<m>:suffix_gap", "num_suffix only: best reading WITH the written suffix minus the best bare number, nats; > 0 = the audio prefers no suffix"),
    ("<m>:suffix_issue", "num_suffix only: why the written suffix looks wrong (the audio prefers the bare number, or a vowel suffix after -ം such as 55000-യുടെ); always flagged for review"),
    ("<m>:spelling", "`audio` = the audio's spelling kept (style_gap >= `style_margin`); `default` = it was too close, the default `style` spelling was used"),
    ("<m>:match:<ref>", "True if the pick appears in `hyp:<ref>`"),
    ("default:match:<ref>", "True if the default reading appears in `hyp:<ref>` (the no-scoring baseline)"),
]


def _load_hyps(hyps_dir, stem):
    if not hyps_dir:
        return None
    for ext in (".jsonl", ".json"):
        p = os.path.join(hyps_dir, stem + ext)
        if os.path.exists(p):
            if ext == ".jsonl":
                return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            d = json.load(open(p, encoding="utf-8"))
            return d.get("sentences") or d.get("segments") or []
    return None


def _hyp_text(hyps, s, pad=0.5):
    try:
        a, b = float(s["start_time"]) - pad, float(s["end_time"]) + pad
    except (KeyError, TypeError, ValueError):
        return ""
    t = " ".join(h.get("text") or "" for h in hyps
                 if float(h.get("end_time") or 0) >= a and float(h.get("start_time") or 0) <= b)
    return " " + _ml.normalize(t) + " "


def _in_hyp(text, hyp):
    t = _ml.normalize(text)
    return bool(t) and (" " + t + " ") in hyp


def _malayalam_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum("\u0d00" <= c <= "\u0d7f" for c in letters) / len(letters) if letters else 0.0


def compare_dir(data_dir: str, audio_dir: str, scorer, cfg: ScoringConfig, out_root: str = "outputs",
                hyps_dirs=None, limit: Optional[int] = None, log=print,
                transcribe: bool = False) -> dict:
    """``hyps_dirs``: a dir, a list of dirs, or {name: dir}; each becomes a reference."""
    if isinstance(hyps_dirs, str):
        hyps_dirs = [hyps_dirs]
    if isinstance(hyps_dirs, (list, tuple)):
        hyps_dirs = {os.path.basename(os.path.normpath(d)): d for d in hyps_dirs}
    hyps_dirs = dict(hyps_dirs or {})
    refs = list(hyps_dirs) + (["model"] if transcribe else [])
    srcs = list_sources(data_dir)[:limit]
    out = os.path.join(out_root, os.path.basename(os.path.normpath(data_dir)), "scoring_compare")
    os.makedirs(out, exist_ok=True)
    heads = [h for h in HEADS if h in getattr(scorer, "heads", HEADS)]
    methods = [(h, u) for (h, u) in METHODS if h in heads]
    rows, timing = [], Counter()
    t_start = time.perf_counter()
    if hasattr(scorer, "stats"):
        scorer.stats.clear()
    done = 0
    for items, load_s in iter_groups(srcs, audio_dir, cfg, UNITS):
        timing["load_and_plan"] += load_s
        ok = [it for it in items if "error" not in it]
        group_plans = [p for it in ok for p in it["plans"]]
        for u in UNITS:                                   # one pooled scoring pass per unit
            t0 = time.perf_counter()
            _run(scorer, group_plans, u, heads)
            timing[f"score_{u}"] += time.perf_counter() - t0
        own = {}
        if transcribe:
            todo = [(it["stem"], p) for it in ok for p in it["plans"] if p.get("clip") is not None]
            t0 = time.perf_counter()
            texts = scorer.transcribe([p["clip"] for _, p in todo]) if todo else []
            timing["transcribe"] += time.perf_counter() - t0
            for (stem, p), t in zip(todo, texts):
                own[(stem, p["idx"])] = " " + _ml.normalize(t) + " "
        for it in items:
            done += 1
            stem = it["stem"]
            if "error" in it:
                log(f"[{done}/{len(srcs)}] skip {stem}: {it['error']}")
                continue
            doc, plans = it["doc"], it["plans"]
            timing["audio_s"] += it["audio_s"]
            rows.extend(_rows_for(stem, doc, plans, hyps_dirs, refs, own, methods, cfg, scorer))
            log(f"[{done}/{len(srcs)}] {stem}: {sum(len(p['numbers']) for p in plans)} numbers")

    timing["total"] = time.perf_counter() - t_start
    rep = _report(rows, methods, timing, refs, cfg)
    rep["timing"] = _timing(timing, getattr(scorer, "stats", None), len(rows))
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(_finite(rep), f, ensure_ascii=False, indent=2)
    name = os.path.basename(os.path.normpath(data_dir))
    with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as f:
        f.write(page(f"Scoring compare — {name}", _markdown(rep, name),
                     numbers_payload(rows, [f"{h}/{u}" for h, u in methods], refs, f"{cfg.head}/{cfg.unit}")))
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out, "numbers.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, keys, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    rep["out_dir"] = out
    rep["rows"] = rows                      # in-memory only (not in report.json); used by sweep
    return rep


def _timing(t, stats, n_numbers):
    """Wall-clock per stage + scorer batching counters. rtf = processing s / audio s."""
    audio = t.get("audio_s", 0.0)
    out = {"audio_hours": round(audio / 3600, 3), "numbers": n_numbers,
           "stages_s": {k: round(v, 2) for k, v in t.items() if k != "audio_s"}}
    if audio:
        out["rtf_total"] = round(t.get("total", 0) / audio, 4)
        for k in ("score_segment", "score_word", "score_window", "transcribe"):
            if t.get(k):
                out[f"rtf_{k}"] = round(t[k] / audio, 4)
    if n_numbers and t.get("total"):
        out["numbers_per_s"] = round(n_numbers / t["total"], 2)
    if stats:
        s = dict(stats)
        out["scorer"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}
        if s.get("pairs_scored") and s.get("clips_encoded"):
            out["scorer"]["candidates_per_clip"] = round(s["pairs_scored"] / s["clips_encoded"], 2)
    return out


def _rows_for(stem, doc, plans, hyps_dirs, refs, own, methods, cfg, scorer):
    loaded = {name: _load_hyps(d, stem) for name, d in hyps_dirs.items()}
    rows = []
    for p in plans:
        sent = p["sentence"]
        hyp = {name: (_hyp_text(h, sent, cfg.hyp_pad_s) if h is not None else None)
               for name, h in loaded.items()}
        if "model" in refs:
            hyp["model"] = own.get((stem, p["idx"]))
        for n in p["numbers"]:
            w = (doc.get("words") or [])[n.word_idx[0]] if n.word_idx else {}
            r = {"file": stem, "sentence": p["idx"],
                 "sent_start": sent.get("start_time"), "sent_end": sent.get("end_time"),
                 "word_start": w.get("start_time"), "word_end": w.get("end_time"),
                 "core": n.span.core, "class": n.span.cls,
                 "word_unit": n.unit_used.get("word"), "n_cands": len(n.cands),
                 "candidates": " | ".join(c.text for c in n.cands),
                 "default": n.cands[0].text,
                 "sentence_text": (sent.get("text") or "").replace("\t", " ")}
            for name in refs:
                hx = hyp.get(name)
                if hx is not None:
                    r[f"hyp:{name}"] = hx.strip()
                    r[f"oracle:{name}"] = any(_in_hyp(c.text, hx) for c in n.cands)
            for h, u in methods:
                d = decide(n, u, h, cfg, scorer)
                c = n.cands[d["best"]]
                key = f"{h}/{u}"
                r[key] = c.text
                r[key + ":lang"], r[key + ":form"] = c.lang, c.form
                r[key + ":margin"] = round(d["margin"], 4) if d["margin"] != float("inf") else None
                r[key + ":review_margin"] = (round(d["review_margin"], 4)
                                             if d["review_margin"] != float("inf") else None)
                r[key + ":review"] = d["needs_review"]
                r[key + ":style_gap"] = round(d["style_gap"], 4) if d.get("style_gap") is not None else None
                r[key + ":spelling"] = ("default" if d.get("style_fallback") else "audio") if d.get("style_gap") is not None else None
                r[key + ":suffix_issue"] = d.get("suffix_issue") or ""
                r[key + ":text_issue"] = d.get("text_issue") or ""
                r[key + ":suffix_gap"] = round(d["suffix_gap"], 4) if d.get("suffix_gap") is not None else None
                for name in refs:
                    if hyp.get(name) is not None:
                        r[f"{key}:match:{name}"] = _in_hyp(c.text, hyp[name])
                        if d.get("suffix_gap") is not None:
                            r[f"{key}:bare_match:{name}"] = _in_hyp(d["suffix_bare"], hyp[name])
                            r[f"{key}:with_match:{name}"] = _in_hyp(d["suffix_with"], hyp[name])
            for name in refs:
                if hyp.get(name) is not None:
                    r[f"default:match:{name}"] = _in_hyp(n.cands[0].text, hyp[name])
            rows.append(r)
    return rows


def _true(v):
    return v is True or v == "True"


OUTCOMES = ("success", "verify", "failed", "unchecked")
OUTCOME_HELP = {
    "success": "not flagged, and the pick appears in the reference hypothesis",
    "verify": "flagged needs_review (review_margin < margin_threshold) -> goes to a human",
    "failed": "not flagged, a candidate appears in the reference, but not the pick (confident and wrong)",
    "unchecked": "not flagged, and no candidate appears in the reference (cannot be checked)",
}


def outcomes(rows, method, ref):
    """Per-number outcome of ``method`` against reference ``ref`` (rows = numbers.tsv rows, as dicts
    of python values or of strings). Returns counts, % of all numbers and % of the checkable ones."""
    c = Counter()
    for r in rows:
        if f"hyp:{ref}" not in r or r.get(f"hyp:{ref}") is None:
            continue
        if _true(r.get(f"{method}:review")):
            c["verify"] += 1
            c["verify_pick_matches"] += _true(r.get(f"{method}:match:{ref}"))
        elif _true(r.get(f"{method}:match:{ref}")):
            c["success"] += 1
        elif _true(r.get(f"oracle:{ref}")):
            c["failed"] += 1
        else:
            c["unchecked"] += 1
    n = sum(c[k] for k in OUTCOMES)
    out = {"n": n, **{k: c[k] for k in OUTCOMES}, "verify_pick_matches": c["verify_pick_matches"]}
    out.update({f"{k}_pct": _pct(c[k], n) for k in OUTCOMES})
    checked = c["success"] + c["failed"]
    out["success_of_checked_pct"] = _pct(c["success"], checked)     # auto-accepted, checkable rows
    out["failed_of_checked_pct"] = _pct(c["failed"], checked)
    return out


def outcomes_markdown(table, refs):
    """table: {method: {ref: outcomes()}}"""
    L = ["## Outcomes per number", "",
         "; ".join(f"**{k}** = {v}" for k, v in OUTCOME_HELP.items()) + ". "
         "*of checked* = among auto-accepted numbers whose reference contains a candidate.", ""]
    for ref in refs:
        L += [f"Reference `{ref}`:", "",
              "| Method | Success | Verify | Failed | Unchecked | Success / failed of checked |",
              "|---|---|---|---|---|---|"]
        for m, per in table.items():
            o = per[ref]
            L.append(f"| {m} | {o['success']} ({o['success_pct']}%) | {o['verify']} ({o['verify_pct']}%) | "
                     f"{o['failed']} ({o['failed_pct']}%) | {o['unchecked']} ({o['unchecked_pct']}%) | "
                     f"{o['success_of_checked_pct']}% / {o['failed_of_checked_pct']}% |")
        L.append("")
    return L


def _pct(a, b):
    return round(100 * a / b, 2) if b else 0.0


RM_BANDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, 20), (20, float("inf"))]


def _band_label(a, b):
    return f"{a}–{'∞' if b == float('inf') else b}"


def _suffix_distribution(rows, methods, refs):
    """num_suffix numbers per |suffix gap| band and winner (bare / suffix), with how often the
    reference contains the bare reading vs the suffixed one — to choose a suffix margin."""
    out = {}
    for h, u in methods:
        k = f"{h}/{u}"
        sel_all = [r for r in rows if r.get(k + ":suffix_gap") is not None]
        if not sel_all:
            continue
        bands = []
        for a, b in RM_BANDS:
            band = {"band": _band_label(a, b)}
            for side, test in (("bare", lambda g: g > 0), ("suffix", lambda g: g <= 0)):
                sel = [r for r in sel_all if test(r[k + ":suffix_gap"]) and a <= abs(r[k + ":suffix_gap"]) < b]
                x = {"n": len(sel), "ref": {}}
                for ref in refs:
                    have = [r for r in sel if (k + ":bare_match:" + ref) in r]
                    x["ref"][ref] = {"rows": len(have),
                                     "bare": sum(bool(r[k + ":bare_match:" + ref]) for r in have),
                                     "with": sum(bool(r[k + ":with_match:" + ref]) for r in have)}
                band[side] = x
            bands.append(band)
        out[k] = bands
    return out


def _rm_distribution(rows, methods, refs):
    """Per review-margin band and method: numbers, and how often the pick matches each reference
    (given oracle) — shows how accuracy grows with the margin, to choose margin_threshold."""
    out = {}
    for h, u in methods:
        k = f"{h}/{u}"
        bands = []
        for a, b in RM_BANDS + [(None, None)]:
            sel = [r for r in rows if (r.get(k + ":review_margin") is None if a is None
                                       else r.get(k + ":review_margin") is not None and a <= r[k + ":review_margin"] < b)]
            band = {"band": "none" if a is None else _band_label(a, b), "n": len(sel), "match": {}}
            for ref in refs:
                orc = [r for r in sel if r.get(f"oracle:{ref}")]
                hit = sum(bool(r.get(f"{k}:match:{ref}")) for r in orc)
                band["match"][ref] = {"hit": hit, "oracle": len(orc), "pct": _pct(hit, len(orc)) if orc else None}
            bands.append(band)
        out[k] = bands
    return out


def _report(rows, methods, timing, refs, cfg):
    N = len(rows)
    rep = {"numbers": N, "files": len({r["file"] for r in rows}),
           "word_unit_fallbacks": sum(1 for r in rows if r.get("word_unit") == "segment(fallback)"),
           "seconds": {u: round(t, 2) for u, t in timing.items() if u != "audio_s"},
           "margin_threshold": cfg.margin_threshold, "hyp_pad_s": cfg.hyp_pad_s, "methods": {}, "agreement": {}, "references": {}}
    for name in refs:
        have = [r for r in rows if f"hyp:{name}" in r]
        orc = [r for r in have if r[f"oracle:{name}"]]
        ref = {"rows": len(have), "oracle": len(orc), "oracle_pct": _pct(len(orc), len(have)),
               "hyp_malayalam_pct": _pct(sum(_malayalam_share(r[f"hyp:{name}"]) >= 0.5 for r in have), len(have)),
               "default_match": sum(bool(r.get(f"default:match:{name}")) for r in have)}
        ref["default_match_pct"] = _pct(ref["default_match"], len(have))
        ref["default_match_given_oracle_pct"] = _pct(ref["default_match"], len(orc))
        rep["references"][name] = ref
    for h, u in methods:
        k = f"{h}/{u}"
        margins = [r[k + ":margin"] for r in rows if r.get(k + ":margin") is not None]
        m = {"lang": dict(Counter(r[k + ":lang"] for r in rows)),
             "form": dict(Counter(r[k + ":form"] for r in rows).most_common()),
             "changed_from_default": sum(r[k] != r["default"] for r in rows),
             "needs_review": sum(r[k + ":review"] for r in rows),
             "needs_review_pct": _pct(sum(r[k + ":review"] for r in rows), N),
             "median_margin": round(statistics.median(margins), 3) if margins else None,
             "match": {}}
        for name in refs:
            have = [r for r in rows if f"hyp:{name}" in r]
            hit = sum(bool(r.get(f"{k}:match:{name}")) for r in have)
            orc = sum(bool(r[f"oracle:{name}"]) for r in have)
            m["match"][name] = {"match": hit, "pct": _pct(hit, len(have)), "given_oracle_pct": _pct(hit, orc)}
        rep["methods"][k] = m
    for (a, b) in combinations([f"{h}/{u}" for h, u in methods], 2):
        same = sum(r[a] == r[b] for r in rows)
        rep["agreement"][f"{a} vs {b}"] = {"same": same, "pct": _pct(same, N)}
    by_cls = defaultdict(Counter)
    for r in rows:
        for h, u in methods:
            by_cls[r["class"]][f"{h}/{u}:review"] += r[f"{h}/{u}:review"]
        by_cls[r["class"]]["n"] += 1
    rep["by_class"] = {c: dict(v) for c, v in sorted(by_cls.items(), key=lambda x: -x[1]["n"])}
    rep["outcomes"] = {f"{h}/{u}": {ref: outcomes(rows, f"{h}/{u}", ref) for ref in refs} for h, u in methods}
    rep["review_margin_distribution"] = _rm_distribution(rows, methods, refs)
    rep["suffix_gap_distribution"] = _suffix_distribution(rows, methods, refs)
    return rep


def _markdown(rep, name):
    refs = list(rep["references"])
    L = [f"# Number-reading scoring: method comparison — `{name}`", "",
         f"Numbers **{rep['numbers']}** in **{rep['files']}** files · review threshold "
         f"{rep['margin_threshold']} nats · hypothesis window ±{rep['hyp_pad_s']} s · word-unit fallbacks to segment: {rep['word_unit_fallbacks']} · "
         f"time (s): {rep['seconds']}", ""]
    hdr = "| Method | ml / en picks | Changed from default | Needs review | Median margin |"
    sep = "|---|---|---|---|---|"
    for ref in refs:
        hdr += f" Matches `{ref}` (all / given oracle) |"
        sep += "---|"
    L += [hdr, sep]
    for k, m in rep["methods"].items():
        line = (f"| {k} | {m['lang'].get('ml', 0)} / {m['lang'].get('en', 0)} | {m['changed_from_default']} | "
                f"{m['needs_review']} ({m['needs_review_pct']}%) | {m.get('median_margin')} |")
        for ref in refs:
            x = m["match"][ref]
            line += f" {x['match']} ({x['pct']}% / {x['given_oracle_pct']}%) |"
        L.append(line)
    if refs:
        L += ["", "## References", "",
              "| Reference | Rows | Hyp mostly Malayalam script | Oracle (any candidate in hyp) | Default reading matches (all / given oracle) |",
              "|---|---|---|---|---|"]
        for ref, x in rep["references"].items():
            L.append(f"| `{ref}` | {x['rows']} | {x['hyp_malayalam_pct']}% | {x['oracle']} ({x['oracle_pct']}%) | "
                     f"{x['default_match']} ({x['default_match_pct']}% / {x['default_match_given_oracle_pct']}%) |")
        L += ["", "Different ASR engines write numbers differently (and a per-segment engine may switch "
              "script on short clips), so compare methods within one reference; *given oracle* is the "
              "fairest column."]
    dist = rep.get("review_margin_distribution")
    if dist:
        ms = list(dist)
        n_all = rep["numbers"] or 1
        L += ["", "## Review-margin distribution", "",
              "Review margin = score of the best *different reading* minus the pick (nats); below "
              f"`margin_threshold` ({rep['margin_threshold']}) a number needs review. 1 nat ≈ the pick is 2.7× more "
              "likely than the alternative, 2 ≈ 7×, 5 ≈ 150×. *none* = only one reading family. Cells: numbers "
              "in the band (% of all numbers).", "",
              "| Review margin (nats) | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
        for i, b in enumerate(dist[ms[0]]):
            L.append(f"| {b['band']} | " + " | ".join(
                f"{dist[m][i]['n']} ({round(100 * dist[m][i]['n'] / n_all, 1)}%)" for m in ms) + " |")
        for ref in refs:
            L += ["", f"Pick matches `{ref}` per band (given oracle: hits / numbers whose reference contains a "
                  "candidate). If accuracy is still low just above the threshold, a higher threshold would catch "
                  "more wrong picks.", "",
                  "| Review margin (nats) | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
            for i, b in enumerate(dist[ms[0]]):
                cells = []
                for m in ms:
                    x = dist[m][i]["match"][ref]
                    cells.append(f"{x['pct']}% ({x['hit']}/{x['oracle']})" if x["oracle"] else "–")
                L.append(f"| {b['band']} | " + " | ".join(cells) + " |")
    sdist = rep.get("suffix_gap_distribution")
    for m, bands in (sdist or {}).items():
        L += ["", f"## Suffix-gap distribution — `{m}` (num_suffix)", "",
              "Suffix gap = best reading WITH the written suffix minus the best bare number (nats). "
              "*Bare wins*: the audio prefers no suffix (all flagged now). For each reference: how many of "
              "those numbers have the **bare** reading in it vs the **suffixed** one — if bare wins above some "
              "gap are mostly bare in the reference, that gap is a safe suffix margin.", ""]
        head = "| Gap (nats) | Bare wins |" + "".join(f" `{x}` bare / suffixed |" for x in refs) + \
               " Suffix wins |" + "".join(f" `{x}` bare / suffixed |" for x in refs)
        L += [head, "|---|" + "---|" * (2 + 2 * len(refs))]
        for b in bands:
            cells = [str(b["bare"]["n"])] + [f"{b['bare']['ref'][x]['bare']} / {b['bare']['ref'][x]['with']}" for x in refs] + \
                    [str(b["suffix"]["n"])] + [f"{b['suffix']['ref'][x]['bare']} / {b['suffix']['ref'][x]['with']}" for x in refs]
            L.append(f"| {b['band']} | " + " | ".join(cells) + " |")
    if refs and rep.get("outcomes"):
        L += [""] + outcomes_markdown(rep["outcomes"], refs)
    tm = rep.get("timing")
    if tm:
        L += ["", "## Timing", "", f"Audio **{tm['audio_hours']} h**, numbers **{tm['numbers']}**, "
              f"real-time factor (total) **{tm.get('rtf_total')}**, numbers/s {tm.get('numbers_per_s')}.", "",
              "| Stage | Seconds |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in tm["stages_s"].items()]
        if tm.get("scorer"):
            L += ["", "Scorer batching: " + ", ".join(f"{k} {v}" for k, v in tm["scorer"].items())]
    L += ["", "## Agreement between methods", "", "| Pair | Same pick |", "|---|---|"]
    for k, v in rep["agreement"].items():
        L.append(f"| {k} | {v['same']} ({v['pct']}%) |")
    L += ["", "## Picks by form", ""]
    for k, m in rep["methods"].items():
        L.append(f"- **{k}**: {m['form']}")
    L += ["", "## Needs review by number class", "", "| Class | n | " +
          " | ".join(rep["methods"]) + " |", "|---|---|" + "---|" * len(rep["methods"])]
    for c, v in rep["by_class"].items():
        L.append(f"| {c} | {v['n']} | " + " | ".join(str(v.get(k + ':review', 0)) for k in rep["methods"]) + " |")
    L += ["", "## Columns of numbers.tsv", "", "One row per number.", "", "| Column | Meaning |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in COLUMNS]
    L.append("")
    return "\n".join(L)
