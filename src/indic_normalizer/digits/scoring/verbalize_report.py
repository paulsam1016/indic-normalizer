"""report.html + summary.json for ``verbalize`` (in OUT_DIR/<data dir name>/). report.html: totals, picks by language / form /
class, margin distribution, the config, and a numbers explorer (filter by accepted / verify,
class, form, language, margin; search; CSV export)."""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict

from .. import verbalize as _verbalize
from .html_report import numbers_payload, page


def report_rows(stem, plans, cfg, scorer=None):
    """One row per number (column names as in compare's numbers.tsv, method = head/unit)."""
    from .pipeline import decide, _round
    m = f"{cfg.head}/{cfg.unit}"
    rows = []
    for p in plans:
        if not p["numbers"]:
            continue
        s = p["sentence"]
        ds = [decide(n, cfg.unit, cfg.head, cfg, scorer) for n in p["numbers"]]
        it = iter(n.cands[d["best"]].text for n, d in zip(p["numbers"], ds))
        verb = _verbalize(p["text_train"], style=cfg.style, chooser=lambda span, cs: next(it))
        for n, d in zip(p["numbers"], ds):
            c = n.cands[d["best"]]
            rows.append({"file": stem, "sentence": p["idx"], "sent_start": s.get("start_time"),
                         "sent_end": s.get("end_time"), "core": n.span.core, "class": n.span.cls,
                         "n_cands": len(n.cands), "candidates": " | ".join(x.text for x in n.cands),
                         "default": n.cands[0].text, m: c.text, f"{m}:lang": c.lang, f"{m}:form": c.form,
                         f"{m}:margin": _round(d["margin"]), f"{m}:review_margin": _round(d["review_margin"]),
                         f"{m}:review": bool(d["needs_review"]), "text_verbalized": verb,
                         f"{m}:style_gap": _round(d.get("style_gap")),
                         f"{m}:spelling": (("default" if d.get("style_fallback") else "audio")
                                            if d.get("style_gap") is not None else None),
                         f"{m}:suffix_issue": d.get("suffix_issue") or "",
                         f"{m}:text_issue": d.get("text_issue") or "",
                         f"{m}:suffix_gap": _round(d.get("suffix_gap")),
                         "sentence_text": (s.get("text") or "").replace("\t", " ")})
    return rows


def _pct(a, b):
    return round(100 * a / b, 2) if b else 0.0


def _markdown(rows, tot, cfg, name):
    m = f"{cfg.head}/{cfg.unit}"
    N = len(rows)
    rev = sum(r[f"{m}:review"] for r in rows)
    L = [f"# Verbalize report — `{name}`", "",
         f"Method `{m}` · dtype `{cfg.compute_dtype}` · review threshold {cfg.margin_threshold} nats · "
         f"output → `{tot.get('out_dir')}`", "",
         "## Totals", "",
         "What the run produced. Every input file lands in exactly one of **verbalized** (all numbers "
         "decided confidently; written with verbalized texts), **verify** (at least one number needs review: "
         "written with the confident numbers verbalized and the uncertain ones still as digits) or **failed** "
         "(could not be processed: missing / unreadable audio or JSON; the input is copied unchanged). "
         "*Needs review* = numbers whose review margin is below the threshold (the best different reading "
         "scored almost as well as the pick). *Pick ≠ default* = numbers where the audio changed the reading "
         "away from the no-audio default (native, contracted). Number counts cover verbalized + verify files.", "",
         "| Item | Value |", "|---|---|",
         f"| Input files | {tot.get('files')} |",
         f"| → verbalized | {tot.get('verbalized')} ({_pct(tot.get('verbalized') or 0, tot.get('files') or 0)}%) |",
         f"| → verify | {tot.get('verify')} ({_pct(tot.get('verify') or 0, tot.get('files') or 0)}%) |",
         f"| → failed | {tot.get('failed')} ({_pct(tot.get('failed') or 0, tot.get('files') or 0)}%) |",
         f"| Numbers | {N} |",
         f"| Spelling: audio's kept / default `{cfg.style}` used (gap < {getattr(cfg, 'style_margin', 0)} nats) | "
         f"{sum(r.get(m + ':spelling') == 'audio' for r in rows)} / {sum(r.get(m + ':spelling') == 'default' for r in rows)} |",
         f"| Needs review | {rev} ({_pct(rev, N)}%) |",
         f"| Pick ≠ default reading | {sum(r[m] != r['default'] for r in rows)} ({_pct(sum(r[m] != r['default'] for r in rows), N)}%) |",
         f"| Audio | {tot.get('audio_hours')} h |", f"| Wall time | {tot.get('seconds')} s (rtf {tot.get('rtf')}) |",
         ]
    if "word_fallbacks" in tot:
        L += [f"| Words given the default reading (couldn't be matched to the sentence) | {tot['word_fallbacks']} |",
              f"| Sentences whose words don't spell the sentence text | {tot['word_text_mismatch']} |"]
    lang = Counter(r[f"{m}:lang"] for r in rows)
    L += ["", "## Picks by language", "",
          "**Language** of the chosen reading: `ml` = Malayalam number words, `en` = English number words "
          "written in Malayalam script (ട്വന്റി ഫൈവ്). **Numbers** = how many picks, **%** = share of all numbers.", "",
          "| Language | Numbers | % |", "|---|---|---|"]
    L += [f"| {k} | {v} | {_pct(v, N)}% |" for k, v in lang.most_common()]
    form = Counter(r[f"{m}:form"] for r in rows)
    frev = Counter(r[f"{m}:form"] for r in rows if r[f"{m}:review"])
    L += ["", "## Picks by form", "",
          "**Form** = how the chosen reading is written: `contracted` (ഇരുപത്തഞ്ച്), `glide` (ഇരുപത്തിയഞ്ച്), "
          "`spaced` (ഇരുപത്തി അഞ്ച്), `english`, `ordinal` / `ordinal_th` (ഇരുപത്തഞ്ചാം / ട്വന്റി ഫിഫ്ത്ത്), "
          "`drop_thousand` (\"thousand\" left out: 2500 → രണ്ട് അഞ്ഞൂറ്), `half`, `hh_mm` / `oclock` "
          "/ `hh_mani_mm` / `bare` / `quarter` (times: 4 മണി, 4 30, 4 മണി 30, നാലേകാൽ), `point` (decimals), "
          "`d_m_y` (dates), `digit_by_digit` (phones, ids), `piecewise` (ids read in groups), `times` (×). "
          "Ranges: `a_b` (5-6 → അഞ്ച് ആറ്), `a_to_b` (… ടു …), `from_to` (… മുതൽ … വരെ), `a_by_b` (… ബൈ …). "
          "Percent: `<form>+shathamanam` (ശതമാനം), `+percent` (പെർസെന്റ്), `+percentage` (പെർസെന്റേജ്). "
          "**Needs review** = how many picks of that form were flagged.", "",
          "| Form | Numbers | % | Needs review |", "|---|---|---|---|"]
    L += [f"| {k} | {v} | {_pct(v, N)}% | {frev[k]} |" for k, v in form.most_common()]
    by = defaultdict(list)
    for r in rows:
        by[r["class"]].append(r)
    L += ["", "## By number class", "",
          "**Class** = what kind of number it is in the text (integer, num_suffix = number + suffix like 23-ന്, "
          "time, ordinal, percent, range, decimal, date, year, phone, ...). **Needs review / % review** = flagged "
          "numbers of that class. **English picks** = how many were read in English. **Top form** = most common "
          "form chosen. **Median margin** = typical gap (nats) between the pick and the runner-up; larger = "
          "more confident.", "",
          "| Class | Numbers | Needs review | % review | English picks | Top form | Median margin |",
          "|---|---|---|---|---|---|---|"]
    for c, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        mg = [r[f"{m}:margin"] for r in rs if r[f"{m}:margin"] is not None]
        rv = sum(r[f"{m}:review"] for r in rs)
        top = Counter(r[f"{m}:form"] for r in rs).most_common(1)[0][0]
        L.append(f"| {c} | {len(rs)} | {rv} | {_pct(rv, len(rs))}% | "
                 f"{sum(r[f'{m}:lang'] == 'en' for r in rs)} | {top} | "
                 f"{round(statistics.median(mg), 2) if mg else ''} |")
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, 20), (20, float("inf"))]
    rm = [r[f"{m}:review_margin"] for r in rows]
    L += ["", "## Review-margin distribution", "",
          "Review margin = score of the best *different reading* minus the best (nats); below the "
          f"threshold ({cfg.margin_threshold}) a number needs review. Empty = only one reading family.", "",
          "**Numbers / %** = how many numbers fall in each band; the first band (below 1 nat with the "
          "default threshold) is what gets flagged. 1 nat ≈ the pick is e ≈ 2.7× more likely than the "
          "alternative; 5 nats ≈ 150×.", "",
          "| Review margin (nats) | Numbers | % |", "|---|---|---|"]
    for a, b in edges:
        k = sum(1 for x in rm if x is not None and a <= x < b)
        L.append(f"| {a}–{'∞' if b == float('inf') else b} | {k} | {_pct(k, N)}% |")
    k = sum(1 for x in rm if x is None)
    L.append(f"| none | {k} | {_pct(k, N)}% |")
    sg = [r[f"{m}:suffix_gap"] for r in rows if r.get(f"{m}:suffix_gap") is not None]
    if sg:
        L += ["", "## Suffix-gap distribution (num_suffix)", "",
              "Suffix gap = score of the best reading WITH the written suffix minus the best bare number "
              "(nats). *Bare wins* = the audio prefers the number without the suffix (all flagged now); "
              "*suffix wins* = the written suffix fits. Use it to choose a suffix margin.", "",
              "| Gap (nats) | Bare wins | Suffix wins |", "|---|---|---|"]
        for a, b in edges:
            bw = sum(1 for x in sg if x > 0 and a <= x < b)
            sw = sum(1 for x in sg if x <= 0 and a <= -x < b)
            L.append(f"| {a}–{'∞' if b == float('inf') else b} | {bw} ({_pct(bw, len(sg))}%) | {sw} ({_pct(sw, len(sg))}%) |")
        L.append(f"| total | {sum(1 for x in sg if x > 0)} | {sum(1 for x in sg if x <= 0)} |")
    if tot.get("stages_s") or tot.get("scorer"):
        L += ["", "## Timing", "",
              "`load_s` = reading JSON + audio and building the candidate jobs (overlaps with scoring on a "
              "background thread); `score_s` = model time (encoder + decoder/joint + likelihood).", "",
              "| Stage | Seconds |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in (tot.get("stages_s") or {}).items()]
        if tot.get("scorer"):
            L += ["", "Scorer: " + ", ".join(f"{k} {v}" for k, v in tot["scorer"].items())]
    L += ["", "## Output files", "",
          f"All under `{tot.get('out_dir')}`:", "",
          "- `verbalized/<stem>.json` — the input's own schema (transcription / words / sentences); sentence, "
          "word and transcription texts train-normalized with every number replaced by its pick. Use these for "
          "training.",
          "- `verify/<stem>.json` — the file in the output schema with every confident number verbalized; "
          "only the uncertain numbers are still written as digits (rows with status `pending` in `verify.tsv`; "
          "the others are `auto`). Decide them with `indic-digits verifier`; once none is open the file moves to "
          "`verbalized/` and its rows from `verify.tsv` to `verbalized.tsv` (status `verified`).",
          "- `failed/<stem>.json` — the input file, unchanged; `failed.tsv` gives the stage (read_json, "
          "find_audio, read_audio, plan, verbalize) and the error. The verifier can verbalize these too.",
          "- `verbalized.tsv` / `verify.tsv` — one row per number: `file`, `sentence`, `number_index`, "
          "`start_time`, `end_time`, `number`, `class`, `candidates` and `scores` (`|`-separated, NLL, lower = "
          "better), `model_pick`, `model_form`, `model_lang`, `margin`, `review_margin`, `needs_review`, "
          "`file_reasons`, `status` (auto / pending / verified), `final_pick`, `verified_at`, `sentence_text`. "
          "UTF-8 with BOM, opens in Excel.",
          "- `summary.json` — these totals, timing and the config; `report.html` — this page.",
          "", "## Config", "", "The exact settings used for this run (also in `summary.json`).", "",
          "| Key | Value |", "|---|---|"]
    L += [f"| {k} | `{json.dumps(v, ensure_ascii=False)}` |" for k, v in cfg.to_dict().items()
          if k not in ("decoding",)]
    L.append("")
    return "\n".join(L)


def write_report(report_dir, rows, tot, cfg, name, html_name="report.html"):
    os.makedirs(report_dir, exist_ok=True)
    m = f"{cfg.head}/{cfg.unit}"
    with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({**tot, "config": cfg.to_dict()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(report_dir, html_name), "w", encoding="utf-8") as f:
        f.write(page(f"Verbalize — {name}", _markdown(rows, tot, cfg, name),
                     numbers_payload(rows, [m], [], m)))
