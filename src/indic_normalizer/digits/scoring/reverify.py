"""Re-check files that are already in ``verbalized/`` with the CURRENT candidates and model.

A finished file is re-scored from its original input (``<run>/input/<stem>.json`` or
``--data-dir`` / summary.json ``data_dir``) and every number is compared with its stored reading
in verbalized.tsv:

    auto (model-decided)  flagged if the new decision is uncertain (needs_review) or picks a
                          different reading than the stored one
    verified (by hand)    flagged if the model now CONFIDENTLY disagrees: the stored reading is
                          a candidate and scores at least ``margin_threshold`` nats worse than the
                          new pick. A typed reading that is not a candidate is kept as is.

A file with flagged numbers moves to its own folder, ``reverify/`` (kept apart from verify/),
for the verifier: flagged numbers are open again (written as digits, status pending, the old
reading kept in ``previous_pick``); all other numbers keep their stored reading and status. Its
rows move from verbalized.tsv to ``reverify.tsv``, the original input is copied to ``input/`` and
summary.json is updated. Once checked in the verifier the file returns to verbalized/.
``--dry-run`` only writes the report. Every run writes ``reverify_report.tsv`` (one row per
flagged number) and ``reverify_report.json`` (totals) in the run dir.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import List, Optional

from .details import NUMBER_COLS, SEP, append_typed, number_rows, read_tsv, tsv_path, write_tsv
from .pipeline import (_number_details, _round, _run, decide, iter_groups, same_document)

NUMBER_REPORT_COLS = ["file", "sentence", "number_index", "number", "class", "status", "final_pick", "model_pick",
                      "agrees", "margin", "review_margin", "suffix_gap", "suffix_bare", "suffix_with", "human_choice",
                      "flagged", "sentence_text"]
BANDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, 20), (20, float("inf"))]
REPORT_COLS = ["file", "sentence", "number_index", "number", "status_before", "previous_pick", "model_pick",
               "previous_gap", "margin", "review_margin", "reason", "action", "sentence_text"]


def _original(run, stem, data_dir):
    for d in (os.path.join(run, "input"), data_dir):
        p = os.path.join(d, stem + ".json") if d else None
        if p and os.path.exists(p):
            return p
    return None


def _summary(run):
    try:
        with open(os.path.join(run, "summary.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def check_number(row, n, d) -> Optional[str]:
    """Why a stored reading should be looked at again, or None (see module docstring)."""
    prev = row.get("final_pick") or ""
    new = n.cands[d["best"]].text
    if row.get("status") == "verified":
        texts = [c.text for c in n.cands]
        if prev == new or prev not in texts:
            return None
        gap = d["metric"][texts.index(prev)] - d["metric"][d["model_best"]]
        if gap >= d.get("threshold", 1.0):
            return f"model now prefers {new} by {gap:.2f} nats over your {prev}"
        return None
    if d["needs_review"]:
        why = d.get("suffix_issue") or d.get("text_issue") or f"uncertain (review margin {_round(d['review_margin'])})"
        return f"now needs review: {why}"
    if new != prev:
        return f"pick changed: {prev} -> {new}"
    return None


def reverify(run: str, audio_dir: str, scorer, cfg, data_dir: Optional[str] = None,
             files: Optional[List[str]] = None, limit: Optional[int] = None, dry_run: bool = False,
             log=print, refresh: bool = True) -> dict:
    run = os.path.abspath(run)
    vpath = tsv_path(run, "verbalized")
    vrows = read_tsv(vpath)
    by_file = {}
    for r in vrows:
        by_file.setdefault(r["file"], []).append(r)
    stems = [s for s in by_file if os.path.exists(os.path.join(run, "verbalized", s + ".json"))]
    if files:
        want = set(files)
        stems = [s for s in stems if s in want]
    stems = sorted(stems)[:limit]
    data_dir = data_dir or _summary(run).get("data_dir")
    srcs, missing = [], []
    for s in stems:
        p = _original(run, s, data_dir)
        (srcs if p else missing).append(p or s)
    for s in missing:
        log(f"skip {s}: original input not found (input/ or --data-dir)")

    tot = {"run_dir": run, "files_checked": 0, "files_flagged": 0, "numbers_checked": 0, "numbers_flagged": 0,
           "flagged_auto": 0, "flagged_verified": 0, "skipped_no_input": len(missing), "skipped_error": 0,
           "skipped_misaligned": 0, "dry_run": dry_run}
    report, moved, measured, typed = [], [], [], []
    done = 0
    t0 = time.perf_counter()
    for items, _ in iter_groups(srcs, audio_dir, cfg, (cfg.unit,)):
        ok = [it for it in items if "error" not in it]
        try:
            _run(scorer, [p for it in ok for p in it["plans"]], cfg.unit, (cfg.head,))
        except Exception as e:                       # e.g. CUDA OOM: skip this group, go on
            for it in ok:
                it["error"] = f"score: {type(e).__name__}: {e}"
        for it in items:
            stem = it["stem"]
            done += 1
            pre = f"[{done}/{len(srcs)}] {stem}:"     # progress: 687/1155
            if "error" in it:
                tot["skipped_error"] += 1
                log(f"{pre} skipped ({it.get('stage', '')} {it['error']})")
                continue
            rows = {(int(r["sentence"]), int(r["number_index"])): r for r in by_file[stem]}
            plans = it["plans"]
            nums = [(p, k, n) for p in plans for k, n in enumerate(p["numbers"])]
            if len(nums) != len(rows) or any((p["idx"], k) not in rows or rows[(p["idx"], k)].get("number") != n.span.core
                                             for p, k, n in nums):
                tot["skipped_misaligned"] += 1
                log(f"{pre} skipped (numbers in the input no longer match verbalized.tsv)")
                continue
            tot["files_checked"] += 1
            flagged = {}
            details = []
            for p, k, n in nums:
                d = dict(decide(n, cfg.unit, cfg.head, cfg, scorer), threshold=cfg.margin_threshold)
                r = rows[(p["idx"], k)]
                why = check_number(r, n, d)
                tot["numbers_checked"] += 1
                details.append(_number_details(p, n, d, cfg))
                measured.append(_measure(stem, p, k, n, r, d, why))
                tr = _typed_record(stem, p, k, n, r, d)
                if tr:
                    typed.append(tr)
                if why:
                    flagged[(p["idx"], k)] = why
                    tot["numbers_flagged"] += 1
                    tot["flagged_verified" if r.get("status") == "verified" else "flagged_auto"] += 1
                    texts = [c.text for c in n.cands]
                    prev = r.get("final_pick") or ""
                    gap = (d["metric"][texts.index(prev)] - d["metric"][d["model_best"]]) if prev in texts else None
                    report.append({"file": stem, "sentence": p["idx"], "number_index": k, "number": n.span.core,
                                   "status_before": r.get("status"), "previous_pick": prev,
                                   "model_pick": n.cands[d["best"]].text, "previous_gap": _round(gap),
                                   "margin": _round(d["margin"]), "review_margin": _round(d["review_margin"]),
                                   "reason": why, "action": "dry-run" if dry_run else "-> reverify",
                                   "sentence_text": p["sentence"].get("text") or ""})
            if not flagged:
                log(f"{pre} ok ({len(nums)} numbers)")
                continue
            tot["files_flagged"] += 1
            log(f"{pre} {len(flagged)} of {len(nums)} number(s) to check again"
                + (" (dry run)" if dry_run else " -> reverify/"))
            if dry_run:
                continue
            # the verify copy: flagged numbers open (as written), the rest as stored
            keep = {}
            for p in plans:
                if p["numbers"]:
                    keep[p["idx"]] = [n.span.text if (p["idx"], k) in flagged else rows[(p["idx"], k)]["final_pick"]
                                      for k, n in enumerate(p["numbers"])]
            out, _, _ = same_document(it["doc"], plans, cfg, choices=keep)
            reason = f"reverify: {len(flagged)} number(s) to check again"
            new_rows = number_rows(stem, plans, details, "verify", [reason])
            for nr in new_rows:
                key = (int(nr["sentence"]), int(nr["number_index"]))
                old = rows[key]
                if key in flagged:
                    nr.update(status="pending", final_pick="", verified_at="", previous_pick=old.get("final_pick") or "",
                              reverify_note=flagged[key])
                else:
                    nr.update(status=old.get("status") or "auto", final_pick=old.get("final_pick") or "",
                              verified_at=old.get("verified_at") or "", previous_pick="", reverify_note="")
            os.makedirs(os.path.join(run, "reverify"), exist_ok=True)
            os.makedirs(os.path.join(run, "input"), exist_ok=True)
            # new state first (reverify copy, input copy, rows in reverify.tsv), old state last
            _write_json_atomic(os.path.join(run, "reverify", stem + ".json"), out)
            dst = os.path.join(run, "input", stem + ".json")
            if os.path.abspath(it["src"]) != os.path.abspath(dst):
                shutil.copyfile(it["src"], dst)
            vp_ = tsv_path(run, "reverify")
            write_tsv(vp_, NUMBER_COLS, [r for r in read_tsv(vp_) if r.get("file") != stem] + new_rows)
            write_tsv(vpath, NUMBER_COLS, [r for r in read_tsv(vpath) if r.get("file") != stem])
            try:
                os.remove(os.path.join(run, "verbalized", stem + ".json"))
            except OSError:
                pass
            moved.append(stem)

    tot["seconds"] = round(time.perf_counter() - t0, 2)
    tot["files_moved_to_reverify"] = moved
    if refresh:                                   # files still waiting in verify/ and reverify/
        tot["waiting"] = refresh_waiting(run, audio_dir, scorer, cfg, data_dir=data_dir, files=files,
                                         dry_run=dry_run, skip=set(moved), log=log)
    tot["measure"] = measure_summary(measured)
    tot["typed_readings_found"] = len(typed)
    tot["typed_readings_added"] = append_typed(run, typed)     # hand-typed readings -> typed_readings.tsv
    write_tsv(os.path.join(run, "reverify_numbers.tsv"), NUMBER_REPORT_COLS, measured)
    from .html_report import page
    with open(os.path.join(run, "reverify_report.html"), "w", encoding="utf-8") as f:
        f.write(page(f"Reverify — {os.path.basename(run)}", _markdown(tot)))
    write_tsv(os.path.join(run, "reverify_report.tsv"), REPORT_COLS, report)   # (dict rows)
    _write_json_atomic(os.path.join(run, "reverify_report.json"), tot)
    if moved:
        s = _summary(run)
        if s:
            s["verbalized"] = max(0, int(s.get("verbalized") or 0) - len(moved))
            s["reverify"] = int(s.get("reverify") or 0) + len(moved)
            _write_json_atomic(os.path.join(run, "summary.json"), s)
    return tot


# ----------------------------------------------------------------------------- measuring
# How the CURRENT model compares with what is stored — above all with readings a person verified:
# per review-margin band, how often the model's pick equals the human's; and for num_suffix, per
# suffix-gap band, whether the human kept the suffix, dropped it or typed something else.

def _measure(stem, p, k, n, r, d, why):
    tag = "+nosuffix"
    final = r.get("final_pick") or ""
    choice = ""
    if d.get("suffix_gap") is not None:
        bare = {c.text for c in n.cands if c.form.endswith(tag)}
        withs = {c.text for c in n.cands if not c.form.endswith(tag)}
        choice = "bare" if final in bare else ("suffixed" if final in withs else "typed")
    pick = n.cands[d["best"]].text
    return {"file": stem, "sentence": p["idx"], "number_index": k, "number": n.span.core, "class": n.span.cls,
            "status": r.get("status"), "final_pick": final, "model_pick": pick, "agrees": pick == final,
            "margin": _round(d["margin"]), "review_margin": _round(d["review_margin"]),
            "suffix_gap": _round(d.get("suffix_gap")), "suffix_bare": d.get("suffix_bare") or "",
            "suffix_with": d.get("suffix_with") or "", "human_choice": choice, "flagged": bool(why),
            "sentence_text": p["sentence"].get("text") or ""}


def _typed_record(stem, p, k, n, r, d):
    """A verified reading that is not one of the candidates (stored or current) = typed by a person."""
    final = r.get("final_pick") or ""
    if r.get("status") != "verified" or not final:
        return None
    cands = {c.text for c in n.cands} | set((r.get("candidates") or "").split(SEP))
    if final in cands:
        return None
    return {"saved_at": r.get("verified_at") or "", "file": stem, "kind": "verbalized", "sentence": p["idx"],
            "number_index": k, "number": n.span.core, "class": n.span.cls, "typed": final,
            "model_pick": n.cands[d["best"]].text, "previous_pick": r.get("previous_pick") or "",
            "candidates": SEP.join(c.text for c in n.cands), "start_time": p["sentence"].get("start_time"),
            "end_time": p["sentence"].get("end_time"), "sentence_text": p["sentence"].get("text") or ""}


def _band(x):
    for a, b in BANDS:
        if a <= x < b:
            return f"{a}–{'∞' if b == float('inf') else b}"
    return "?"


def measure_summary(rows):
    out = {}
    for st in ("verified", "auto"):
        sel = [r for r in rows if r["status"] == st]
        bands = []
        for a, b in BANDS + [(None, None)]:
            if a is None:
                x = [r for r in sel if r["review_margin"] is None]
                label = "none"
            else:
                x = [r for r in sel if r["review_margin"] is not None and a <= r["review_margin"] < b]
                label = _band(a)
            agree = sum(1 for r in x if r["agrees"])
            bands.append({"band": label, "n": len(x), "agrees": agree,
                          "pct": round(100 * agree / len(x), 1) if x else None})
        out[f"{st}_by_review_margin"] = bands
        out[f"{st}_numbers"] = len(sel)
        out[f"{st}_agrees"] = sum(1 for r in sel if r["agrees"])
    suf = []
    sel = [r for r in rows if r["status"] == "verified" and r["suffix_gap"] is not None]
    for a, b in BANDS:
        row = {"band": _band(a)}
        for side, test in (("bare_wins", lambda g: g > 0), ("suffix_wins", lambda g: g <= 0)):
            x = [r for r in sel if test(r["suffix_gap"]) and a <= abs(r["suffix_gap"]) < b]
            row[side] = {"n": len(x), "human_bare": sum(r["human_choice"] == "bare" for r in x),
                         "human_suffixed": sum(r["human_choice"] == "suffixed" for r in x),
                         "human_typed": sum(r["human_choice"] == "typed" for r in x)}
        suf.append(row)
    out["verified_num_suffix_by_suffix_gap"] = suf
    out["verified_num_suffix"] = len(sel)
    return out


def _markdown(tot):
    m = tot["measure"]
    L = [f"# Reverify — `{os.path.basename(tot['run_dir'])}`", "",
         f"Files checked **{tot['files_checked']}**, numbers **{tot['numbers_checked']}**, flagged "
         f"**{tot['numbers_flagged']}** ({tot['flagged_auto']} auto, {tot['flagged_verified']} verified) in "
         f"{tot['files_flagged']} files" + (" — dry run, nothing moved." if tot["dry_run"] else
                                             f" → `reverify/` ({len(tot['files_moved_to_reverify'])} files)."), "",
         "Every checked number is in `reverify_numbers.tsv` (stored reading, the model's pick now, margins, "
         "suffix gap, what the person chose); flagged ones with the reason in `reverify_report.tsv`.", ""]
    w = tot.get("waiting")
    if w:
        L += [f"Files waiting in `verify/` and `reverify/`: **{w['files']}** re-scored — {w['numbers_refreshed']} "
              f"open numbers now have the current candidates and scores ({w['picks_changed']} with a new model pick)"
              + (" — dry run, not written." if w["dry_run"] else ".") +
              f" Skipped: {w['skipped_no_input']} without input, {w['skipped_misaligned']} misaligned, "
              f"{w['skipped_error']} errors.", ""]
    for st, what in (("verified", "verified by hand"), ("auto", "decided automatically")):
        n = m[f"{st}_numbers"]
        if not n:
            continue
        L += [f"## Model vs readings {what}", "",
              f"{n} numbers; the model's pick now equals the stored reading for **{m[f'{st}_agrees']}** "
              f"({round(100 * m[f'{st}_agrees'] / n, 1)}%). Per review-margin band (how sure the model is now):", "",
              "| Review margin (nats) | Numbers | Model = stored | % |", "|---|---|---|---|"]
        L += [f"| {b['band']} | {b['n']} | {b['agrees']} | {'' if b['pct'] is None else str(b['pct']) + '%'} |"
              for b in m[f"{st}_by_review_margin"] if b["n"]]
        L.append("")
    if m["verified_num_suffix"]:
        L += ["## Suffix gap vs your verified readings (num_suffix)", "",
              "Suffix gap = best reading WITH the written suffix minus the best bare number (nats). For each band: "
              "how many numbers, and whether YOU kept the suffix, dropped it (bare) or typed something else. If "
              "bare wins above some gap are mostly *bare* for you, that gap is a safe suffix margin.", "",
              "| Gap (nats) | Bare wins | you: bare / suffixed / typed | Suffix wins | you: bare / suffixed / typed |",
              "|---|---|---|---|---|"]
        for r in m["verified_num_suffix_by_suffix_gap"]:
            bw, sw = r["bare_wins"], r["suffix_wins"]
            if bw["n"] or sw["n"]:
                L.append(f"| {r['band']} | {bw['n']} | {bw['human_bare']} / {bw['human_suffixed']} / {bw['human_typed']} | "
                         f"{sw['n']} | {sw['human_bare']} / {sw['human_suffixed']} / {sw['human_typed']} |")
    return "\n".join(L)


# ----------------------------------------------------------------------------- waiting files
# Files in verify/ and reverify/ wait for the verifier with the candidates and scores of their
# last scoring. refresh_waiting() re-scores their OPEN (pending) numbers with the current
# candidates and model, so the verifier offers the new readings (Latin-script for English
# sentences, new forms) with fresh scores. Decided numbers (auto / verified) and the JSON copies
# are not touched.

REFRESH_FIELDS = ["candidates", "scores", "model_pick", "model_form", "model_lang", "audio_pick", "style_gap",
                  "style_fallback", "suffix_issue", "suffix_gap", "text_issue", "margin", "review_margin",
                  "needs_review"]


def refresh_waiting(run: str, audio_dir: str, scorer, cfg, data_dir: Optional[str] = None,
                    files: Optional[List[str]] = None, dry_run: bool = False, skip=(), log=print) -> dict:
    run = os.path.abspath(run)
    data_dir = data_dir or _summary(run).get("data_dir")
    want = set(files) if files else None
    tables = {k: read_tsv(tsv_path(run, k)) for k in ("verify", "reverify")}
    kind_of, rows_of = {}, {}
    for k, rows in tables.items():
        for r in rows:
            stem = r["file"]
            if stem in skip or (want and stem not in want):
                continue
            if not os.path.exists(os.path.join(run, k, stem + ".json")):
                continue
            kind_of[stem] = k
            rows_of.setdefault(stem, []).append(r)
    stems = sorted(s for s in rows_of if any(r.get("status") == "pending" for r in rows_of[s]))
    srcs, tot = [], {"files": 0, "numbers_refreshed": 0, "picks_changed": 0, "skipped_no_input": 0,
                     "skipped_error": 0, "skipped_misaligned": 0, "dry_run": dry_run}
    for s in stems:
        p = _original(run, s, data_dir)
        if p:
            srcs.append(p)
        else:
            tot["skipped_no_input"] += 1
            log(f"waiting {s}: skipped (original input not found)")
    done = 0
    for items, _ in iter_groups(srcs, audio_dir, cfg, (cfg.unit,)):
        ok = [it for it in items if "error" not in it]
        try:
            _run(scorer, [p for it in ok for p in it["plans"]], cfg.unit, (cfg.head,))
        except Exception as e:
            for it in ok:
                it["error"] = f"score: {type(e).__name__}: {e}"
        for it in items:
            stem = it["stem"]
            done += 1
            pre = f"[waiting {done}/{len(srcs)}] {stem} ({kind_of.get(stem)}):"
            if "error" in it:
                tot["skipped_error"] += 1
                log(f"{pre} skipped ({it['error']})")
                continue
            rows = {(int(r["sentence"]), int(r["number_index"])): r for r in rows_of[stem]}
            plans = it["plans"]
            nums = [(p, k, n) for p in plans for k, n in enumerate(p["numbers"])]
            if len(nums) != len(rows) or any((p["idx"], k) not in rows or rows[(p["idx"], k)].get("number") != n.span.core
                                             for p, k, n in nums):
                tot["skipped_misaligned"] += 1
                log(f"{pre} skipped (numbers in the input no longer match the TSV)")
                continue
            fresh = number_rows(stem, plans, [_number_details(p, n, decide(n, cfg.unit, cfg.head, cfg, scorer), cfg)
                                              for p, k, n in nums], "verify", [])
            n_ref = n_chg = 0
            for fr in fresh:
                r = rows[(int(fr["sentence"]), int(fr["number_index"]))]
                if r.get("status") != "pending":
                    continue                       # decided numbers stay as they are
                n_chg += fr.get("model_pick") != r.get("model_pick")
                for f in REFRESH_FIELDS:
                    r[f] = fr.get(f)
                n_ref += 1
            tot["files"] += 1
            tot["numbers_refreshed"] += n_ref
            tot["picks_changed"] += n_chg
            log(f"{pre} {n_ref} open number(s) re-scored, {n_chg} with a new pick" + (" (dry run)" if dry_run else ""))
    if not dry_run and tot["files"]:
        for k, rows in tables.items():            # rows were updated in place
            write_tsv(tsv_path(run, k), NUMBER_COLS, rows)
    return tot
