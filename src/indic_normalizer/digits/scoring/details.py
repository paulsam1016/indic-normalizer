"""Details files of a verbalize run (OUT_DIR/<data dir>/{verbalized,verify,failed}.tsv).

TSV, UTF-8 with BOM (Excel shows Malayalam), one row per NUMBER for verbalized / verify and one
row per FILE for failed. In verify.tsv only the uncertain numbers are "pending"; the confident ones
are "auto" (already verbalized in verify/<stem>.json). The verifier turns pending rows into
"verified"; once a file has no pending number it is written to verbalized/, its rows move from
verify.tsv to verbalized.tsv and the copy in verify/ is removed. A failed file that is partly
decided becomes a verify file (number rows in verify.tsv).
Pure python: no numpy / torch, so the verifier runs anywhere."""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List

NUMBER_COLS = ["file", "sentence", "number_index", "start_time", "end_time", "number", "class",
               "candidates", "scores", "model_pick", "model_form", "model_lang", "audio_pick", "style_gap",
               "style_fallback", "suffix_issue", "suffix_gap", "text_issue", "margin", "review_margin",
               "needs_review", "file_reasons", "status", "final_pick", "verified_at", "sentence_text",
               "previous_pick", "reverify_note"]          # set by `reverify` for numbers opened again
FAILED_COLS = ["file", "source", "stage", "error", "status", "final_picks", "verified_at"]
COLS = {"verbalized": NUMBER_COLS, "verify": NUMBER_COLS, "failed": FAILED_COLS,
        "reverify": NUMBER_COLS}                 # reverify.tsv: files sent back by `reverify`
SEP = " | "

# status: "auto" (model, confident) | "pending" (uncertain, waiting) | "verified" (a person chose)


def tsv_path(run_dir: str, status: str) -> str:
    return os.path.join(run_dir, f"{status}.tsv")


def read_tsv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_tsv(path: str, cols: List[str], rows: Iterable[dict]) -> None:
    """Atomic write: temp file + rename, so a crash never leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_clean(r.get(c)) for c in cols])
    os.replace(tmp, path)


def merge(run_dir: str, new_rows: Dict[str, List[dict]], files: Iterable[str]) -> None:
    """Rewrite the three TSVs: rows of ``files`` (re-processed now) are dropped everywhere and
    replaced by ``new_rows``; rows of other files (earlier runs, --limit) are kept."""
    files = set(files)
    for st, cols in COLS.items():
        p = tsv_path(run_dir, st)
        keep = [r for r in read_tsv(p) if r.get("file") not in files]
        write_tsv(p, cols, keep + list(new_rows.get(st, [])))


def number_rows(stem: str, plans, nums: List[dict], status: str, reasons: List[str]) -> List[dict]:
    """``nums`` from pipeline._number_details, in plan order."""
    rows, it = [], iter(nums)
    for p in plans:
        for j, _ in enumerate(p["numbers"]):
            x = next(it)
            pend = status != "verbalized" and x["needs_review"]   # only uncertain numbers wait
            head = [k for k in x["candidates"][0] if k.startswith("nll_")] if x["candidates"] else []
            rows.append({"file": stem, "sentence": x["sentence"], "number_index": j,
                         "start_time": x["start_time"], "end_time": x["end_time"], "number": x["core"],
                         "class": x["class"], "candidates": SEP.join(c["text"] for c in x["candidates"]),
                         "scores": SEP.join("" if c.get(head[0]) is None else str(c[head[0]])
                                            for c in x["candidates"]) if head else "",
                         "model_pick": x["chosen"], "model_form": x["form"], "model_lang": x["lang"],
                         "audio_pick": x.get("model_pick"), "style_gap": x.get("style_gap"),
                         "style_fallback": x.get("style_fallback"),
                         "suffix_issue": x.get("suffix_issue") or "",
                         "text_issue": x.get("text_issue") or "",
                         "suffix_gap": x.get("suffix_gap"),
                         "margin": x["margin"], "review_margin": x["review_margin"],
                         "needs_review": x["needs_review"], "file_reasons": "; ".join(reasons),
                         "status": "pending" if pend else "auto",
                         "final_pick": "" if pend else x["chosen"],
                         "verified_at": "", "sentence_text": p["sentence"].get("text") or ""})
    return rows


TYPED_COLS = ["saved_at", "file", "kind", "sentence", "number_index", "number", "class", "typed", "model_pick",
              "previous_pick", "candidates", "start_time", "end_time", "sentence_text"]


def append_typed(run_dir: str, recs: List[dict]) -> int:
    """Append readings a person TYPED (not one of the candidates) to <run>/typed_readings.tsv —
    data for new candidate rules. Only appended, never rewritten; a reading already in the file
    (same file, sentence, number_index, typed text) is not added twice. Returns rows added."""
    if not recs:
        return 0
    p = os.path.join(run_dir, "typed_readings.tsv")
    have = {(r.get("file"), str(r.get("sentence")), str(r.get("number_index")), r.get("typed"))
            for r in read_tsv(p)} if os.path.exists(p) else set()
    new_file = not os.path.exists(p)
    added = 0
    with open(p, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if new_file:
            w.writerow(TYPED_COLS)
        for r in recs:
            key = (r.get("file"), str(r.get("sentence")), str(r.get("number_index")), r.get("typed"))
            if key in have:
                continue
            have.add(key)
            w.writerow([_clean(r.get(c)) for c in TYPED_COLS])
            added += 1
    return added


DECISION_COLS = ["saved_at", "file", "kind", "action", "sentence", "number_index", "number", "class", "chosen",
                 "choice", "chosen_rank", "chosen_delta", "model_pick", "audio_pick", "previous_pick",
                 "margin", "review_margin", "suffix_gap", "suffix_issue", "text_issue", "n_candidates",
                 "candidates", "scores", "start_time", "end_time", "sentence_text"]


def append_decisions(run_dir: str, recs: List[dict]) -> int:
    """Append every reading saved in the verifier to <run>/verifier_decisions.tsv (one row per
    number per save; never rewritten) — to study how people differ from the model."""
    if not recs:
        return 0
    p = os.path.join(run_dir, "verifier_decisions.tsv")
    new_file = not os.path.exists(p)
    with open(p, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if new_file:
            w.writerow(DECISION_COLS)
        for r in recs:
            w.writerow([_clean(r.get(c)) for c in DECISION_COLS])
    return len(recs)


def decision_record(saved_at, file, kind, action, row: dict, chosen: str, cand_texts=None,
                    sentence=None, previous=None) -> dict:
    """One verifier decision: what was chosen and how it relates to the model's candidates and
    scores (``row`` = the number's TSV row; ``cand_texts`` when the row has none, e.g. failed/)."""
    cands = [c for c in (row.get("candidates") or "").split(SEP) if c] or list(cand_texts or [])
    raw = (row.get("scores") or "").split(SEP) if row.get("scores") else []
    scores = []
    for x in raw:
        try:
            scores.append(float(x))
        except ValueError:
            scores.append(None)
    model = row.get("model_pick") or ""
    choice = "model_pick" if model and chosen == model else ("candidate" if chosen in cands else "typed")
    rank = delta = None
    if chosen in cands:
        i = cands.index(chosen)
        if i < len(scores) and scores[i] is not None:
            valid = sorted(v for v in scores if v is not None)
            rank = 1 + valid.index(scores[i])
            delta = round(scores[i] - valid[0], 4)
        else:
            rank = i + 1
    s = sentence or {}
    return {"saved_at": saved_at, "file": file, "kind": kind, "action": action,
            "sentence": row.get("sentence"), "number_index": row.get("number_index"),
            "number": row.get("number"), "class": row.get("class"), "chosen": chosen, "choice": choice,
            "chosen_rank": rank, "chosen_delta": delta, "model_pick": model, "audio_pick": row.get("audio_pick"),
            "previous_pick": previous if previous is not None else (row.get("previous_pick") or ""),
            "margin": row.get("margin"), "review_margin": row.get("review_margin"),
            "suffix_gap": row.get("suffix_gap"), "suffix_issue": row.get("suffix_issue"),
            "text_issue": row.get("text_issue"), "n_candidates": len(cands), "candidates": SEP.join(cands),
            "scores": row.get("scores") or "", "start_time": row.get("start_time", s.get("start_time")),
            "end_time": row.get("end_time", s.get("end_time")),
            "sentence_text": row.get("sentence_text") or s.get("text") or ""}
