"""``indic-digits verifier``: check the numbers ``verbalize`` was unsure about, in the browser.

A small local web app (Python standard library only — no GPU, torch or numpy; Windows and Linux):
a server on 127.0.0.1 serves one HTML page and a JSON API that reuses the verbalize code, so a
verified file is written exactly as ``verbalize`` would have written it.

Source: a verbalize run dir OUT_DIR/<data dir>/ (verify/, failed/, verify.tsv, failed.tsv).
verify/<stem>.json already has every confident number verbalized; only the uncertain ("pending")
numbers are still written as digits, and only those are shown. Saving:
    chosen readings applied; rows -> status "verified", final_pick, verified_at
    numbers left for later stay open (the copy is updated in verify/; a failed file moves there)
    no open number left -> verbalized/<stem>.json written, the copy removed, the file's rows moved
                           from verify.tsv to verbalized.tsv, summary.json counts updated
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import string
import sys
import threading
import webbrowser
from dataclasses import fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..scoring.config import ScoringConfig
from ..scoring.details import COLS, NUMBER_COLS, SEP, read_tsv, tsv_path, write_tsv
from ..scoring.pipeline import plan_text, same_document
from .. import ml
from .ui import PAGE


def _text_suffix_issue(n):
    """Suffix check that needs no audio (failed files / old runs)."""
    return ml.suffix_issue(n.span.core) if n.span.cls == "num_suffix" else None

SETTINGS = os.path.join(os.path.expanduser("~"), ".indic_digits_verifier.json")
_LOCK = threading.Lock()                  # one save at a time


# ----------------------------------------------------------------------------- settings / run

def _load_settings():
    try:
        return json.load(open(SETTINGS, encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(d):
    try:
        tmp = SETTINGS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SETTINGS)
    except OSError:
        pass


def is_run_dir(p: str) -> bool:
    return bool(p) and os.path.isdir(p) and any(
        os.path.exists(os.path.join(p, x)) for x in ("verify.tsv", "failed.tsv", "verify", "failed",
                                                     "reverify.tsv", "reverify"))


KINDS = ("verify", "reverify", "failed")      # folders the verifier works on
from ..scoring.details import TYPED_COLS  # noqa: E402,F401  (columns of typed_readings.tsv)
from ..scoring.details import append_decisions, decision_record  # noqa: E402


def _rows_kind(kind):
    """TSV holding a kind's number rows: reverify files -> reverify.tsv, the rest -> verify.tsv."""
    return "reverify" if kind == "reverify" else "verify"


def _log_typed(run, recs):
    """Typed readings -> <run>/typed_readings.tsv (details.append_typed)."""
    from ..scoring.details import append_typed
    append_typed(run, recs)


def _summary(run):
    try:
        return json.load(open(os.path.join(run, "summary.json"), encoding="utf-8"))
    except Exception:
        return {}


def run_config(run) -> ScoringConfig:
    """The run's own settings (style, max_candidates, ...) so candidates match verbalize."""
    cfg = (_summary(run).get("config") or {})
    known = {f.name for f in fields(ScoringConfig)}
    return ScoringConfig(**{k: v for k, v in cfg.items() if k in known})


def _safe_stem(name: str) -> str:
    stem = os.path.basename(str(name or ""))
    if stem.endswith(".json"):
        stem = stem[:-5]
    if not stem or stem in (".", "..") or re.search(r'[\\/:*?"<>|]', stem):
        raise ValueError(f"bad file name {name!r}")
    return stem


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _find_wav(audio_dir, stem):
    for ext in (".wav", ".WAV"):
        p = os.path.join(audio_dir or "", stem + ext)
        if audio_dir and os.path.exists(p):
            return p
    return None


# ----------------------------------------------------------------------------- API

def api_state(st):
    run, audio = st.get("run_dir"), st.get("audio_dir")
    out = {"run_dir": run, "audio_dir": audio, "data_dir": st.get("data_dir") or "", "recent": st.get("recent", []), "loaded": is_run_dir(run),
           "settings_file": SETTINGS}
    if out["loaded"]:
        s = _summary(run)
        out["summary"] = {k: s.get(k) for k in ("files", "verbalized", "verify", "reverify", "failed", "verified", "numbers",
                                                "needs_review", "data_dir", "audio_dir")}
        out["audio_ok"] = bool(audio) and os.path.isdir(audio)
    return out


def api_open(st, body):
    run = os.path.abspath(os.path.expanduser(str(body.get("run_dir") or "").strip()))
    audio = str(body.get("audio_dir") or "").strip()
    audio = os.path.abspath(os.path.expanduser(audio)) if audio else ""
    if not is_run_dir(run):
        raise ValueError(f"{run} is not a verbalize run dir (needs verify.tsv / failed.tsv / verify/ / failed/)")
    if audio and not os.path.isdir(audio):
        raise ValueError(f"audio dir {audio} does not exist")
    data = str(body.get("data_dir") or "").strip()
    data = os.path.abspath(os.path.expanduser(data)) if data else ""
    if data and not os.path.isdir(data):
        raise ValueError(f"input dir {data} does not exist")
    st["run_dir"], st["audio_dir"], st["data_dir"] = run, audio, data
    rec = [r for r in st.get("recent", []) if r.get("run_dir") != run]
    st["recent"] = [{"run_dir": run, "audio_dir": audio, "data_dir": data}] + rec[:9]
    _save_settings(st)
    return api_state(st)


def api_browse(path):
    """Folder picker: subdirectories of ``path`` (drives on Windows when empty)."""
    if not path:
        if os.name == "nt":
            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            return {"path": "", "parent": None, "dirs": drives, "is_run": False}
        path = os.path.expanduser("~")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a folder")
    try:
        dirs = sorted((d for d in os.listdir(path) if not d.startswith(".") and os.path.isdir(os.path.join(path, d))),
                      key=str.lower)
    except PermissionError:
        dirs = []
    parent = os.path.dirname(path.rstrip("\\/")) or None
    if parent == path:
        parent = "" if os.name == "nt" else None
    n_wav = 0
    try:
        n_wav = sum(1 for f in os.listdir(path) if f.lower().endswith(".wav"))
    except OSError:
        pass
    return {"path": path, "parent": parent, "dirs": dirs, "is_run": is_run_dir(path), "n_wav": n_wav}


def api_files(st):
    run = st["run_dir"]
    _empty_trash(run)
    files = {}                                        # (kind, stem) -> entry: both kinds can exist
    by_kind = {}
    for kind in ("verify", "reverify"):
        by_file = by_kind[kind] = {}
        for r in read_tsv(tsv_path(run, kind)):
            by_file.setdefault(r["file"], []).append(r)
            f = files.setdefault((kind, r["file"]), {"file": r["file"], "kind": kind, "n_numbers": 0,
                                                     "n_pending": 0, "reasons": r.get("file_reasons", "")})
            f["n_numbers"] += 1
            f["n_pending"] += r.get("status") == "pending"
    for r in read_tsv(tsv_path(run, "failed")):
        files[("failed", r["file"])] = {"file": r["file"], "kind": "failed", "stage": r.get("stage"),
                                        "error": r.get("error")}
    for kind in KINDS:                                # copies without rows (TSV edited by hand) still show up
        d = os.path.join(run, kind)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                stem = fn[:-5]
                if not fn.endswith(".json") or (kind, stem) in files:
                    continue
                if _leftover(run, stem, kind, by_kind.get(_rows_kind(kind), {})):
                    _remove(os.path.join(d, fn))
                    continue
                files[(kind, stem)] = {"file": stem, "kind": kind}
    out = []
    for f in files.values():
        f["has_copy"] = os.path.exists(os.path.join(run, f["kind"], f["file"] + ".json"))
        f["has_audio"] = _find_wav(st.get("audio_dir"), f["file"]) is not None
        out.append(f)
    out.sort(key=lambda f: (KINDS.index(f["kind"]), -f.get("n_pending", 0), f["file"]))
    return {"files": out, "counts": {k: sum(f["kind"] == k for f in out) for k in KINDS}}


def _file_rows(run, stem, kind="verify"):
    return [r for r in read_tsv(tsv_path(run, _rows_kind(kind))) if r.get("file") == stem]


def _pending_by_sentence(rows):
    """{sentence: [pending rows in number order]} — they line up, in order, with the numbers still
    written as digits in the verify copy."""
    out = {}
    for r in sorted((r for r in rows if r.get("status") == "pending"),
                    key=lambda r: (int(r.get("sentence") or 0), int(r.get("number_index") or 0))):
        out.setdefault(int(r["sentence"]), []).append(r)
    return out


def _load(run, stem, kind):
    src, doc, _ = _load_rev(run, stem, kind)
    return src, doc


def _load_rev(run, stem, kind):
    """(path, document, rev): rev is a hash of the file's bytes, so a save can tell whether the
    copy changed since the page was opened (a second Save, another tab)."""
    if kind not in KINDS:
        raise ValueError("kind must be verify, reverify or failed")
    src = os.path.join(run, kind, stem + ".json")
    if not os.path.exists(src):
        raise ValueError(f"{kind}/{stem}.json is missing (already verified?)")
    with open(src, "rb") as f:
        raw = f.read()
    return src, json.loads(raw.decode("utf-8-sig")), hashlib.sha1(raw).hexdigest()[:16]


class Misaligned(ValueError):
    pass


def _open_numbers(plans, rows, kind):
    """The numbers still open in a copy, each with a stable id and its TSV row:
    [(id, sentence, k, number, row)]. In a verify copy the open numbers are exactly the ones
    still written as digits, and they line up (in order, per sentence) with the pending rows;
    if they don't (copy or TSV edited by hand) -> Misaligned, never a silent mismatch."""
    has_rows = kind != "failed"
    pend = _pending_by_sentence(rows) if has_rows else {}
    out = []
    for p in plans:
        lst = pend.get(p["idx"]) or []
        if has_rows and len(lst) != len(p["numbers"]):
            raise Misaligned(f"sentence {p['idx']}: {len(p['numbers'])} open number(s) in the copy but "
                             f"{len(lst)} pending row(s) in verify.tsv")
        for k, n in enumerate(p["numbers"]):
            r = lst[k] if has_rows else None
            if r is not None and r.get("number") != n.span.core:
                raise Misaligned(f"sentence {p['idx']}: copy has {n.span.core}, verify.tsv has {r.get('number')}")
            nid = f"{p['idx']}:{r.get('number_index')}" if r is not None else f"{p['idx']}:f{k}"
            out.append((nid, p["idx"], k, n, r))
    return out


def _given_from(body, plans, opened):
    """Choices as {sentence: [reading per open number]} from either ``picks`` ({id: reading})
    or the positional ``choices``."""
    if body.get("picks") is not None:
        picks = body["picks"]
        unknown = set(picks) - {o[0] for o in opened}
        if unknown:
            raise ValueError(f"unknown number id(s) {sorted(unknown)[:3]} — reload the file")
        given = {}
        for nid, si, k, n, _ in opened:
            given.setdefault(si, [None] * len([o for o in opened if o[1] == si]))[k] = picks.get(nid)
        return given
    return {int(k): v for k, v in (body.get("choices") or {}).items()}


_DIGIT = re.compile(r"\d")


def _original(run, stem, st=None):
    """The INPUT file for ``stem``, or None. Looked up in: <run>/input/ (copy kept by verbalize for
    verify files), the Input folder set in the verifier, summary.json data_dir (the machine
    verbalize ran on — often a different path, e.g. Linux vs Windows)."""
    dirs = [os.path.join(run, "input"), (st or {}).get("data_dir"), _summary(run).get("data_dir")]
    for d in dirs:
        p = os.path.join(d, stem + ".json") if d else None
        if p and os.path.exists(p):
            return p
    return None


def _decided_rows(mine):
    return [r for r in mine if r.get("status") in ("auto", "verified")]


def _rebuild(run, stem, cfg, mine, picks, edits, st=None):
    """Rebuild the file from the ORIGINAL input with a reading for every number: edits for
    already-decided numbers, picks for open ones, else the TSV's final pick (open numbers without
    a pick stay as written). Used when decided numbers are edited — the copy no longer has their
    digits. Returns (doc, plans, choices, changes [(row, text)], n_open)."""
    src = _original(run, stem, st)
    if not src:
        raise ValueError("the original input file was not found: set the Input folder (the data dir "
                         "verbalize read), so decided numbers can be changed")
    with open(src, encoding="utf-8-sig") as f:
        doc = json.load(f)
    plans = plan_text(doc, cfg)
    rows = {}
    for r in mine:
        try:
            rows[(int(r["sentence"]), int(r["number_index"]))] = r
        except (KeyError, TypeError, ValueError):
            raise Misaligned(f"bad row in verify.tsv: {r.get('sentence')}/{r.get('number_index')}")
    full, changes, later, n_total = {}, [], 0, 0
    for p in plans:
        if not p["numbers"]:
            continue
        texts = []
        for k, n in enumerate(p["numbers"]):
            n_total += 1
            rid, r = f"{p['idx']}:{k}", rows.get((p["idx"], k))
            if r is None or r.get("number") != n.span.core:
                raise Misaligned(f"sentence {p['idx']} number {k}: input has {n.span.core}, "
                                 f"verify.tsv has {(r or {}).get('number')}")
            pending = r.get("status") == "pending"
            t = (picks or {}).get(rid) if pending else (edits or {}).get(rid)
            t = " ".join(str(t).split()) if t else ""
            if t and _DIGIT.search(t):
                raise ValueError(f"reading for {n.span.core} contains digits ({t!r}): write it in words")
            if t and not (not pending and t == r.get("final_pick")):
                changes.append((r, t))
                texts.append(t)
            elif pending:
                texts.append(n.span.text)
                later += 1
            else:
                texts.append(r.get("final_pick") or n.span.text)
        full[p["idx"]] = texts
    if n_total != len(rows):
        raise Misaligned(f"input has {n_total} numbers, verify.tsv has {len(rows)} rows")
    return doc, plans, full, changes, later


def api_file(st, stem, kind):
    run = st["run_dir"]
    stem = _safe_stem(stem)
    try:
        src, doc, rev = _load_rev(run, stem, kind)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return {"file": stem, "kind": kind, "error": f"cannot read the JSON: {e}", "sentences": []}
    cfg = run_config(run)
    rows = _file_rows(run, stem, kind) if kind != "failed" else []
    plans = plan_text(doc, cfg)
    try:
        opened = _open_numbers(plans, rows, kind)
    except Misaligned as e:
        return {"file": stem, "kind": kind, "sentences": [], "rev": rev,
                "error": f"the copy and verify.tsv don't line up ({e}); re-run verbalize for this file"}
    by_sent = {}
    for o in opened:
        by_sent.setdefault(o[1], []).append(o)
    sents = []
    for p in plans:
        if not p["numbers"]:
            continue
        s = p["sentence"]
        nums = []
        for nid, _, k, n, r in by_sent.get(p["idx"], []):   # numbers still written as digits = the open ones
            row = r or {}
            texts = [c.text for c in n.cands]
            scores = [None] * len(texts)
            if r and r.get("candidates"):
                texts = r["candidates"].split(SEP)
                scores = [float(x) if x not in ("", "None") else None for x in r.get("scores", "").split(SEP)]
                scores += [None] * (len(texts) - len(scores))
            forms = {c.text: c.form for c in n.cands}
            best = min((x for x in scores if x is not None), default=None)
            cands = [{"text": t, "form": forms.get(t, ""), "order": i + 1,
                      "delta": (round(sc - best, 3) if sc is not None and best is not None else None)}
                     for i, (t, sc) in enumerate(zip(texts, scores))]
            # most likely first (by score); unscored ones (failed/ files) keep the generation order
            cands.sort(key=lambda c: (c["delta"] is None, c["delta"] if c["delta"] is not None else 0))
            nums.append({"j": k, "id": nid, "core": n.span.core, "class": n.span.cls, "candidates": cands,
                         "model_pick": row.get("model_pick") or texts[0], "scored": bool(row.get("scores")),
                         "margin": row.get("margin"), "review_margin": row.get("review_margin"),
                         "audio_pick": row.get("audio_pick") or "",
                         "style_gap": row.get("style_gap") or "",
                         "style_fallback": str(row.get("style_fallback")) == "True",
                         "suffix_issue": row.get("suffix_issue") or _text_suffix_issue(n) or "",
                         "text_issue": row.get("text_issue") or n.text_issue or "",
                         "previous_pick": row.get("previous_pick") or "", "reverify_note": row.get("reverify_note") or ""})
        sents.append({"idx": p["idx"], "start": s.get("start_time"), "end": s.get("end_time"),
                      "speaker": s.get("speaker"), "text": s.get("text") or "", "numbers": nums})
    if kind == "failed":
        info = next((r for r in read_tsv(tsv_path(run, "failed")) if r.get("file") == stem), {})
    else:
        info = {"reasons": (rows[0] if rows else {}).get("file_reasons", ""),
                "decided": sum(r.get("status") in ("auto", "verified") for r in rows), "total": len(rows)}
    decided = [{"id": f"{r.get('sentence')}:{r.get('number_index')}", "sentence": r.get("sentence"),
                "core": r.get("number"), "class": r.get("class"), "status": r.get("status"),
                "final_pick": r.get("final_pick"), "model_pick": r.get("model_pick"),
                "candidates": [c for c in (r.get("candidates") or "").split(SEP) if c],
                "sentence_text": r.get("sentence_text") or "", "verified_at": r.get("verified_at") or "",
                "start": r.get("start_time"), "end": r.get("end_time")}
               for r in _decided_rows(rows)]
    return {"file": stem, "kind": kind, "rev": rev, "sentences": sents, "info": info, "decided": decided,
            "can_edit_decided": bool(decided) and _original(run, stem, st) is not None,
            "has_audio": _find_wav(st.get("audio_dir"), stem) is not None,
            "n_sentences": len(doc.get("sentences") or [])}


def _full_choices(plans, given):
    """Readings for every number still open; None / empty = decide later (stays as written)."""
    out, decided, later = {}, [], 0
    for p in plans:
        if not p["numbers"]:
            continue
        g = given.get(p["idx"]) or []
        row = []
        for k, n in enumerate(p["numbers"]):
            t = " ".join(str(g[k]).split()) if k < len(g) and g[k] is not None else ""
            if t and _DIGIT.search(t):
                raise ValueError(f"reading for {n.span.core} contains digits ({t!r}): write it in words")
            if t:
                row.append(t)
                decided.append((p["idx"], k, n, t))
            else:
                row.append(n.span.text)
                later += 1
        out[p["idx"]] = row
    return out, decided, later


def api_candidates(st, text, style=None):
    """Candidate generator: every number in ``text`` with all its readings. ``phrase`` = the whole
    input with that reading in place (other numbers default), so '55000 രൂപയുടെ' gives
    'അമ്പത്തയ്യായിരം രൂപയുടെ' ready to use as a custom reading."""
    from .. import candidates as _cands, find as _find, verbalize as _verb
    text = " ".join(str(text or "").split())
    if not text:
        raise ValueError("type a number, e.g. 55000, 250യിൽ, 4:30-ന്, 12.5%")
    cfg = run_config(st["run_dir"]) if is_run_dir(st.get("run_dir")) else ScoringConfig()
    style = style or cfg.style
    spans = list(_find(text))
    if not spans:
        raise ValueError(f"no number found in {text!r}")
    from ..latin import is_english
    eng = is_english(text)                        # English input -> Latin-script readings
    groups = []
    for i, sp in enumerate(spans):
        from .. import next_word as _nw
        cands = _cands(sp, style=style, latin=eng, next_word=_nw(text, sp))
        rows = []
        for k, c in enumerate(cands):
            def pick(span, cs, c=c):
                return c.text if span.start == sp.start else None
            rows.append({"text": c.text, "form": c.form, "lang": c.lang,
                         "phrase": _verb(text, style=style, chooser=pick, latin=eng),
                         "scored": k < cfg.max_candidates})
        groups.append({"token": sp.text, "core": sp.core, "class": sp.cls, "candidates": rows,
                       "suffix_issue": (ml.suffix_issue(sp.core) if sp.cls == "num_suffix" else None) or ""})
    return {"text": text, "style": style, "max_candidates": cfg.max_candidates, "groups": groups}


def api_preview(st, body):
    """Verbalized text of each sentence with the given readings (open numbers stay as written)."""
    run = st["run_dir"]
    stem, kind = _safe_stem(body.get("file")), body.get("kind")
    _, doc = _load(run, stem, kind)
    cfg = run_config(run)
    plans = plan_text(doc, cfg)
    rows = _file_rows(run, stem, kind) if kind != "failed" else []
    try:
        if kind != "failed" and body.get("edits"):   # decided numbers changed: from the original
            _, plans, full, _, _ = _rebuild(run, stem, cfg, rows, body.get("picks"), body["edits"], st)
        else:
            given = _given_from(body, plans, _open_numbers(plans, rows, kind))
            full, _, _ = _full_choices(plans, given)
    except Misaligned:
        raise
    except ValueError as e:                        # digits typed: show it, don't fail the preview
        return {"preview": {}, "invalid": str(e)}
    from .. import verbalize as _verbalize
    out = {}
    for p in plans:
        if p["idx"] in full:
            it = iter(full[p["idx"]])
            out[p["idx"]] = _verbalize(p["text_train"], style=cfg.style, chooser=lambda span, cs: next(it))
    return {"preview": out}


class Conflict(ValueError):
    pass


TRASH = ".trash"                               # <run>/.trash: superseded copies on their way out


def _remove(path):
    """Delete a superseded copy so its NAME disappears at once. On Windows a plain delete of a
    file another program has open (Explorer preview / thumbnails, search indexer, antivirus,
    OneDrive) only marks it "delete pending": it stays listed but can't be opened until that
    program lets go. So the copy is first renamed into <run>/.trash/ (a rename works while it is
    open) and then deleted there; whatever can't be deleted yet is emptied on the next start /
    file listing. If even the rename fails the file is left and api_files cleans it up later."""
    if not os.path.exists(path):
        return
    run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    trash = os.path.join(run, TRASH)
    try:
        os.makedirs(trash, exist_ok=True)
        dst = os.path.join(trash, f"{os.path.basename(os.path.dirname(path))}__{os.path.basename(path)}"
                                  f".{_dt.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
        os.replace(path, dst)
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass
        return
    try:
        os.remove(dst)
    except OSError:
        pass


def _empty_trash(run):
    d = os.path.join(run or "", TRASH)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            try:
                os.remove(os.path.join(d, fn))
            except OSError:
                pass


def _leftover(run, stem, kind, rows_by_file):
    """A verify/failed copy whose file is already in verbalized/ and has no open rows: an
    interrupted delete. It is removed instead of listed."""
    return (os.path.exists(os.path.join(run, "verbalized", stem + ".json"))
            and not any(r.get("status") == "pending" for r in rows_by_file.get(stem, [])))


def _bump(run, **delta):
    s = _summary(run)
    if s:
        for k, v in delta.items():
            s[k] = max(0, int(s.get(k) or 0) + v)
        _write_json_atomic(os.path.join(run, "summary.json"), s)


def api_save(st, body):
    """Apply the readings chosen for the open numbers. Numbers left empty stay open. When no
    number is open any more the file is written to verbalized/, its rows move from verify.tsv to
    verbalized.tsv and the copy is removed; otherwise the copy (in verify/) is updated."""
    run = st["run_dir"]
    stem, kind = _safe_stem(body.get("file")), body.get("kind")
    with _LOCK:
        src, doc, rev = _load_rev(run, stem, kind)
        if body.get("rev") and body["rev"] != rev:
            raise Conflict(f"{kind}/{stem}.json changed since it was opened (saved already, or in another "
                           f"tab) — reopen the file")
        if kind == "failed" and os.path.exists(os.path.join(run, "verify", stem + ".json")):
            raise ValueError(f"verify/{stem}.json also exists — check that one (or remove it) first")
        cfg = run_config(run)
        plans = plan_text(doc, cfg)
        rk = _rows_kind(kind)                             # verify.tsv, or reverify.tsv for reverify/
        vrows = read_tsv(tsv_path(run, rk))
        mine = [r for r in vrows if r.get("file") == stem]
        others = [r for r in vrows if r.get("file") != stem]
        now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        edits = body.get("edits") or {}
        if edits and kind == "failed":
            raise ValueError("only numbers of a verify / reverify file can be re-edited")
        if edits:                                         # re-edit decided numbers: rebuild from the input
            orig, oplans, full, changes, later = _rebuild(run, stem, cfg, mine, body.get("picks"), edits, st)
            if not changes:
                raise ValueError("nothing changed")
            out, _, _ = same_document(orig, oplans, cfg, choices=full)
            typed = [r for r, t in changes if t not in (r.get("candidates") or "").split(SEP)]
            decisions = [decision_record(now, stem, kind, "pick" if r.get("status") == "pending" else "edit", r, t,
                                         previous=None if r.get("status") == "pending" else (r.get("final_pick") or ""))
                         for r, t in changes]
            for r, t in changes:
                r.update(status="verified", final_pick=t, verified_at=now)
            decided = changes
            typed_recs = [dict(r, kind=kind, typed=r["final_pick"], saved_at=now) for r in typed]
        else:
            opened = _open_numbers(plans, mine, kind)
            full, decided, later = _full_choices(plans, _given_from(body, plans, opened))
            if not decided:
                raise ValueError("nothing chosen")
            out, _, _ = same_document(doc, plans, cfg, choices=full)
            row_of = {(si, k): r for _, si, k, _, r in opened}
            typed_recs = []
            decisions = []
            for si, k, n, t in decided:                   # readings that are not a candidate = typed
                r = row_of.get((si, k)) or {}
                cands = (r.get("candidates") or "").split(SEP) if r.get("candidates") else [c.text for c in n.cands]
                p_ = next(p for p in plans if p["idx"] == si)
                base = r or {"sentence": si, "number_index": k, "number": n.span.core, "class": n.span.cls}
                decisions.append(decision_record(now, stem, kind, "pick", base, t, [c.text for c in n.cands],
                                                 p_["sentence"]))
                if t not in cands:
                    p = next(p for p in plans if p["idx"] == si)
                    typed_recs.append({"saved_at": now, "file": stem, "kind": kind, "sentence": si,
                                       "number_index": r.get("number_index", k), "number": n.span.core,
                                       "class": n.span.cls, "typed": t, "model_pick": r.get("model_pick", ""),
                                       "previous_pick": r.get("previous_pick", ""), "candidates": SEP.join(cands),
                                       "start_time": p["sentence"].get("start_time"),
                                       "end_time": p["sentence"].get("end_time"),
                                       "sentence_text": r.get("sentence_text") or p["sentence"].get("text") or ""})

        if edits:
            pass
        elif kind != "failed":
            for si, k, n, t in decided:                   # aligned by _open_numbers (else Misaligned)
                row_of[(si, k)].update(status="verified", final_pick=t, verified_at=now)
        else:                                             # failed: number rows created now (no scores)
            frow = next((r for r in read_tsv(tsv_path(run, "failed")) if r.get("file") == stem), {})
            reason = f"from failed/ ({frow.get('stage', '')}: {frow.get('error', '')})"
            chosen = {(si, k): t for si, k, _, t in decided}
            mine = []
            for p in plans:
                for k, n in enumerate(p["numbers"]):
                    t = chosen.get((p["idx"], k))
                    mine.append({"file": stem, "sentence": p["idx"], "number_index": k,
                                 "start_time": p["sentence"].get("start_time"), "end_time": p["sentence"].get("end_time"),
                                 "number": n.span.core, "class": n.span.cls,
                                 "candidates": SEP.join(c.text for c in n.cands), "scores": "", "model_pick": "",
                                 "suffix_issue": _text_suffix_issue(n) or "",
                                 "text_issue": n.text_issue or "",
                                 "file_reasons": reason, "status": "verified" if t else "pending",
                                 "final_pick": t or "", "verified_at": now if t else "",
                                 "sentence_text": p["sentence"].get("text") or ""})
            frows = read_tsv(tsv_path(run, "failed"))
            write_tsv(tsv_path(run, "failed"), COLS["failed"], [r for r in frows if r.get("file") != stem])

        if later == 0:                                    # complete -> verbalized
            # new state first (json, then its rows in verbalized.tsv), old state last (rows out of
            # verify.tsv, then the copy): a crash or a locked file leaves a duplicate, never a loss
            os.makedirs(os.path.join(run, "verbalized"), exist_ok=True)
            _write_json_atomic(os.path.join(run, "verbalized", stem + ".json"), out)
            vp = tsv_path(run, "verbalized")
            write_tsv(vp, NUMBER_COLS, [r for r in read_tsv(vp) if r.get("file") != stem] + mine)
            write_tsv(tsv_path(run, rk), NUMBER_COLS, others)
            _bump(run, **{kind: -1, "verbalized": 1, "verified": 1})
            _remove(src)
            _remove(os.path.join(run, "input", stem + ".json"))    # input copy no longer needed
        else:                                             # still open numbers -> (stays in) verify/ / reverify/
            folder = "reverify" if kind == "reverify" else "verify"
            os.makedirs(os.path.join(run, folder), exist_ok=True)
            _write_json_atomic(os.path.join(run, folder, stem + ".json"), out)
            write_tsv(tsv_path(run, rk), NUMBER_COLS, others + mine)
            if kind == "failed":                          # the failed copy IS the input: keep it
                os.makedirs(os.path.join(run, "input"), exist_ok=True)
                shutil.copyfile(src, os.path.join(run, "input", stem + ".json"))
                _bump(run, failed=-1, verify=1)
                _remove(src)
        _log_typed(run, typed_recs)
        append_decisions(run, decisions)              # every saved reading -> verifier_decisions.tsv
    return {"ok": True, "file": stem, "decided": len(decided), "open": later, "done": later == 0,
            "typed": len(typed_recs)}


# ----------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "indic-digits-verifier"

    def log_message(self, fmt, *args):            # quiet
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _err(self, e, code=400):
        self._json({"error": str(e)}, code)

    def _body(self):
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":               # forces a CORS preflight on cross-site posts
            raise PermissionError("Content-Type must be application/json")
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _local_only(self):
        """Only this machine's own page may use the API: the Host must be a loopback name (stops
        DNS rebinding) and a cross-site Origin is refused (stops other websites' requests)."""
        host = (self.headers.get("Host") or "").lower()
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if name not in ("127.0.0.1", "localhost", "[::1]"):
            raise PermissionError(f"bad Host {host!r}")
        origin = self.headers.get("Origin")
        if origin and origin.lower().rstrip("/") not in (f"http://{host}",):
            raise PermissionError(f"cross-site request from {origin!r} refused")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        st = self.server.state
        try:
            self._local_only()
            if u.path in ("/", "/index.html"):
                b = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif u.path == "/api/state":
                self._json(api_state(st))
            elif u.path == "/api/browse":
                self._json(api_browse(q.get("path", "")))
            elif u.path == "/api/files":
                self._need_run(st)
                self._json(api_files(st))
            elif u.path == "/api/file":
                self._need_run(st)
                self._json(api_file(st, q.get("file"), q.get("kind")))
            elif u.path == "/api/candidates":
                self._json(api_candidates(st, q.get("q", ""), q.get("style") or None))
            elif u.path == "/api/audio":
                self._audio(st, q.get("file"))
            else:
                self._err("not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except PermissionError as e:
            self._err(e, 403)
        except Exception as e:
            self._err(e)

    def do_POST(self):
        st = self.server.state
        try:
            self._local_only()
            body = self._body()
            path = urlparse(self.path).path
            if path == "/api/open":
                self._json(api_open(st, body))
            elif path == "/api/save":
                self._need_run(st)
                self._json(api_save(st, body))
            elif path == "/api/preview":
                self._need_run(st)
                self._json(api_preview(st, body))
            else:
                self._err("not found", 404)
        except PermissionError as e:
            self._err(e, 403)
        except Conflict as e:
            self._err(e, 409)
        except Exception as e:
            self._err(e)

    @staticmethod
    def _need_run(st):
        if not is_run_dir(st.get("run_dir")):
            raise ValueError("load a run folder first")

    def _audio(self, st, name):
        p = _find_wav(st.get("audio_dir"), _safe_stem(name))
        if not p:
            return self._err("audio not found (set the audio folder)", 404)
        size = os.path.getsize(p)
        rng = _parse_range(self.headers.get("Range"), size)
        if rng == "invalid":
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if rng:
            start, end = rng
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            start, end = 0, size - 1
            self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def _parse_range(header, size):
    """None = whole file (no / multi-range / unparsable header), (start, end) inclusive, or
    "invalid" when the range can't be satisfied (-> 416)."""
    m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", header or "")
    if not m or not (m.group(1) or m.group(2)):
        return None
    if m.group(1):
        start = int(m.group(1))
        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    else:
        n = int(m.group(2))
        if n == 0:
            return "invalid"
        start, end = max(0, size - n), size - 1
    if start >= size or start > end:
        return "invalid"
    return start, end


def serve(run_dir=None, audio_dir=None, port=8765, open_browser=True):
    st = _load_settings()
    if run_dir:
        try:
            api_open(st, {"run_dir": run_dir, "audio_dir": audio_dir or ""})
        except ValueError as e:
            print(f"warning: {e}", file=sys.stderr)
    elif audio_dir:
        st["audio_dir"] = os.path.abspath(audio_dir)
    srv = None
    for p in range(port, port + 20):                 # first free port
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"no free port in {port}-{port + 19}")
    srv.state = st
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"verifier running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        srv.server_close()
