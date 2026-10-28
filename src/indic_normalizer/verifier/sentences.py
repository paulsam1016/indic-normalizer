"""Hand edits of sentence TEXT (typos, wrong words) in a verbalize run's output files.

An edit replaces one sentence's text, train-normalized like the rest of the output (no digits:
numbers are written in words). The file's ``transcription`` is re-joined, and the sentence's
entries in ``words`` are replaced by the edited words, the sentence's word time span (or the
sentence's own span when it had no words) spread across them in proportion to their length:
approximate timings, not a forced alignment. Each edited word has ``"edited": true``.

Every edit is appended to <run>/sentence_edits.tsv (never rewritten). The latest row per
(file, sentence) is that sentence's text from then on: ``apply_edits`` puts it back whenever a
file is rebuilt from its original input (a decided number changed in the verifier, ``reverify``),
so a hand fix is never lost by a rebuild. Re-running ``verbalize`` on a file starts it afresh.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, List, Optional

from ..digits.scoring.details import _clean, read_tsv
from ..digits.scoring.pipeline import _sentences, _split_times, _word_owners, _word_owners_by_text

EDIT_COLS = ["saved_at", "file", "kind", "sentence", "start_time", "end_time", "before", "after"]
EDITS_TSV = "sentence_edits.tsv"
_DIGIT = re.compile(r"\d")


def normalize(text: str) -> str:
    """The edited text as the output stores sentences (train mode); ValueError if unusable."""
    from ..languages import malayalam as _ml
    t = " ".join(_ml.normalize(" ".join(str(text or "").split()), mode="train").split())
    if not t:
        raise ValueError("the sentence is empty after normalization")
    if _DIGIT.search(t):
        raise ValueError(f"the sentence contains digits ({t!r}): write numbers in words")
    return t


def has_sentences(doc: dict) -> bool:
    """Only files with a sentence list (``sentences`` / ``segments``) can have sentences edited."""
    return isinstance(doc.get("sentences"), list) and bool(doc["sentences"]) or \
        isinstance(doc.get("segments"), list) and bool(doc["segments"])


def edit_sentence(doc: dict, idx: int, text: str) -> dict:
    """A copy of ``doc`` with sentence ``idx`` set to ``text`` (already normalized), its words
    re-split and ``transcription`` re-joined."""
    out = json.loads(json.dumps(doc))
    sents = _sentences(out)
    if not has_sentences(out) or not 0 <= idx < len(sents):
        raise ValueError(f"no sentence {idx} in this file")
    words = out.get("words") if isinstance(out.get("words"), list) else None
    if words is not None:
        owner = _word_owners(words, sents)
        owner.update(_word_owners_by_text(words, sents, {k: s.get("text") or "" for k, s in enumerate(sents)}))
        mine = [i for i, w in enumerate(words) if owner.get(id(w)) == idx]
    sents[idx]["text"] = text
    if "transcription" in out or words is not None:
        out["transcription"] = " ".join(x.get("text", "") for x in sents if x.get("text"))
    if words is None:
        return out
    s = sents[idx]
    if mine:
        t0, t1 = words[mine[0]].get("start_time"), words[mine[-1]].get("end_time")
        base, at = dict(words[mine[0]]), mine[0]
    else:
        t0, t1 = s.get("start_time"), s.get("end_time")
        at = next((i for i, w in enumerate(words) if _num(w.get("start_time")) is not None
                   and _num(t0) is not None and _num(w.get("start_time")) >= _num(t0)), len(words))
        base = {k: v for k, v in (words[at - 1] if at else (words[0] if words else {})).items()
                if k not in ("word", "start_time", "end_time")}
        if s.get("speaker") is not None:
            base["speaker"] = s["speaker"]
    parts = text.split()
    new = []
    for part, (a, b) in zip(parts, _split_times(t0, t1, parts)):
        w = dict(base, word=part, edited=True)
        if a is not None:
            w["start_time"], w["end_time"] = a, b
        new.append(w)
    keep = set(mine)
    out["words"] = [w for i, w in enumerate(words[:at]) if i not in keep] + new + \
                   [w for i, w in enumerate(words[at:], at) if i not in keep]
    return out


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def append_edit(run: str, rec: dict) -> None:
    p = os.path.join(run, EDITS_TSV)
    new_file = not os.path.exists(p)
    with open(p, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if new_file:
            w.writerow(EDIT_COLS)
        w.writerow([_clean(rec.get(c)) for c in EDIT_COLS])


def edits_for(run: str, stem: str) -> Dict[int, dict]:
    """{sentence: latest edit row} of one file; ``original`` = the text before its first edit."""
    out = {}
    for r in read_tsv(os.path.join(run, EDITS_TSV)):
        if r.get("file") == stem:
            try:
                k = int(r["sentence"])
            except (KeyError, TypeError, ValueError):
                continue
            out[k] = dict(r, original=(out[k]["original"] if k in out else r.get("before") or ""))
    return out


def apply_edits(run: str, stem: str, doc: dict) -> dict:
    """``doc`` with every hand-edited sentence of ``stem`` put back (see the module docstring)."""
    eds = edits_for(run, stem)
    if not eds or not has_sentences(doc):
        return doc
    n = len(_sentences(doc))
    for idx in sorted(eds):
        if idx < n and (_sentences(doc)[idx].get("text") or "") != eds[idx]["after"]:
            doc = edit_sentence(doc, idx, eds[idx]["after"])
    return doc
