"""``indic-digits status RUN_DIR``: is a verbalize run complete? (no GPU, read-only)

Checks that every input file (``--data-dir``, else summary.json ``data_dir``) has its output in
verbalized/, and that those outputs are finished:
    missing        input files with no verbalized/<stem>.json — and where they are instead
                   (verify/, reverify/, failed/, or nowhere: not processed yet)
    open numbers   pending rows still waiting in verify.tsv / reverify.tsv
    digits left    verbalized files whose sentence texts still contain digits (a number that was
                   never verbalized)
    no rows        verbalized files without rows in verbalized.tsv; rows still pending there
    extra          verbalized files that are not in the input dir
Exit code 0 when everything is complete, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

from .details import read_tsv, tsv_path

_DIGIT = re.compile(r"\d")


def _stems(d: str) -> set:
    return {f[:-5] for f in os.listdir(d) if f.endswith(".json")} if os.path.isdir(d) else set()


def _summary(run: str) -> dict:
    try:
        with open(os.path.join(run, "summary.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _n_numbers(run: str, data_dir: Optional[str], stem: str) -> Optional[int]:
    """Numbers in a file's ORIGINAL input (input/ copy or data dir); None if it can't be found."""
    from ...languages import malayalam as _ml
    from .. import find
    for d in (os.path.join(run, "input"), data_dir):
        p = os.path.join(d, stem + ".json") if d else None
        if p and os.path.exists(p):
            try:
                with open(p, encoding="utf-8-sig") as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                return None
            sents = doc.get("sentences") if isinstance(doc.get("sentences"), list) else \
                [{"text": doc.get("transcription") or ""}]
            return sum(len(find(_ml.normalize(str(x.get("text") or ""), mode="train"))) for x in sents)
    return None


def run_status(run: str, data_dir: Optional[str] = None) -> Dict:
    run = os.path.abspath(run)
    data_dir = data_dir or _summary(run).get("data_dir")
    folders = {k: _stems(os.path.join(run, k)) for k in ("verbalized", "verify", "reverify", "failed")}
    if data_dir and os.path.isdir(data_dir):
        inputs = {f[:-5] for f in os.listdir(data_dir)
                  if f.endswith(".json") and not f.endswith(".verbalized.json")}
    else:
        inputs = None
    rep: Dict = {"run_dir": run, "data_dir": data_dir if inputs is not None else None,
                 "counts": {k: len(v) for k, v in folders.items()}}
    if inputs is not None:
        rep["inputs"] = len(inputs)
        missing = sorted(inputs - folders["verbalized"])
        where = {}
        for s in missing:
            k = next((k for k in ("reverify", "verify", "failed") if s in folders[k]), "not processed")
            where.setdefault(k, []).append(s)
        rep["missing"] = where
        rep["extra"] = sorted(folders["verbalized"] - inputs)
    open_rows = {}
    for k in ("verify", "reverify"):
        n = {}
        for r in read_tsv(tsv_path(run, k)):
            if r.get("status") == "pending":
                n[r.get("file")] = n.get(r.get("file"), 0) + 1
        open_rows[k] = n
    rep["open_numbers"] = {k: sum(v.values()) for k, v in open_rows.items()}
    vrows = read_tsv(tsv_path(run, "verbalized"))
    with_rows = {r.get("file") for r in vrows}
    no_rows = sorted(folders["verbalized"] - with_rows)
    # a file without any number has no rows by design: only the ones whose input HAS numbers count
    rep["verbalized_without_numbers"] = [s for s in no_rows if _n_numbers(run, data_dir, s) == 0]
    rep["verbalized_without_rows"] = [s for s in no_rows if s not in rep["verbalized_without_numbers"]]
    rep["verbalized_rows_pending"] = sorted({r.get("file") for r in vrows if r.get("status") == "pending"})
    digits_left = {}
    for s in sorted(folders["verbalized"]):
        try:
            with open(os.path.join(run, "verbalized", s + ".json"), encoding="utf-8-sig") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            digits_left[s] = [f"cannot read: {e}"]
            continue
        sents = doc.get("sentences") if isinstance(doc.get("sentences"), list) else []
        bad = [f"{i}: {x.get('text')}" for i, x in enumerate(sents) if _DIGIT.search(str(x.get("text") or ""))]
        if bad:
            digits_left[s] = bad
    rep["digits_left"] = digits_left
    problems = (sum(len(v) for v in rep.get("missing", {}).values()) + len(rep.get("extra", []))
                + sum(rep["open_numbers"].values()) + len(rep["verbalized_without_rows"])
                + len(rep["verbalized_rows_pending"]) + len(digits_left))
    rep["complete"] = problems == 0 and inputs is not None
    return rep


def print_status(rep: Dict, show: int = 20, log=print) -> None:
    c = rep["counts"]
    log(f"run dir   {rep['run_dir']}")
    if rep.get("data_dir"):
        log(f"inputs    {rep['inputs']} in {rep['data_dir']}")
    else:
        log("inputs    unknown: give --data-dir (summary.json data_dir not found here)")
    log(f"folders   verbalized {c['verbalized']} · verify {c['verify']} · reverify {c['reverify']} · failed {c['failed']}")
    log(f"open      {rep['open_numbers']['verify']} number(s) in verify/, {rep['open_numbers']['reverify']} in reverify/")

    def lst(title, items):
        if items:
            log(f"{title}: {len(items)}")
            for x in list(items)[:show]:
                log(f"  {x}")
            if len(items) > show:
                log(f"  ... ({len(items) - show} more)")
    for k, v in (rep.get("missing") or {}).items():
        lst(f"not in verbalized/ — {k}", v)
    lst("in verbalized/ but not an input", rep.get("extra", []))
    if rep.get("verbalized_without_numbers"):
        log(f"verbalized files without numbers (no rows needed): {len(rep['verbalized_without_numbers'])}")
    lst("verbalized files WITH numbers but no rows in verbalized.tsv (rows lost: re-run verbalize "
        "on them, or ignore if the text is right)", rep["verbalized_without_rows"])
    lst("verbalized files with pending rows", rep["verbalized_rows_pending"])
    lst("verbalized files with digits left", [f"{s}  ({len(v)} sentence(s), e.g. {v[0][:80]})"
                                             for s, v in rep["digits_left"].items()])
    log("COMPLETE: every input is verbalized" if rep["complete"] else "NOT complete")
