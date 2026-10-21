"""``indic-digits fix RUN_DIR``: repair a run dir after interrupted saves / locked files (no GPU).

    1. empty <run>/.trash (copies the verifier parked there while Windows still had them open)
    2. leftover copies: a verify/ reverify/ failed/ copy whose file is already in verbalized/ and
       has no open (pending) rows -> removed (an interrupted delete)
    3. stale rows: verify.tsv / reverify.tsv rows of a file that is in verbalized/ and has no copy
       left -> moved to verbalized.tsv when that has no rows for it, otherwise dropped
    4. failed.tsv rows of a file that is no longer in failed/ -> dropped
    5. summary.json counts recomputed from the folders
Problems it can't decide are listed (rows without any file, copies without rows). ``--dry-run``
only reports. A file Windows still holds open ("delete pending": listed, but can't be opened)
disappears once the program holding it lets go; run fix again after that.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .details import COLS, NUMBER_COLS, read_tsv, tsv_path, write_tsv

KINDS = ("verify", "reverify", "failed")


def _stems(d: str) -> set:
    return {f[:-5] for f in os.listdir(d) if f.endswith(".json")} if os.path.isdir(d) else set()


def fix_run(run: str, dry_run: bool = False, log=print) -> Dict[str, List[str]]:
    from ..verifier.server import _empty_trash, _remove
    run = os.path.abspath(run)
    rep = {k: [] for k in ("trash_emptied", "leftover_copies_removed", "rows_moved_to_verbalized",
                           "rows_dropped", "failed_rows_dropped", "rows_without_file", "copies_without_rows")}
    trash = os.path.join(run, ".trash")
    if os.path.isdir(trash):
        rep["trash_emptied"] = sorted(os.listdir(trash))
        if not dry_run:
            _empty_trash(run)
    verbalized = _stems(os.path.join(run, "verbalized"))
    copies = {k: _stems(os.path.join(run, k)) for k in KINDS}
    vrows = read_tsv(tsv_path(run, "verbalized"))
    v_files = {r.get("file") for r in vrows}
    changed_v = False
    for kind in ("verify", "reverify"):
        p = tsv_path(run, kind)
        rows = read_tsv(p)
        by_file: Dict[str, list] = {}
        for r in rows:
            by_file.setdefault(r.get("file"), []).append(r)
        keep = []
        for stem, rs in by_file.items():
            pending = any(r.get("status") == "pending" for r in rs)
            if stem in copies[kind]:
                if stem in verbalized and not pending:          # 2. interrupted delete
                    rep["leftover_copies_removed"].append(f"{kind}/{stem}.json")
                    if not dry_run:
                        _remove(os.path.join(run, kind, stem + ".json"))
                    if stem not in v_files:
                        vrows += rs
                        v_files.add(stem)
                        changed_v = True
                        rep["rows_moved_to_verbalized"].append(f"{kind}: {stem}")
                    else:
                        rep["rows_dropped"].append(f"{kind}: {stem}")
                    continue
                keep += rs
                continue
            if stem in verbalized:                               # 3. rows left behind
                if stem not in v_files and not pending:
                    vrows += rs
                    v_files.add(stem)
                    changed_v = True
                    rep["rows_moved_to_verbalized"].append(f"{kind}: {stem}")
                else:
                    rep["rows_dropped"].append(f"{kind}: {stem}")
                continue
            rep["rows_without_file"].append(f"{kind}: {stem}")   # can't decide: keep
            keep += rs
        if not dry_run and len(keep) != len(rows):
            write_tsv(p, NUMBER_COLS, keep)
        for stem in sorted(copies[kind] - set(by_file)):     # copies whose rows are gone
            if stem in verbalized:                              # finished: an interrupted delete
                rep["leftover_copies_removed"].append(f"{kind}/{stem}.json")
                if not dry_run:
                    _remove(os.path.join(run, kind, stem + ".json"))
            else:
                rep["copies_without_rows"].append(f"{kind}/{stem}.json")
    if changed_v and not dry_run:
        write_tsv(tsv_path(run, "verbalized"), NUMBER_COLS, vrows)
    fp = tsv_path(run, "failed")
    frows = read_tsv(fp)
    fkeep = [r for r in frows if r.get("file") in copies["failed"]]
    rep["failed_rows_dropped"] = [r.get("file") for r in frows if r.get("file") not in copies["failed"]]
    for stem in copies["failed"]:                                # leftover failed copies
        if stem in verbalized or stem in copies["verify"] or stem in copies["reverify"]:
            rep["leftover_copies_removed"].append(f"failed/{stem}.json")
            fkeep = [r for r in fkeep if r.get("file") != stem]
            if not dry_run:
                _remove(os.path.join(run, "failed", stem + ".json"))
    if not dry_run and len(fkeep) != len(frows):
        write_tsv(fp, COLS["failed"], fkeep)
    if not dry_run:                                              # 5. summary from the folders
        sp = os.path.join(run, "summary.json")
        try:
            with open(sp, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError):
            s = {}
        counts = {k: len(_stems(os.path.join(run, k))) for k in ("verbalized", "verify", "reverify", "failed")}
        s.update(counts, files=sum(counts.values()))
        tmp = sp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        os.replace(tmp, sp)
        rep["summary"] = [f"{k} {v}" for k, v in counts.items()]
    for k, v in rep.items():
        if v:
            log(f"{k}: {len(v)}" + "".join(f"\n  {x}" for x in v[:20]) + ("\n  ..." if len(v) > 20 else ""))
    return rep
