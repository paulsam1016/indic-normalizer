"""Temporary: apply the currency-sign handling (₹ / Rs / INR -> "xxx രൂപ / റുപ്പീസ്" candidates,
always reviewed) to an EXISTING run, without re-verbalizing everything.

Finds the files of a run whose input has an amount with a currency sign, then runs `reverify`
on just those files: finished files in verbalized/ go to reverify/ (their currency numbers are
now always flagged), files waiting in verify/ / reverify/ get the new candidates and scores.

    # which files? (no GPU)
    uv run python scripts/currency_reverify.py OUT_DIR/v1 --list-only [--data-dir D]
    # look first, then apply (GPU)
    uv run python scripts/currency_reverify.py OUT_DIR/v1 --audio-dir AUDIO --config my.yaml --dry-run
    uv run python scripts/currency_reverify.py OUT_DIR/v1 --audio-dir AUDIO --config my.yaml
"""

import argparse
import json
import os
import re
import sys

from indic_normalizer.digits import find
from indic_normalizer.digits.detect import join_currency
from indic_normalizer.digits.scoring.details import read_tsv, tsv_path
from indic_normalizer.digits.scoring.pipeline import _sentences
from indic_normalizer.languages.malayalam import normalize


def _summary(run):
    try:
        with open(os.path.join(run, "summary.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def currency_files(run, data_dir=None):
    """{stem: [amounts]} for the run's files whose input has an amount with ₹ / Rs / INR."""
    data_dir = data_dir or _summary(run).get("data_dir")
    stems = set()
    for k in ("verbalized", "verify", "reverify"):
        d = os.path.join(run, k)
        if os.path.isdir(d):
            stems |= {f[:-5] for f in os.listdir(d) if f.endswith(".json")}
    plain = not stems and os.path.isdir(run)
    if plain:                      # an input folder, not a run dir: scan its files directly
        stems = {f[:-5] for f in os.listdir(run) if f.endswith(".json") and not f.endswith(".verbalized.json")}
        data_dir = run
    # sentence texts stored in the run's TSVs: the fallback when the original input is gone
    stored = {}
    for k in ("verbalized", "verify", "reverify"):
        for r in read_tsv(tsv_path(run, k)):
            stored.setdefault(r.get("file"), {})[r.get("sentence")] = r.get("sentence_text") or ""
    out, missing = {}, []
    scan = {"files_read": 0, "from_tsv": 0, "sentences": 0, "raw_hits": 0, "raw_examples": []}
    raw = re.compile(r"₹|(?<![a-z])(?:rs\.?|inr)(?![a-z])|രൂപ|റുപ്പീസ്|rupees?", re.I)
    for s in sorted(stems):
        p = next((os.path.join(d, s + ".json") for d in (os.path.join(run, "input"), data_dir)
                  if d and os.path.exists(os.path.join(d, s + ".json"))), None)
        if p:
            with open(p, encoding="utf-8-sig") as f:
                sents = _sentences(json.load(f))
            scan["files_read"] += 1
        elif s in stored:
            missing.append(s)            # found from the TSV, but reverify can't re-score it
            sents = [{"text": t} for t in stored[s].values()]
            scan["from_tsv"] += 1
        else:
            missing.append(s)
            continue
        scan["sentences"] += len(sents)
        for x in sents:
            t = str(x.get("text") or "")
            if raw.search(t):
                scan["raw_hits"] += 1
                if len(scan["raw_examples"]) < 10:
                    scan["raw_examples"].append(f"{s}: {t[:100]}")
        amounts = [sp.core for x in sents
                   for sp in find(join_currency(normalize(str(x.get("text") or ""), mode="train")))
                   if sp.cls == "currency"]
        if amounts:
            out[s] = amounts
    currency_files.scan = dict(scan, stems=len(stems), data_dir=data_dir, plain=plain)
    return out, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--audio-dir")
    ap.add_argument("--config", help="scoring config (needed unless --list-only)")
    ap.add_argument("--data-dir", help="input dir (default: <run>/input/, then summary.json data_dir)")
    ap.add_argument("--list-only", action="store_true", help="only list the files with currency amounts (no GPU)")
    ap.add_argument("--dry-run", action="store_true", help="reverify --dry-run: report, move nothing")
    a = ap.parse_args(argv)
    files, missing = currency_files(a.run_dir, a.data_dir)
    sc = currency_files.scan
    if sc["plain"]:
        print(f"{a.run_dir} has no verbalized/ verify/ reverify/: scanning it as an INPUT folder "
              "(listing only; to re-score, give the verbalize run dir, i.e. the --out-dir)")
    print(f"run has {sc['stems']} file(s): {sc['files_read']} original input(s) found "
          f"(input/, then data_dir={sc['data_dir']}), {sc['from_tsv']} read from the TSVs' sentence_text")
    print(f"sentences with a currency sign or word (₹ rs inr രൂപ റുപ്പീസ് rupee): {sc['raw_hits']} of {sc['sentences']}")
    for e in sc["raw_examples"]:
        print(f"  {e}")
    gone = set(missing)
    print(f"{len(files)} file(s) with currency amounts ({sum(len(v) for v in files.values())} amounts)")
    for s, v in list(files.items())[:50]:
        print(f"  {s}: {', '.join(v[:6])}" + (" ..." if len(v) > 6 else "")
              + ("   [no original input: can't re-score]" if s in gone else ""))
    no_src = [s for s in files if s in gone]
    if no_src:
        print(f"{len(no_src)} of them have no original input (not in input/ or --data-dir): "
              "restore the originals or give --data-dir to re-score those")
    other = len(gone) - len(no_src)
    if other:
        print(f"{other} other file(s) have no original input")
    files = {s: v for s, v in files.items() if s not in gone}
    if a.list_only or not files or sc["plain"]:
        return 0
    if not (a.audio_dir and a.config):
        sys.exit("--audio-dir and --config are needed to re-score (or use --list-only)")
    from indic_normalizer.digits.scoring.config import load_config
    from indic_normalizer.digits.scoring.nemo_scorer import NemoScorer
    from indic_normalizer.digits.scoring.reverify import reverify
    cfg = load_config(a.config)
    tot = reverify(a.run_dir, a.audio_dir, NemoScorer(cfg), cfg, data_dir=a.data_dir,
                   files=sorted(files), dry_run=a.dry_run)
    w = tot.get("waiting") or {}
    print(f"checked {tot['files_checked']} finished file(s): {tot['numbers_flagged']} number(s) flagged in "
          f"{tot['files_flagged']} file(s)" + (" (dry run)" if a.dry_run else " -> reverify/"))
    print(f"waiting files re-scored: {w.get('files', 0)} ({w.get('numbers_refreshed', 0)} open numbers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
