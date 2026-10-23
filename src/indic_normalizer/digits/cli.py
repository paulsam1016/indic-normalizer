"""``indic-digits`` command line.

    indic-digits stats DATA_DIR [--out-root outputs] [--tokenizer tokenizer.model]
    indic-digits latin DATA_DIR [--out-root outputs] [--normalize none|train|eval]
    indic-digits verbalize --config cfg.yaml DATA_DIR AUDIO_DIR [--out-dir OUT_DIR] [--limit N] [--resume | --retry-failed]
    indic-digits compare   --config cfg.yaml DATA_DIR AUDIO_DIR [--hyps-dir base_hypotheses]
                           [--out-root outputs] [--limit N]
    indic-digits sweep     --config cfg.yaml [--sweep-config sweep.yaml] DATA_DIR AUDIO_DIR [--limit N]
                           [--batch-size-values 8,16,32,64] [--greedy] [--out-root outputs/sweep]
    indic-digits verifier  [RUN_DIR] [--audio-dir DIR]                # check verify/ + failed/ in the browser
    indic-digits reverify  --config cfg.yaml RUN_DIR AUDIO_DIR [--data-dir DIR] [--files a,b] [--dry-run]
                           # re-score verbalized/ files; doubtful ones go to reverify/
    indic-digits check     --config cfg.yaml DATA_DIR AUDIO_DIR   # model sanity check, run first
    indic-digits status    RUN_DIR [--data-dir DIR]                # complete? missing / open / digits left
    indic-digits fix       RUN_DIR [--dry-run]                     # repair leftovers / stale rows / counts
    indic-digits text "ഫീസ് 1,75,000 ആണ്"          # default reading, no audio
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _scoring_args(p):
    p.add_argument("data_dir", help="transcript JSON dir (e.g. eval_frozen)")
    p.add_argument("audio_dir", help="dir with <stem>.wav for each transcript")
    p.add_argument("--config", required=True, help="YAML/JSON scoring config (model_path, batch_size, ...)")
    p.add_argument("--model-path")
    p.add_argument("--cuda", type=int, help="CUDA device index (negative = CPU)")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--limit", type=int, help="only the first N files")


def _cfg(a, **extra):
    """config file < ``extra`` (e.g. a sweep config's ``base``) < command-line flags (when given)."""
    from .scoring.config import load_config
    kw = dict(extra)
    for k, v in (("model_path", a.model_path), ("cuda", a.cuda), ("batch_size", a.batch_size)):
        if v is not None or k not in kw:
            kw[k] = v
    return load_config(a.config, **kw)


def _scorer(cfg):
    from .scoring.nemo_scorer import NemoScorer
    return NemoScorer(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="indic-digits", description="Digit normalization tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    from . import stats
    stats.add_arguments(sub.add_parser("stats", help="digit / special-character distribution of a dir"))
    from . import latin_stats
    latin_stats.add_arguments(sub.add_parser("latin", help="English (Latin-script) character / word distribution of a dir"))

    v = sub.add_parser("verbalize", help="pick each number's reading by audio scoring; write JSON")
    _scoring_args(v)
    v.add_argument("--head", choices=["tdt", "ctc"])
    v.add_argument("--unit", choices=["segment", "word"])
    v.add_argument("--resume", action="store_true",
                   help="skip input files that already have a result in OUT_DIR/<data dir> (interrupted run / new files)")
    v.add_argument("--retry-failed", action="store_true",
                   help="like --resume, but files in failed/ (errors) are run again")
    v.add_argument("--out-dir", "--output-dir", dest="out_dir",
                   help="root output dir (config out_dir): writes OUT_DIR/<data dir>/{verbalized,verify,failed}/"
                        "<stem>.json, verbalized.tsv, verify.tsv, failed.tsv, summary.json, report.html")

    c = sub.add_parser("compare", help="compare tdt/ctc x segment/word on a dir; write a report")
    _scoring_args(c)
    c.add_argument("--hyps-dir", action="append", default=[],
                   help="hypotheses dir (<stem>.jsonl) for agreement stats; repeat for several "
                        "(each reported under its dir name)")
    c.add_argument("--transcribe", action="store_true",
                   help="also transcribe each sentence with the model and report agreement with that")
    c.add_argument("--out-root", default="outputs")
    c.add_argument("--hyp-pad", type=float, dest="hyp_pad_s",
                   help="seconds of hypothesis text searched around each sentence (config hyp_pad_s)")

    w = sub.add_parser("sweep", help="time compare while changing batch_size / joint_chunk_frames / "
                                     "encoder_batch_s / group_audio_s one at a time; recommend the fastest")
    _scoring_args(w)
    w.add_argument("--sweep-config", help="sweep YAML/JSON: grid of values per setting + options "
                                          "(see examples/digits_sweep.yaml); flags below override it")
    from .scoring.sweep import GRID, PARAMS
    for p in PARAMS:
        dflt = ",".join(map(str, GRID[p])) if p in GRID else "not swept"
        w.add_argument(f"--{p.replace('_', '-')}-values", dest=f"{p}_values",
                       help=f"comma-separated values to try (built-in default: {dflt}; "
                            "'none' = skip this parameter)")
    w.add_argument("--hyps-dir", action="append", default=[], help="as for compare (adds accuracy columns)")
    w.add_argument("--out-root", default=None, help="default outputs/sweep/<data dir name>")
    w.add_argument("--greedy", action="store_true", default=None,
                   help="carry each parameter's best value into the next parameter's runs")
    w.add_argument("--max-changed", type=int,
                   help="a value is accepted only if at most this many picks differ from the baseline (default 0)")
    w.add_argument("--min-gain", type=float,
                   help="replace a config value only if at least this much faster (default 0.03 = 3%%)")
    w.add_argument("--warmup-files", type=int, help="untimed warm-up run on N files (default 2; 0 = none)")

    r = sub.add_parser("verifier", help="check verify/ + failed/ files of a verbalize run in the browser "
                                        "(no GPU; Windows / Linux)")
    r.add_argument("run_dir", nargs="?", help="OUT_DIR/<data dir> of a verbalize run (or choose it in the page)")
    r.add_argument("--audio-dir", help="folder with <stem>.wav (or choose it in the page)")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")

    k = sub.add_parser("check", help="sanity-check the model scorer on a few sentences")
    _scoring_args(k)
    k.add_argument("--sentences", type=int, default=8)

    r = sub.add_parser("reverify", help="re-score files in verbalized/ with the current candidates/model; "
                                        "files with doubtful numbers go to reverify/ for the verifier")
    r.add_argument("run_dir", help="a verbalize run dir: OUT_DIR/<data dir>")
    r.add_argument("audio_dir", help="dir with <stem>.wav for each transcript")
    r.add_argument("--config", required=True, help="YAML/JSON scoring config (current settings are used)")
    r.add_argument("--model-path")
    r.add_argument("--cuda", type=int)
    r.add_argument("--batch-size", type=int)
    r.add_argument("--limit", type=int, help="only the first N files")
    r.add_argument("--data-dir", help="original input dir (default: <run>/input/, then summary.json data_dir)")
    r.add_argument("--files", help="comma-separated stems to check (default: all of verbalized/)")
    r.add_argument("--dry-run", action="store_true", help="only write reverify_report.tsv / .json, move nothing")
    r.add_argument("--no-refresh", action="store_true",
                   help="don't re-score the open numbers of files waiting in verify/ and reverify/")
    stt = sub.add_parser("status", help="is a run complete? every input verbalized, nothing open (no GPU)")
    stt.add_argument("run_dir", help="OUT_DIR/<data dir>")
    stt.add_argument("--data-dir", help="input dir (default: summary.json data_dir)")
    stt.add_argument("--show", type=int, default=20, help="list at most this many files per problem")
    stt.add_argument("--json", action="store_true", help="print the full report as JSON")
    fx = sub.add_parser("fix", help="repair a run dir: leftover copies, stale TSV rows, .trash, summary counts (no GPU)")
    fx.add_argument("run_dir", help="OUT_DIR/<data dir>")
    fx.add_argument("--dry-run", action="store_true", help="only report what would be fixed")
    t = sub.add_parser("text", help="verbalize text with default readings (no audio)")
    t.add_argument("text")
    t.add_argument("--style", default="contracted", choices=["contracted", "glide"])
    t.add_argument("--candidates", action="store_true", help="also list candidates per number")

    a = ap.parse_args(argv)
    if a.cmd == "stats":
        stats.run(a)
    elif a.cmd == "latin":
        latin_stats.run(a)
    elif a.cmd == "reverify":
        from .scoring.reverify import reverify
        cfg = _cfg(a)
        if not os.path.exists(os.path.join(a.run_dir, "verbalized.tsv")):
            sys.exit(f"{a.run_dir}: no verbalized.tsv (give OUT_DIR/<data dir> of a verbalize run)")
        tot = reverify(a.run_dir, a.audio_dir, _scorer(cfg), cfg, data_dir=a.data_dir,
                       files=[s.strip() for s in a.files.split(",")] if a.files else None,
                       limit=a.limit, dry_run=a.dry_run, refresh=not a.no_refresh)
        print(json.dumps(tot, ensure_ascii=False, indent=2))
        print(f"checked {tot['files_checked']} files / {tot['numbers_checked']} numbers: {tot['numbers_flagged']} "
              f"flagged ({tot['flagged_auto']} auto, {tot['flagged_verified']} verified) in {tot['files_flagged']} "
              f"files" + (" (dry run)" if a.dry_run else " -> reverify/") + f"; report {a.run_dir}/reverify_report.tsv")
        w = tot.get("waiting")
        if w:
            print(f"waiting files (verify/ + reverify/): {w['files']} re-scored, {w['numbers_refreshed']} open "
                  f"numbers updated, {w['picks_changed']} with a new pick")
        print(f"typed readings found {tot['typed_readings_found']}, added {tot['typed_readings_added']} "
              f"-> {a.run_dir}/typed_readings.tsv")
    elif a.cmd == "status":
        from .scoring.status import print_status, run_status
        rep = run_status(a.run_dir, a.data_dir)
        if a.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        else:
            print_status(rep, a.show)
        sys.exit(0 if rep["complete"] else 1)
    elif a.cmd == "fix":
        from .scoring.fixrun import fix_run
        rep = fix_run(a.run_dir, dry_run=a.dry_run)
        if not any(v for k, v in rep.items() if k != "summary"):
            print("nothing to fix")
        if a.dry_run:
            print("(dry run: nothing changed)")
    elif a.cmd == "verbalize":
        from .scoring.pipeline import run_dir, verbalize_dir
        cfg = _cfg(a, head=a.head, unit=a.unit, out_dir=a.out_dir)
        run_dir(a.data_dir, cfg)                 # fail before loading the model if out_dir is missing
        tot = verbalize_dir(a.data_dir, a.audio_dir, _scorer(cfg), cfg, limit=a.limit, resume=a.resume,
                            retry_failed=a.retry_failed)
        print(json.dumps(tot, ensure_ascii=False, indent=2))
        r, t = tot["this_run"], tot["totals"]
        print(f"this run: {r['files']} files -> verbalized {r['verbalized']} · verify {r['verify']} · failed "
              f"{r['failed']}" + (f" ({tot['resumed_skipped']} already done, skipped)" if (a.resume or a.retry_failed) else ""))
        print(f"run dir now: verbalized {t['verbalized']} · verify {t['verify']} · reverify {t['reverify']} · "
              f"failed {t['failed']} -> {tot['out_dir']}")
    elif a.cmd == "compare":
        from .scoring.compare import compare_dir
        cfg = _cfg(a, hyp_pad_s=a.hyp_pad_s)
        rep = compare_dir(a.data_dir, a.audio_dir, _scorer(cfg), cfg, a.out_root, a.hyps_dir or None,
                          a.limit, transcribe=a.transcribe)
        print(f"wrote {rep['out_dir']}/report.html")
    elif a.cmd == "sweep":
        from .scoring.sweep import PARAMS, load_sweep_config, sweep
        sc = load_sweep_config(a.sweep_config, grid={p: getattr(a, f"{p}_values") for p in PARAMS},
                               hyps_dirs=a.hyps_dir, limit=a.limit, out_root=a.out_root, greedy=a.greedy,
                               max_changed=a.max_changed, min_gain=a.min_gain, warmup_files=a.warmup_files)
        cfg = _cfg(a, **sc["base"])            # sweep config "base" overrides the scoring config
        out = sc["out_root"] or os.path.join("outputs", "sweep", os.path.basename(os.path.normpath(a.data_dir)))
        res = sweep(a.data_dir, a.audio_dir, _scorer(cfg), cfg, out, sc["grid"], sc["hyps_dirs"] or None,
                    sc["limit"], greedy=bool(sc["greedy"]), max_changed=sc["max_changed"],
                    warmup_files=sc["warmup_files"], min_gain=sc["min_gain"], make_scorer=_scorer)
        print(json.dumps({"best": res["best_settings"], "speedup": res["speedup_final"]}, indent=2))
        print(f"wrote {out}/sweep.html, best_config.yaml")
    elif a.cmd == "verifier":
        from .verifier.server import serve
        serve(a.run_dir, a.audio_dir, a.port, open_browser=not a.no_browser)
    elif a.cmd == "check":
        return _check(a)
    elif a.cmd == "text":
        from . import candidates, find, verbalize
        from ..languages import malayalam
        txt = malayalam.normalize(a.text, mode="train")
        print(verbalize(txt, style=a.style))
        if a.candidates:
            for sp in find(txt):
                print(f"  {sp.core} [{sp.cls}]: " + " | ".join(f"{c.text} ({c.lang},{c.form})"
                                                           for c in candidates(sp, style=a.style)))
    return 0


def _audio_info(path):
    try:
        import soundfile as sf
        i = sf.info(path)
        return {"sample_rate": i.samplerate, "channels": i.channels, "seconds": round(i.duration, 2)}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def _check(a):
    """Sanity checks before compare/verbalize. Prints:
      1. the audio format and whether the transcript timestamps fit the wav length;
      2. real text vs the SAME words shuffled (model + scoring code sane);
      3. real text vs ANOTHER sentence's text on this audio (audio <-> timestamps aligned: if the
         segment audio is the wrong place, real text stops winning);
      4. batch vs one-by-one scores;
      5. a few model transcripts next to the reference (script / language sanity)."""
    import random
    from .scoring.audio import find_audio, load_wav, slice_audio
    from .scoring.pipeline import _sentences, list_sources
    from . import verbalize
    from ..languages import malayalam
    cfg = _cfg(a)
    scorer = _scorer(cfg)
    print(f"model heads: {scorer.heads}  device: {scorer.device}  dtype: {scorer.dtype}")
    items, shown = [], 0
    for src in list_sources(a.data_dir):
        stem = os.path.splitext(os.path.basename(src))[0]
        try:
            wav = find_audio(a.audio_dir, stem)
        except FileNotFoundError:
            continue
        doc = json.load(open(src, encoding="utf-8"))
        sents = _sentences(doc)
        if shown < 3:
            last = max((float(s.get("end_time") or 0) for s in sents), default=0)
            print(f"audio {os.path.basename(wav)}: {_audio_info(wav)}; last timestamp {last:.2f}s")
            shown += 1
        x = load_wav(wav, cfg.sample_rate)
        for s in sents:
            t = verbalize(malayalam.normalize(s.get("text") or "", mode="train"))
            if len(t.split()) >= 4 and s.get("end_time") is not None:
                seg = slice_audio(x, float(s["start_time"]), float(s["end_time"]), cfg.sample_rate, cfg.pad_s)
                items.append((seg, t))
            if len(items) >= a.sentences:
                break
        if len(items) >= a.sentences:
            break
    if len(items) < 2:
        print("no usable sentences found")
        return 1
    n = len(items)
    shuffled, other = [], []
    for k, (seg, t) in enumerate(items):
        w = t.split()
        random.Random(k).shuffle(w)
        shuffled.append((seg, " ".join(w)))
        other.append((seg, items[(k + 1) % n][1]))          # another sentence's text, same audio
    ok = True
    for h in scorer.heads:
        real = scorer.score(items, h)
        shuf = scorer.score(shuffled, h)
        oth = scorer.score(other, h)
        one = [scorer.score([p], h)[0] for p in items]
        w_shuf = sum(r < u for r, u in zip(real, shuf))
        w_oth = sum(r < u for r, u in zip(real, oth))
        drift = max(abs(r - o) for r, o in zip(real, one))
        print(f"[{h}] real < shuffled {w_shuf}/{n}; real < other sentence's text {w_oth}/{n}; mean NLL "
              f"real {sum(real)/n:.2f} / shuffled {sum(shuf)/n:.2f} / other {sum(oth)/n:.2f}; "
              f"batch vs single max diff {drift:.4f}")
        good = w_shuf >= 0.8 * n and w_oth >= 0.75 * n and drift < 0.05 * max(1.0, max(map(abs, real)))
        if h == "tdt" or len(scorer.heads) == 1:
            ok &= good                    # the TDT head is the one verbalize uses by default
        elif not good:
            print(f"  note: the {h} head is weaker here (informational; it does not fail the check)")
    if hasattr(scorer, "transcribe"):
        print("model transcripts (reference | model):")
        for (seg, t), hyp in zip(items[:5], scorer.transcribe([seg for seg, _ in items[:5]])):
            print(f"  {t[:70]}\n  {hyp[:70]}")
    print("OK" if ok else "CHECK FAILED - scores look wrong; please send me this output")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
