"""Speed sweep: find the fastest ``batch_size`` / ``joint_chunk_frames`` / ``encoder_batch_s`` /
``group_audio_s`` by changing ONE value at a time and running ``compare_dir`` for each.

The model is loaded once. Run order:
    1. warm-up (``warmup_files`` files, not timed) — CUDA kernels / allocator / file cache;
    2. baseline = the config as given (all four at their config values);
    3. for each parameter, each value in its grid (smallest first) with the others at the baseline
       values (``greedy=True``: at the best values found so far instead); if a value runs out of GPU
       memory, the larger values are skipped and the sweep moves on to the next parameter;
    4. ``final`` = every parameter at its best value (checks the combination).
Each run records wall-clock per stage, real-time factor, scorer encode / decode seconds, peak GPU
memory, OOM, and how many numbers' picks differ from the baseline (per method). A value is *safe*
when it did not OOM and changed at most ``max_changed`` picks; the best value is the fastest safe
one (total seconds).

Outputs in ``<out_root>/``: ``sweep.html`` (sortable tables + recommended config), ``sweep.json``, ``sweep.tsv``,
``best_config.yaml`` and each run's compare report in ``<param>_<value>/``.
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional

from .compare import compare_dir
from .config import ScoringConfig

PARAMS = ("batch_size", "joint_chunk_frames", "encoder_batch_s", "group_audio_s", "compute_dtype")
RELOAD = {"compute_dtype"}                     # changing these needs a model reload (make_scorer)
GRID = {
    "batch_size": [4, 8, 16, 32, 64],
    "joint_chunk_frames": [8, 16, 24, 32, 64, 128],
    "encoder_batch_s": [30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0],
    "group_audio_s": [0.0, 300.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0],
}
# compute_dtype is sweepable (grid: {compute_dtype: [float32, bfloat16]}) but not by default: float32 is
# the default; bf16 is ~1.4x faster but its picks depend on how clips are batched.
_CAST = {"batch_size": int, "joint_chunk_frames": int, "encoder_batch_s": float, "group_audio_s": float}


def parse_values(param: str, text: str) -> list:
    if param not in _CAST:                       # text-valued (compute_dtype)
        return [v.strip() for v in text.split(",") if v.strip()]
    return [_CAST[param](float(v)) for v in text.split(",") if v.strip()]


SWEEP_DEFAULTS = {"base": {}, "grid": GRID, "greedy": False, "max_changed": 0, "min_gain": 0.03, "warmup_files": 2,
                  "hyps_dirs": [], "limit": None, "out_root": None}


def load_sweep_config(path: Optional[str] = None, **overrides) -> dict:
    """Read a sweep YAML/JSON file (see examples/digits_sweep.yaml) and apply non-None overrides.

    ``base``: scoring-config keys applied before the sweep (the baseline), e.g. {compute_dtype: bfloat16}.
    ``grid``: {parameter: [values]}; a parameter that is missing, null, [] or "none" is not swept.
    Without a file (or without ``grid``) the built-in GRID is used."""
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith((".yaml", ".yml")):
            import yaml
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"sweep config {path} must be a mapping")
    unknown = set(data) - set(SWEEP_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown sweep config keys: {sorted(unknown)} (known: {sorted(SWEEP_DEFAULTS)})")
    out = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in SWEEP_DEFAULTS.items()}
    if "grid" in data:
        g = data["grid"] or {}
        bad = set(g) - set(PARAMS)
        if bad:
            raise ValueError(f"unknown sweep grid parameters: {sorted(bad)} (known: {list(PARAMS)})")
        out["grid"] = {}
        for p, v in g.items():
            if v is None or v == [] or (isinstance(v, str) and v.strip().lower() == "none"):
                continue
            vals = parse_values(p, ",".join(map(str, v)) if isinstance(v, list) else str(v))
            out["grid"][p] = vals
    for k, v in data.items():
        if k != "grid":
            out[k] = v
    if not isinstance(out["base"] or {}, dict):
        raise ValueError("sweep config 'base' must be a mapping of scoring-config keys")
    out["base"] = dict(out["base"] or {})
    if isinstance(out["hyps_dirs"], str):
        out["hyps_dirs"] = [out["hyps_dirs"]]
    for k, v in overrides.items():
        if k == "grid":
            for p, txt in (v or {}).items():
                if txt is None:
                    continue
                if txt.strip().lower() == "none":
                    out["grid"].pop(p, None)
                else:
                    out["grid"][p] = parse_values(p, txt)
        elif v is not None and v != []:
            out[k] = v
    return out


def _margins(rows, method):
    return [r.get(method + ":margin") for r in rows]


def _drift(base_rows, rows, method, cap=20.0):
    """median / max |margin - baseline margin| (nats) for one method, over margins < cap (huge
    margins are far from any decision and carry large absolute rounding)."""
    ds = sorted(abs(a - b) for a, b in zip(_margins(base_rows, method), _margins(rows, method))
                if a is not None and b is not None and a < cap)
    if not ds:
        return None, None
    return round(ds[len(ds) // 2], 4), round(ds[-1], 4)


def _review_flips(base_rows, rows, methods):
    return {m: sum(a.get(m + ":review") != b.get(m + ":review") for a, b in zip(base_rows, rows))
            for m in methods}


def _apply(scorer, cfg: ScoringConfig):
    """Point an already-loaded scorer at new batching settings (no model reload)."""
    if hasattr(scorer, "cfg"):
        scorer.cfg = cfg
    if hasattr(scorer, "batch_size"):
        scorer.batch_size = cfg.batch_size
    if hasattr(scorer, "chunk"):
        scorer.chunk = int(cfg.joint_chunk_frames)


def _torch():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover
        return None


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _picks(rows, methods) -> Dict[tuple, Dict[str, str]]:
    out, seen = {}, Counter()
    for r in rows:
        k = (r["file"], r["sentence"], r["core"])
        seen[k] += 1
        out[k + (seen[k],)] = {m: r.get(m) for m in methods}
    return out


def _diff(base, cur, methods) -> Dict[str, int]:
    d = {m: 0 for m in methods}
    for k, b in base.items():
        c = cur.get(k)
        for m in methods:
            if c is None or c[m] != b[m]:
                d[m] += 1
    return d


def _label(param, value):
    if isinstance(value, str):
        return f"{param}_{value}"
    v = int(value) if float(value).is_integer() else value
    return f"{param}_{v}"


def sweep(data_dir: str, audio_dir: str, scorer, cfg: ScoringConfig, out_root: str = "sweeps",
          grid: Optional[Dict[str, list]] = None, hyps_dirs=None, limit: Optional[int] = None,
          greedy: bool = False, max_changed: int = 0, warmup_files: int = 2, min_gain: float = 0.03, log=print,
          compare_log=None, make_scorer=None) -> dict:
    """``make_scorer(cfg)`` builds a scorer; needed only for RELOAD parameters (compute_dtype)."""
    grid = {p: list(v) for p, v in (grid or GRID).items() if p in PARAMS and v}
    if make_scorer is None:
        for p in RELOAD & set(grid):
            log(f"{p}: skipped (needs make_scorer to reload the model)")
            grid.pop(p)
    os.makedirs(out_root, exist_ok=True)
    torch = _torch()
    quiet = compare_log or (lambda *_: None)
    runs: List[dict] = []
    cache: Dict[tuple, dict] = {}
    base_vals = {p: getattr(cfg, p) for p in PARAMS}
    state = {"base_picks": None, "methods": None, "base_rows": None,
             "loaded": {p: getattr(cfg, p) for p in RELOAD}}
    primary = f"{cfg.head}/{cfg.unit}"

    def warmup(c, tag):
        if not warmup_files:
            return
        log(f"warm-up ({tag}) on {warmup_files} files")
        try:
            compare_dir(data_dir, audio_dir, scorer, c, os.path.join(out_root, "_warmup"), hyps_dirs,
                        warmup_files if limit is None else min(limit, warmup_files), log=quiet)
        except Exception as e:  # pragma: no cover
            log(f"  warm-up failed: {e}")

    def reload_if_needed(c):
        nonlocal scorer
        want = {p: getattr(c, p) for p in RELOAD}
        if want == state["loaded"]:
            return
        log(f"  reloading model for {want}")
        scorer = None
        gc.collect()
        if torch:
            torch.cuda.empty_cache()
        scorer = make_scorer(c)
        state["loaded"] = want
        warmup(c, json.dumps(want))

    def run(label, vals, param=None):
        key = tuple(sorted(vals.items()))
        if key in cache:                           # same settings already measured: reuse
            r = dict(cache[key], label=label, param=param, value=vals.get(param), reused=True)
            runs.append(r)
            return r
        c = dataclasses.replace(cfg, **vals).validate()
        try:
            reload_if_needed(c)
        except Exception as e:
            r = {"label": label, "param": param, "value": vals.get(param), "settings": dict(vals),
                 "reused": False, "oom": _is_oom(e), "error": f"reload: {type(e).__name__}: {str(e)[:200]}"}
            log(f"  {label}: {r['error']}")
            runs.append(r)
            cache[key] = r
            return r
        _apply(scorer, c)
        r = {"label": label, "param": param, "value": vals.get(param), "settings": dict(vals), "reused": False}
        if torch:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            rep = compare_dir(data_dir, audio_dir, scorer, c, os.path.join(out_root, label),
                              hyps_dirs, limit, log=quiet)
        except Exception as e:                    # OOM (or anything else) -> record, keep going
            if torch:
                torch.cuda.empty_cache()
            r.update(oom=_is_oom(e), error=f"{type(e).__name__}: {str(e)[:200]}",
                     total_s=round(time.perf_counter() - t0, 2))
            log(f"  {label}: {'OOM' if r['oom'] else 'ERROR'} {r['error']}")
            runs.append(r)
            cache[key] = r
            return r
        tm = rep.get("timing", {})
        st = tm.get("stages_s", {})
        sc = tm.get("scorer", {})
        rows = rep.pop("rows", [])
        methods = list(rep.get("methods", {}))
        picks = _picks(rows, methods)
        if state["base_picks"] is None:
            state["base_picks"], state["methods"], state["base_rows"] = picks, methods, rows
        changed = _diff(state["base_picks"], picks, state["methods"])
        med, mx = _drift(state["base_rows"], rows, primary) if primary in methods else (None, None)
        flips = _review_flips(state["base_rows"], rows, state["methods"])
        r.update(oom=False, error=None, total_s=st.get("total"),
                 score_s={k[6:]: v for k, v in st.items() if k.startswith("score_")},
                 load_and_plan_s=st.get("load_and_plan"), rtf=tm.get("rtf_total"),
                 numbers=tm.get("numbers"), numbers_per_s=tm.get("numbers_per_s"),
                 encode_s=sc.get("encode_s"), decode_s=sc.get("decode_s"),
                 encoder_batches=sc.get("encoder_batches"), decoder_batches=sc.get("decoder_batches"),
                 peak_gpu_gb=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch else None),
                 changed=changed, changed_total=sum(changed.values()),
                 margin_drift_median=med, margin_drift_max=mx,
                 review_flips=flips, review_flips_total=sum(flips.values()),
                 accuracy={m: {ref: x["given_oracle_pct"] for ref, x in v["match"].items()}
                           for m, v in rep.get("methods", {}).items()},
                 out_dir=rep.get("out_dir"))
        log(f"  {label}: {r['total_s']} s (score {r['score_s']}), peak {r['peak_gpu_gb']} GB, "
            f"changed picks {r['changed_total']}, {primary} margin drift {med}/{mx}, "
            f"review flips {r['review_flips_total']}")
        runs.append(r)
        cache[key] = r
        return r

    _apply(scorer, cfg)
    warmup(cfg, "start")
    log("baseline " + json.dumps(base_vals))
    base = run("baseline", dict(base_vals))
    if base.get("error"):
        raise RuntimeError(f"baseline run failed: {base['error']}")

    current = dict(base_vals)
    best: Dict[str, dict] = {}
    for p, values in grid.items():
        if p in _CAST:
            values = sorted(values)                # small -> large, so an OOM ends the setting
        log(f"{p}: {values}")
        trial = []
        for i, v in enumerate(values):
            vals = dict(current if greedy else base_vals, **{p: v})
            t = run(_label(p, v), vals, p)
            trial.append(t)
            if t.get("oom") and p in _CAST:        # larger values need more memory: stop here
                for w in values[i + 1:]:
                    runs.append({"label": _label(p, w), "param": p, "value": w, "skipped": True,
                                 "oom": False, "error": f"skipped: {p}={v} ran out of memory"})
                    log(f"  {_label(p, w)}: skipped (OOM at {v})")
                break
        safe = [t for t in trial if not t.get("error") and t["changed_total"] <= max_changed]
        ref_t = next((t["total_s"] for t in trial if t["value"] == (current if greedy else base_vals)[p]
                      and not t.get("error")), base["total_s"])
        safe = [t for t in safe if t["value"] == (current if greedy else base_vals)[p]
                or t["total_s"] < ref_t * (1 - min_gain)] or safe   # must beat the config by min_gain
        if safe:
            b = min(safe, key=lambda t: (t["value"] != (current if greedy else base_vals)[p]
                                         and t["total_s"] >= ref_t * (1 - min_gain), t["total_s"]))
            best[p] = {"value": b["value"], "total_s": b["total_s"],
                       "speedup_vs_baseline": round(base["total_s"] / b["total_s"], 3) if b["total_s"] else None}
            if greedy:
                current[p] = b["value"]
        else:
            best[p] = {"value": base_vals[p], "total_s": base["total_s"], "note": "no safe value; kept config"}

    best_vals = {p: best[p]["value"] if p in best else base_vals[p] for p in PARAMS}
    final = run("final", best_vals) if best_vals != base_vals else base
    res = {"primary_method": primary, "data_dir": data_dir, "limit": limit, "greedy": greedy, "max_changed": max_changed,
           "min_gain": min_gain,
           "baseline": base_vals, "best": best, "best_settings": best_vals,
           "final": {k: final.get(k) for k in ("total_s", "rtf", "peak_gpu_gb", "changed_total", "error")},
           "speedup_final": (round(base["total_s"] / final["total_s"], 3)
                             if final.get("total_s") and not final.get("error") else None),
           "runs": runs}
    _write(res, out_root, cfg)
    return res


def _write(res, out_root, cfg):
    with open(os.path.join(out_root, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    cols = ["label", "param", "value", "reused", "skipped", "oom", "error", "total_s", "rtf", "numbers_per_s",
            "load_and_plan_s", "encode_s", "decode_s", "encoder_batches", "decoder_batches",
            "peak_gpu_gb", "changed_total",
            "margin_drift_median", "margin_drift_max", "review_flips_total"]
    with open(os.path.join(out_root, "sweep.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        stages = sorted({k for r in res["runs"] for k in (r.get("score_s") or {})})
        w.writerow(cols + [f"score_{s}_s" for s in stages] + ["settings"])
        for r in res["runs"]:
            w.writerow([r.get(c) for c in cols] + [(r.get("score_s") or {}).get(s) for s in stages]
                       + [json.dumps(r.get("settings"))])
    with open(os.path.join(out_root, "best_config.yaml"), "w", encoding="utf-8") as f:
        f.write("# fastest safe values from indic-digits sweep (merge into your scoring config)\n")
        for p, v in res["best_settings"].items():
            f.write(f"{p}: {v}\n")
    base_t = next(r["total_s"] for r in res["runs"] if r["label"] == "baseline")
    L = ["# Scoring speed sweep", "",
         f"Data `{res['data_dir']}`" + (f" (first {res['limit']} files)" if res["limit"] else "") +
         f" · {'greedy (best so far carried forward)' if res['greedy'] else 'one at a time from the baseline'}"
         f" · safe = no OOM and ≤ {res['max_changed']} changed picks · a value replaces the config value only if"
         f" ≥ {round(100 * res['min_gain'])}% faster", "",
         f"Baseline {json.dumps(res['baseline'])}: **{base_t} s**.", "",
         f"Accuracy = `{res['primary_method']}` matches given oracle per reference; margin drift = median / max "
         f"|margin − baseline margin| of `{res['primary_method']}` in nats (margins < 20); review flips = "
         "needs_review changes vs baseline, all methods.", "",
         "| Run | Value | Total s | vs baseline | Encode s | Decode s | Score s (per unit) | Peak GPU GB | Changed picks "
         "| Accuracy | Margin drift | Review flips |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        if r.get("error"):
            what = "skipped (OOM below)" if r.get("skipped") else ("OOM" if r.get("oom") else "error")
            L.append(f"| {r['label']} | {r.get('value')} | {what} | | | | | | | | | |")
            continue
        sp = f"{base_t / r['total_s']:.2f}x" if r.get("total_s") else ""
        L.append(f"| {r['label']}{' (reused)' if r.get('reused') else ''} | {r.get('value')} | {r['total_s']} | {sp} | "
                 f"{r.get('encode_s')} | {r.get('decode_s')} | "
                 f"{', '.join(f'{k} {v}' for k, v in (r.get('score_s') or {}).items())} | {r.get('peak_gpu_gb')} | "
                 f"{r['changed_total']} | "
                 f"{' / '.join(str(v) for v in (r.get('accuracy') or {}).get(res['primary_method'], {}).values())} | "
                 f"{r.get('margin_drift_median')} / {r.get('margin_drift_max')} | {r.get('review_flips_total')} |")
    L += ["", "## Best per parameter", "", "| Parameter | Baseline | Best | Total s | Speed-up |", "|---|---|---|---|---|"]
    for p, b in res["best"].items():
        L.append(f"| {p} | {res['baseline'][p]} | **{b['value']}** | {b['total_s']} | "
                 f"{b.get('speedup_vs_baseline') or b.get('note', '')} |")
    fin = res["final"]
    L += ["", f"Combined best {json.dumps(res['best_settings'])}: "
          + (f"**{fin['total_s']} s** ({res['speedup_final'] or '-'}x), peak {fin['peak_gpu_gb']} GB, "
             f"changed picks {fin['changed_total']}." if not fin.get("error") else f"failed: {fin['error']}"),
          "", "Changed picks = numbers whose pick (any method) differs from the baseline run; small "
          "non-zero counts can come from float rounding in a different batch shape — check "
          "`<run>/…/numbers.tsv` before accepting them.", "",
          "Timing is noisy (other GPU jobs, disk cache); differences under ~5% are not meaningful.", ""]
    from .html_report import page
    with open(os.path.join(out_root, "sweep.html"), "w", encoding="utf-8") as f:
        f.write(page("Scoring speed sweep", "\n".join(L)))
