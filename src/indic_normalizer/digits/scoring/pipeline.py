"""Choose how each number was spoken by scoring its candidate readings against the audio.

Input: a transcript directory like eval_frozen (one JSON per recording with ``sentences``
[{text, start_time, end_time, ...}] and ``words`` [{word, start_time, end_time}]) and an audio
directory with ``<same stem>.wav``.

For every sentence the text is first normalized with the core normalizer in ``train`` mode
(punctuation stripped except inside numbers), numbers are found with ``digits.find`` and their
readings listed with ``digits.candidates``. Each candidate is then scored with the ASR model:

    unit = "segment"  the whole sentence audio, with that number's candidate substituted and every
                      other number at its default reading (robust to timestamp error, uses context)
    unit = "word"     only the number's own word(s) +- ``context_words`` neighbours, cut out by word
                      timestamps and encoded ON THEIR OWN (no acoustic context outside the window).
    unit = "window"   the WHOLE sentence is encoded once (full acoustic context, shared with the
                      segment unit), then each candidate is scored only on the encoder frames of
                      the number's word window (+- ``context_words`` words, +- ``pad_s``) with the
                      window's text. Accuracy of segment context, cost of a short window.
    "word" and "window" fall back to "segment" for a sentence whose numbers can't be matched
    to timestamped words.

The candidate with the lowest NLL wins. ``margin`` = runner-up NLL - best NLL. ``review_margin`` =
NLL of the best candidate from a DIFFERENT reading family - best NLL, where a family ignores
spelling/word-boundary style (contracted ഇരുപത്തഞ്ച്, glide ഇരുപത്തിയഞ്ച് and spaced ഇരുപത്തി അഞ്ച്
are one family; English, drop-thousand, half, ... are others). Near-ties inside a family are
harmless for the target (all are faithful), so only ``review_margin < margin_threshold`` flags the
number (and its sentence/file) ``needs_review``.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
from typing import Dict, List, Optional

from ...languages import malayalam as _ml
from .. import candidates as _candidates
from .. import find as _find
from .. import verbalize as _verbalize
from .details import merge as merge_details
from .details import number_rows
from .verbalize_report import report_rows
from .audio import find_audio, load_wav, slice_audio
from .config import ScoringConfig
from .. import ml
from ..detect import to_ascii_digits
from ..latin import is_english as _is_english, sentence_is_english as _sent_english
from .. import next_word as _next_word
from .. import alone as _alone


def _doc_text(doc) -> str:
    return " ".join(str(s.get("text") or "") for s in _sentences(doc))


def doc_languages(doc, cfg):
    """[(latin: bool, review reason or None)] per sentence — see latin.sentence_languages."""
    from ..latin import sentence_languages
    langs = sentence_languages([str(s.get("text") or "") for s in _sentences(doc)],
                               getattr(cfg, "dominant_language_share", 0.8))
    return [(lang == "en", why) for lang, why in langs]


def _doc_is_english(doc) -> bool:
    """Language of the whole document, for sentences without letters ("155.", "0031.")."""
    return _is_english(" ".join(str(s.get("text") or "") for s in _sentences(doc)))


# --------------------------------------------------------------------------- data loading

def list_sources(data_dir: str, suffix: str = ".verbalized.json") -> List[str]:
    """Transcript JSONs in ``data_dir`` (``*.verbalized.json`` left by old versions are skipped)."""
    out = sorted(p for p in glob.glob(os.path.join(data_dir, "*.json")) if not p.endswith(suffix))
    if not out:
        raise FileNotFoundError(f"no .json transcripts in {data_dir}")
    return out


def _sentences(doc: dict) -> List[dict]:
    if doc.get("sentences"):
        return doc["sentences"]
    if doc.get("segments"):
        return doc["segments"]
    words = doc.get("words") or []
    end = max((float(w.get("end_time", 0) or 0) for w in words), default=None)
    return [{"text": doc.get("transcription", ""), "start_time": 0.0, "end_time": end}]


def _words_in(doc: dict, s: dict) -> List[dict]:
    try:
        a, b = float(s["start_time"]), float(s["end_time"])
    except (KeyError, TypeError, ValueError):
        return []
    return [w for w in (doc.get("words") or [])
            if w.get("start_time") is not None and a - 1e-3 <= float(w["start_time"]) <= b + 1e-3]


# --------------------------------------------------------------------------- planning

class Number:
    """One number in one sentence and the scoring jobs for its candidates."""

    def __init__(self, sent_idx, span, cands, default_text):
        self.sent_idx, self.span, self.cands = sent_idx, span, cands
        self.default_text = default_text
        self.jobs: Dict[str, list] = {}          # unit -> [(audio, text)] aligned with cands
        self.unit_used: Dict[str, str] = {}
        self.word_idx: Optional[List[int]] = None
        self.scores: Dict[tuple, list] = {}      # (unit, head) -> nll per candidate
        self.text_issue: Optional[str] = None    # see context_issue()

    def as_dict(self):
        return {"core": self.span.core, "class": self.span.cls, "token_index": self.span.token_index,
                "word_index": self.word_idx}


def _sub(text_train: str, spans, j: int, cand_text: str, style: str, latin: bool = False) -> str:
    """Sentence text with number j read as cand_text and every other number at default."""
    k = iter(range(len(spans)))

    def chooser(span, cands):
        i = next(k)
        return cand_text if i == j else 0
    return _verbalize(text_train, style=style, chooser=chooser, latin=latin)


CURRENCY_WORDS = ("₹", "rs", "inr")


def currency_issue(span) -> Optional[str]:
    """Every amount with a currency sign (₹ / Rs / INR, joined or written apart) is reviewed:
    whether രൂപ / റുപ്പീസ് was said is not certain."""
    if span.cls == "currency" or re.match(rf"(?i)^{_SIGN}\d|^\d.*₹$", span.core):
        return f"currency sign ({span.core})"
    return None


def context_issue(text: str, span) -> Optional[str]:
    """A problem in the text AROUND a number that its reading can't fix, so a person should
    look: a currency sign written as a separate word (₹ 500, Rs. 500, INR 500, 500 ₹) — the
    number is read alone and the sign is left in the text."""
    before = text[:span.start].split()
    after = text[span.end:].split()
    prev = before[-1].lower().rstrip(".") if before else ""
    nxt = after[0].lower().rstrip(".") if after else ""
    if prev in CURRENCY_WORDS:
        return f"currency sign written apart ({before[-1]} {span.core})"
    if nxt == "₹":
        return f"currency sign written apart ({span.core} {after[0]})"
    return None


from ..detect import SIGN as _SIGN, join_currency  # noqa: E402
_SIGN_WORD = re.compile(rf"(?i){_SIGN}")


def bare_core(core: str) -> str:
    """A number core without its currency sign (₹70000 / rs500 / 500₹ -> 70000 / 500)."""
    return re.sub(rf"(?i)^{_SIGN}|₹$", "", core or "")


def _word_map(words, spans):
    """Position in ``words`` (the sentence's timed words) of each span, or None if they don't
    line up with the sentence's numbers."""
    if not words:
        return None
    flat = [(i, sp.core) for i, w in enumerate(words)
            for sp in _find(_ml.normalize(str(w.get("word", "")), mode="train"))]
    if [bare_core(c) for _, c in flat] != [bare_core(sp.core) for sp in spans]:
        return None
    return [i for i, _ in flat]


def _index_of(doc, w) -> int:
    """Index of the word dict ``w`` (the same object) in doc['words']."""
    return next(i for i, x in enumerate(doc.get("words") or []) if x is w)


def _timed(w) -> bool:
    try:
        float(w["start_time"]), float(w["end_time"])
        return True
    except (KeyError, TypeError, ValueError):
        return False


def plan_document(doc: dict, audio, cfg: ScoringConfig, units=("segment",)) -> List[dict]:
    """Per-sentence plan: text_train, numbers and (audio, text) jobs for the requested units."""
    plans, sr = [], cfg.sample_rate
    langs = doc_languages(doc, cfg)
    for k, s in enumerate(_sentences(doc)):
        text_train = join_currency(_ml.normalize(s.get("text") or "", mode="train"))
        spans = _find(text_train)
        plan = {"idx": k, "sentence": s, "text_train": text_train, "numbers": [], "clip": None}
        plans.append(plan)
        if not spans:
            continue
        eng, lang_issue = langs[k]
        plan["latin"] = eng                               # English sentence: Latin-script readings
        default_text = _verbalize(text_train, style=cfg.style, latin=eng)
        seg_audio = None
        if s.get("start_time") is not None and s.get("end_time") is not None:
            seg_audio = slice_audio(audio, float(s["start_time"]), float(s["end_time"]), sr, cfg.pad_s).copy()
        else:
            seg_audio = audio
        plan["clip"] = seg_audio                          # sentence clip (transcribe, debugging)
        clip_t0 = (max(0, int((float(s["start_time"]) - cfg.pad_s) * sr)) / sr
                   if s.get("start_time") is not None else 0.0)
        # context words come from THIS sentence only (its clip is what is scored), timed ones only
        swords = [w for w in _words_in(doc, s) if _timed(w)] if ("word" in units or "window" in units) else []
        wmap = _word_map(swords, spans) if swords else None
        for j, sp in enumerate(spans):
            cands = _candidates(sp, style=cfg.style, latin=eng, next_word=_next_word(text_train, sp),
                                alone=_alone(text_train))[:cfg.max_candidates]
            n = Number(k, sp, cands, default_text)
            n.text_issue = currency_issue(sp) or context_issue(text_train, sp) or lang_issue
            if "segment" in units:
                n.jobs["segment"] = [(seg_audio, _sub(text_train, spans, j, c.text, cfg.style, eng)) for c in cands]
                n.unit_used["segment"] = "segment"
            for wu in ("word", "window"):
                if wu in units and wmap is None:
                    n.jobs[wu] = n.jobs.get("segment") or \
                        [(seg_audio, _sub(text_train, spans, j, c.text, cfg.style, eng)) for c in cands]
                    n.unit_used[wu] = "segment(fallback)"
            if wmap is not None:
                wi = wmap[j]
                n.word_idx = [_index_of(doc, swords[wi])]
                lo, hi = max(0, wi - cfg.context_words), min(len(swords) - 1, wi + cfg.context_words)
                left = " ".join(_ml.normalize(str(swords[i].get("word", "")), mode="train")
                                for i in range(lo, wi))
                right = " ".join(_ml.normalize(str(swords[i].get("word", "")), mode="train")
                                 for i in range(wi + 1, hi + 1))
                right = _verbalize(right, style=cfg.style, latin=eng)
                left = _verbalize(left, style=cfg.style, latin=eng)
                w0, w1 = float(swords[lo]["start_time"]), float(swords[hi]["end_time"])
                own = _ml.normalize(str(swords[wi].get("word", "")), mode="train")
                texts = []
                for c in cands:
                    mid = _verbalize(own, style=cfg.style, latin=eng, chooser=lambda span, cs, t=c.text:
                                     t if span.core == sp.core else 0)
                    texts.append(" ".join(x for x in (left, mid, right) if x))
                if "word" in units:
                    win = slice_audio(audio, w0, w1, sr, cfg.pad_s).copy()
                    n.jobs["word"] = [(win, t) for t in texts]
                    n.unit_used["word"] = "word"
                if "window" in units:
                    # frames of the sentence encoding, as seconds from the clip start
                    rel = (max(0.0, w0 - cfg.pad_s - clip_t0), max(0.0, w1 + cfg.pad_s - clip_t0))
                    n.jobs["window"] = [(seg_audio, t, rel) for t in texts]
                    n.unit_used["window"] = "window"
            plan["numbers"].append(n)
    return plans


# --------------------------------------------------------------------------- scoring

def _run(scorer, plans, unit, heads):
    """Score every job for ``unit`` with all ``heads`` -> fills number.scores[(unit, head)]."""
    pairs, owners = [], []
    for p in plans:
        for n in p["numbers"]:
            for ci, job in enumerate(n.jobs[unit]):
                pairs.append(job)
                owners.append((n, ci))
    if not pairs:
        return
    if hasattr(scorer, "score_heads"):
        res = scorer.score_heads(pairs, heads)
    else:
        res = {h: scorer.score(pairs, h) for h in heads}
    for h, vals in res.items():
        for (n, ci), v in zip(owners, vals):
            n.scores.setdefault((unit, h), [None] * len(n.cands))[ci] = v


_STYLE_FORMS = {"contracted": "plain", "glide": "plain", "spaced": "plain",
                "drop_thousand_spaced": "drop_thousand",
                # colloquial variants (digits.ml.ALT_SUBS): same reading as their base form
                "contracted_alt": "plain", "glide_alt": "plain", "spaced_alt": "plain",
                "drop_thousand_alt": "drop_thousand", "drop_thousand_spaced_alt": "drop_thousand",
                "drop_thousand_e": "drop_thousand",           # ഇരുപത്തിനാലേ അഞ്ഞൂറ്
                "lakh_e_glide": "lakh_e", "lakh_e_contracted": "lakh_e", "lakh_e_spaced": "lakh_e",
                "english_lakhs": "english", "latin_lakhs": "latin_english",   # lakh / lakhs
                "english_no_one": "english", "latin_no_one": "latin_english",  # (one) thousand six hundred
                # തേർട്ടിയത്ത് = തേർട്ടിത്ത് (spelling of the same English ordinal)
                "ordinal_th_ya": "ordinal_th", "ordinal_ya": "ordinal"}


def family(c) -> str:
    """Reading family of a candidate: language + form with the spelling style folded away
    (contracted / glide / spaced -> plain; '+suffix' parts such as +shathamanam are kept)."""
    head, _, rest = c.form.partition("+")
    return f"{c.lang}:{_STYLE_FORMS.get(head, head)}" + (f"+{rest}" if rest else "")


_SPELLINGS = {"contracted", "glide", "spaced", "drop_thousand", "drop_thousand_spaced"}


def _spelling_parts(c):
    """(base form, is _alt, +suffix) — spelling variants of one reading share all but the style."""
    head, _, rest = c.form.partition("+")
    alt = head.endswith("_alt")
    return (head[:-4] if alt else head), alt, rest


def _spelling_group(cands, i):
    """Indices of the spelling variants of candidate i (same reading family, same _alt-ness and
    suffix, a different contracted / glide / spaced spelling)."""
    base, alt, rest = _spelling_parts(cands[i])
    if base not in _SPELLINGS:
        return [i]
    fam = family(cands[i])
    return [j for j, c in enumerate(cands)
            if family(c) == fam and _spelling_parts(c)[1:] == (alt, rest) and _spelling_parts(c)[0] in _SPELLINGS]


def decide(n: Number, unit: str, head: str, cfg: ScoringConfig, scorer=None) -> dict:
    """Pick the reading with the lowest score. Then the spelling: the audio's spelling is kept
    only if it beats the other spellings of the same reading by at least ``style_margin`` nats;
    otherwise the default spelling (``style``, contracted) is used — the audio can hardly tell
    ഇരുപത്തഞ്ച് / ഇരുപത്തിയഞ്ച് / ഇരുപത്തി അഞ്ച് apart. margin / review_margin are about the
    audio's own best candidate (the reading decision).

    The result is cached on the number (decide runs for the text, the TSV rows, the verify copy
    and the report): every caller sees the same decision, computed once. NaN scores count as
    the worst possible (inf), so they never win and never reach the JSON/HTML outputs."""
    costs = form_costs_of(cfg)
    key = (unit, head, tuple(n.scores[(unit, head)]), cfg.margin_threshold, getattr(cfg, "style_margin", 0.0),
           cfg.style, cfg.length_norm, tuple(sorted(costs.items())))
    cache = n.__dict__.setdefault("_decisions", {})
    if key in cache:
        return cache[key]
    nll = [v if v == v else float("inf") for v in n.scores[(unit, head)]]      # NaN -> inf
    metric = list(nll)
    if cfg.length_norm and scorer is not None and hasattr(scorer, "n_tokens"):
        metric = [v / scorer.n_tokens(t) for v, (_, t) in zip(nll, n.jobs[unit])]
    audio = metric                                # the audio alone (suffix_gap measures this)
    if costs:                                     # readings the audio over-picks (form_costs)
        metric = [v + form_cost(n, c, costs) for v, c in zip(metric, n.cands)]
    order = sorted(range(len(metric)), key=lambda i: metric[i])
    best = order[0]
    margin = (metric[order[1]] - metric[best]) if len(order) > 1 else float("inf")
    fam = family(n.cands[best])
    other = [i for i in order[1:] if family(n.cands[i]) != fam]
    review_margin = (metric[other[0]] - metric[best]) if other else float("inf")
    chosen, style_gap, fallback = best, None, False
    group = _spelling_group(n.cands, best)
    if len(group) > 1:
        style_gap = min(metric[j] for j in group if j != best) - metric[best]
        if style_gap < getattr(cfg, "style_margin", 0.0):
            base = _spelling_parts(n.cands[best])[0]
            want = "drop_thousand" if base.startswith("drop_thousand") else cfg.style
            dflt = next((j for j in group if _spelling_parts(n.cands[j])[0] == want), None)
            if dflt is not None and dflt != best:
                chosen, fallback = dflt, True
    issue = _suffix_issue(n, n.cands[best])
    text_issue = getattr(n, "text_issue", None)
    sg = suffix_gap(n, audio)
    cache[key] = out = {"best": chosen, "model_best": best, "margin": margin, "review_margin": review_margin,
            "style_gap": style_gap, "style_fallback": fallback, "suffix_issue": issue, "text_issue": text_issue,
            "suffix_gap": sg[0] if sg else None, "suffix_bare": sg[1] if sg else None,
            "suffix_with": sg[2] if sg else None,
            "needs_review": not (review_margin >= cfg.margin_threshold) or issue is not None or text_issue is not None,
            "nll": nll, "metric": metric}
    return out


def form_costs_of(cfg) -> dict:
    from .config import DEFAULT_FORM_COSTS
    c = getattr(cfg, "form_costs", None)
    return dict(DEFAULT_FORM_COSTS) if c is None else {k: float(v) for k, v in c.items() if v}


def form_cost(n, cand, costs) -> float:
    """Extra nats for a reading the audio score over-picks (ScoringConfig.form_costs):
        nosuffix     the written suffix left out (27-ന് -> ഇരുപത്തേഴ്)
        short_split  digit by digit for a number of up to 3 digits (22 -> രണ്ട് രണ്ട്); codes
                     with a leading zero (07) and longer numbers are not affected
        percent      പെർസെന്റ് (55% -> ... പെർസെന്റ്; പെർസെന്റേജ് / ശതമാനം are not)
        bare_hour    a whole hour without മണി / ഓ ക്ലോക്ക് (11:00 -> പതിനൊന്ന്)"""
    f, cost = cand.form or "", 0.0
    if "+" + ml.NOSUFFIX in f:
        cost += costs.get("nosuffix", 0.0)
    if f.startswith(("digit_by_digit", "doubled")):
        d = re.match(r"\d+", to_ascii_digits(getattr(getattr(n, "span", None), "core", "") or ""))
        if d and len(d.group()) <= 3 and not (len(d.group()) > 1 and d.group()[0] == "0"):
            cost += costs.get("short_split", 0.0)
    if f.endswith("+percent"):
        cost += costs.get("percent", 0.0)
    if f == "bare":
        cost += costs.get("bare_hour", 0.0)
    return cost


def suffix_gap(n, metric):
    """num_suffix only: (best WITH-suffix score minus best BARE score, bare text, suffixed text).
    Positive = the audio prefers the bare number by that many nats; negative = the suffix wins.
    None when the number has no bare/suffixed pair. Measured to choose a suffix margin."""
    if getattr(getattr(n, "span", None), "cls", None) != "num_suffix":
        return None
    tag = "+" + ml.NOSUFFIX
    bare = [i for i, c in enumerate(n.cands) if c.form.endswith(tag)]
    withs = [i for i, c in enumerate(n.cands) if not c.form.endswith(tag)]
    if not bare or not withs:
        return None
    b = min(bare, key=lambda i: metric[i])
    w = min(withs, key=lambda i: metric[i])
    return metric[w] - metric[b], n.cands[b].text, n.cands[w].text


def _suffix_issue(n, best_cand) -> Optional[str]:
    """num_suffix only: the audio prefers the bare number (the written suffix was not said, or a
    word came in between: 55000-യുടെ spoken അമ്പത്തയ്യായിരം രൂപയുടെ), or the suffix looks
    unusual for the number (ml.suffix_issue). Such numbers always go to verify."""
    span = getattr(n, "span", None)
    if getattr(span, "cls", None) != "num_suffix":
        return None
    if best_cand.form.endswith("+" + ml.NOSUFFIX):
        return "audio prefers the number without the suffix"
    return ml.suffix_issue(span.core)


# --------------------------------------------------------------------------- verbalize a dir

def _round(x):
    """Rounded for JSON/TSV; None for None / inf / NaN (no invalid JSON tokens)."""
    if x is None or x != x or x in (float("inf"), float("-inf")):
        return None
    return round(x, 4)


# --------------------------------------------------------------------------- grouping
# Files are processed in GROUPS (``group_audio_s`` seconds of recordings each): every clip and
# candidate of every file in the group goes to the scorer together, so GPU batches are full
# across recording boundaries and sorted by length over the whole group (less padding). Only
# the clips are kept (copies), never whole recordings, so memory stays bounded. The next group
# is loaded + planned on a background thread while the GPU scores the current one.

def _load_one(src, audio_dir, cfg, units):
    stem = os.path.splitext(os.path.basename(src))[0]
    stage = "read_json"
    try:
        doc = json.load(open(src, encoding="utf-8"))
        stage = "find_audio"
        wav = find_audio(audio_dir, stem)
        stage = "read_audio"
        audio = load_wav(wav, cfg.sample_rate)
        stage = "plan"
        plans = plan_document(doc, audio, cfg, units=units)
    except Exception as e:                      # one bad file never stops the run
        return {"src": src, "stem": stem, "stage": stage, "error": f"{type(e).__name__}: {e}"}
    return {"src": src, "stem": stem, "doc": doc, "plans": plans,
            "audio_s": len(audio) / cfg.sample_rate}


def _load_group(srcs, audio_dir, cfg, units):
    import time
    t0 = time.perf_counter()
    items = [_load_one(s, audio_dir, cfg, units) for s in srcs]
    return items, time.perf_counter() - t0


def _split_groups(srcs, audio_dir, cfg):
    """Groups of source files totalling about ``group_audio_s`` seconds of audio (file sizes are
    used as a proxy before loading; 1 s of 16-bit mono 16 kHz WAV = 32 kB)."""
    groups, cur, acc = [], [], 0.0
    for s in srcs:
        cur.append(s)
        acc += _approx_seconds(s, audio_dir, cfg)
        if acc >= cfg.group_audio_s:
            groups.append(cur)
            cur, acc = [], 0.0
    if cur:
        groups.append(cur)
    return groups


def _approx_seconds(src, audio_dir, cfg):
    try:                                     # same lookup as loading (.wav / .WAV)
        p = find_audio(audio_dir, os.path.splitext(os.path.basename(src))[0])
    except (FileNotFoundError, OSError):
        return 0.0
    return os.path.getsize(p) / (2.0 * cfg.sample_rate)


def iter_groups(srcs, audio_dir, cfg, units):
    """Yield (items, load_seconds) per group; the next group loads in the background."""
    from concurrent.futures import ThreadPoolExecutor
    groups = _split_groups(srcs, audio_dir, cfg) if cfg.group_audio_s else [[s] for s in srcs]
    if not groups:
        return
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_load_group, groups[0], audio_dir, cfg, units)
        for gi in range(len(groups)):
            items, load_s = fut.result()
            if gi + 1 < len(groups):
                fut = ex.submit(_load_group, groups[gi + 1], audio_dir, cfg, units)
            yield items, load_s


def verbalize_document(path: str, audio_dir: str, scorer, cfg: ScoringConfig) -> dict:
    item = _load_one(path, audio_dir, cfg, (cfg.unit,))
    if "error" in item:
        raise FileNotFoundError(item["error"])
    _run(scorer, item["plans"], cfg.unit, (cfg.head,))
    return same_document(item["doc"], item["plans"], cfg, scorer)[0]


STATUSES = ("verbalized", "verify", "failed")


def run_dir(data_dir: str, cfg: ScoringConfig) -> str:
    """OUT_DIR/<data dir name>: verbalized/ verify/ failed/ <stem>.json + summary / details / report."""
    if not cfg.out_dir:
        raise ValueError("set out_dir in the config (or --out-dir)")
    rd = os.path.join(cfg.out_dir, os.path.basename(os.path.normpath(data_dir)))
    ins = os.path.abspath(data_dir)
    for st in STATUSES:
        if os.path.abspath(os.path.join(rd, st)) == ins:
            raise ValueError(f"out_dir would write into the input dir {data_dir}")
    return rd


def output_path(src: str, cfg: ScoringConfig, data_dir: Optional[str] = None, status: str = "verbalized") -> str:
    """OUT_DIR/<data dir>/<status>/<stem>.json"""
    return os.path.join(run_dir(data_dir or os.path.dirname(src), cfg), status, os.path.basename(src))


def _split_times(t0, t1, parts):
    """Split [t0, t1] across ``parts`` (strings) in proportion to their length."""
    if len(parts) == 1 or t0 is None or t1 is None:
        return [(t0, t1)] * len(parts)
    t0, t1 = float(t0), float(t1)
    total = sum(max(1, len(p)) for p in parts)
    out, a = [], t0
    for i, p in enumerate(parts):
        b = t1 if i == len(parts) - 1 else a + (t1 - t0) * max(1, len(p)) / total
        out.append((round(a, 3), round(b, 3)))
        a = b
    return out


def _word_owners(words, sents):
    """id(word) -> index of the sentence it belongs to: the one its time span overlaps most
    (ties -> nearest sentence middle). Boundary words shared by two sentences go to one only."""
    spans = []
    for k, s in enumerate(sents):
        try:
            spans.append((k, float(s["start_time"]), float(s["end_time"])))
        except (KeyError, TypeError, ValueError):
            pass
    owner = {}
    for w in words:
        try:
            a, b = float(w["start_time"]), float(w["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        mid, best = (a + b) / 2, None
        for k, s0, s1 in spans:
            ov = min(b, s1) - max(a, s0)
            if ov < -1e-3:
                continue
            key = (ov, -abs(mid - (s0 + s1) / 2))
            if best is None or key > best[0]:
                best = (key, k)
        if best is not None:
            owner[id(w)] = best[1]
    return owner


def _word_owners_by_text(words, sents, text_train):
    """Walk words and sentences in order; a sentence owns the run of words whose train-normalized
    tokens spell exactly its own train text. Sentences that don't line up are left to the
    time-based owners (a mismatch resynchronises at the next sentence's first word, by time)."""
    norm = [(_ml.normalize(str(w.get("word", "")), mode="train").split(), w) for w in words]
    owner, i = {}, 0
    for k, s in enumerate(sents):
        want = (text_train.get(k) or "").split()
        if not want:
            continue
        j, got, run = i, [], []
        while j < len(norm) and len(got) < len(want):
            toks, w = norm[j]
            got += toks
            run.append(w)
            j += 1
        if got == want:
            for w in run:
                owner[id(w)] = k
            i = j
        else:                                    # resync: first word starting at/after this sentence's end
            try:
                end = float(s["end_time"])
                while i < len(norm) and float(norm[i][1].get("start_time") or 0) < end - 1e-3:
                    i += 1
            except (KeyError, TypeError, ValueError):
                pass
    return owner


def plan_text(doc: dict, cfg: ScoringConfig) -> List[dict]:
    """Like plan_document but without audio (no scoring jobs): text_train, numbers and their
    candidates per sentence. Used by the verifier to rebuild a file from chosen readings."""
    plans = []
    langs = doc_languages(doc, cfg)
    for k, s in enumerate(_sentences(doc)):
        text_train = join_currency(_ml.normalize(s.get("text") or "", mode="train"))
        spans = _find(text_train)
        eng, lang_issue = langs[k]
        plan = {"idx": k, "sentence": s, "text_train": text_train, "numbers": [], "clip": None, "latin": eng}
        default_text = _verbalize(text_train, style=cfg.style, latin=eng) if spans else text_train
        for sp in spans:
            n = Number(k, sp, _candidates(sp, style=cfg.style, latin=eng, alone=_alone(text_train),
                                          next_word=_next_word(text_train, sp))[:cfg.max_candidates], default_text)
            n.text_issue = currency_issue(sp) or context_issue(text_train, sp) or lang_issue
            plan["numbers"].append(n)
        plans.append(plan)
    return plans


def same_document(doc: dict, plans, cfg: ScoringConfig, scorer=None, choices=None):
    """Output in the INPUT's own schema: every sentence ``text`` -> its verbalized train text,
    ``words`` -> train-normalized words with numbers replaced by the chosen reading (a number that
    becomes several words gets its word's time span split across them), ``transcription`` -> the
    sentences joined. Other keys are kept as they are. Returns (doc, stats, review_rows).
    ``choices``: {sentence index: [reading per number, in order]} chosen by a person (verifier);
    numbers not covered are decided by the scores as usual."""
    out = json.loads(json.dumps(doc))
    sents = _sentences(out)
    words = out.get("words") if isinstance(out.get("words"), list) else None
    stats = {"numbers": 0, "needs_review": 0, "word_fallbacks": 0, "word_text_mismatch": 0}
    review, owner = [], {}
    picks_by_sent = {}
    for p in plans:
        s = sents[p["idx"]]
        picks = []
        chosen = (choices or {}).get(p["idx"])
        if chosen is not None:
            if len(chosen) != len(p["numbers"]):
                raise ValueError(f"sentence {p['idx']}: {len(chosen)} readings for {len(p['numbers'])} numbers")
            for n, t in zip(p["numbers"], chosen):
                if not str(t).strip():
                    raise ValueError(f"sentence {p['idx']}: empty reading for {n.span.core}")
                picks.append((n.span.core, " ".join(str(t).split())))
                stats["numbers"] += 1
            p_numbers = []
        else:
            p_numbers = p["numbers"]
        for n in p_numbers:
            d = decide(n, cfg.unit, cfg.head, cfg, scorer)
            c = n.cands[d["best"]]
            picks.append((n.span.core, c.text))
            stats["numbers"] += 1
            if d["needs_review"]:
                stats["needs_review"] += 1
                review.append({"sentence": p["idx"], "start_time": s.get("start_time"), "end_time": s.get("end_time"),
                               "core": n.span.core, "class": n.span.cls, "chosen": c.text, "form": c.form,
                               "margin": _round(d["margin"]), "review_margin": _round(d["review_margin"]),
                               "candidates": " | ".join(x.text for x in n.cands),
                               "sentence_text": (s.get("text") or "").replace("\t", " ")})
        it = iter(t for _, t in picks)
        s["text"] = (_verbalize(p["text_train"], style=cfg.style, chooser=lambda span, cs: next(it))
                     if picks else p["text_train"])
        picks_by_sent[p["idx"]] = picks
    out["transcription"] = " ".join(x.get("text", "") for x in sents if x.get("text"))
    if words is not None:
        owner = _word_owners(words, sents)                       # by time ...
        owner.update(_word_owners_by_text(words, sents, {p["idx"]: p["text_train"] for p in plans}))  # ... text wins
        new_words, queues = [], {k: list(v) for k, v in picks_by_sent.items()}
        latin_of = {p["idx"]: bool(p.get("latin")) for p in plans}
        sent_text = {p["idx"]: p["text_train"] for p in plans}
        sent_words = {}
        for w in words:
            k = owner.get(id(w))
            t = _ml.normalize(str(w.get("word", "")), mode="train")
            if not t:
                continue
            if _SIGN_WORD.fullmatch(t) and k in sent_text and \
                    not re.search(rf"(?i)(?<!\S){re.escape(t)}(?!\S)", sent_text[k]):
                continue                              # a sign glued to its amount (₹ 500): read with it
            q = queues.get(k, [])

            def choose(span, cands, q=q):
                if q and bare_core(q[0][0]) == bare_core(span.core):
                    return q.pop(0)[1]
                stats["word_fallbacks"] += 1          # word's number doesn't line up: default reading
                return None
            vt = _verbalize(t, style=cfg.style, chooser=choose, latin=latin_of.get(k, False))
            parts = vt.split()
            for part, (a, b) in zip(parts, _split_times(w.get("start_time"), w.get("end_time"), parts)):
                nw = dict(w, word=part)
                if a is not None:
                    nw["start_time"], nw["end_time"] = a, b
                new_words.append(nw)
                sent_words.setdefault(k, []).append(part)
        out["words"] = new_words
        for k, x in enumerate(sents):
            if k in sent_words and " ".join(sent_words[k]) != " ".join((x.get("text") or "").split()):
                stats["word_text_mismatch"] += 1
    return out, stats, review


def _number_details(p, n, d, cfg):
    c = n.cands[d["best"]]
    s = p["sentence"]
    return {"sentence": p["idx"], "start_time": s.get("start_time"), "end_time": s.get("end_time"),
            "core": n.span.core, "class": n.span.cls, "chosen": c.text, "lang": c.lang, "form": c.form,
            "margin": _round(d["margin"]), "review_margin": _round(d["review_margin"]),
            "needs_review": bool(d["needs_review"]),
            "model_pick": n.cands[d.get("model_best", d["best"])].text,
            "style_gap": _round(d.get("style_gap")), "style_fallback": bool(d.get("style_fallback")),
            "suffix_issue": d.get("suffix_issue") or "",
            "text_issue": d.get("text_issue") or "",
            "suffix_gap": _round(d.get("suffix_gap")),
            "candidates": [{"text": x.text, "lang": x.lang, "form": x.form, f"nll_{cfg.head}": _round(v)}
                           for x, v in zip(n.cands, d["nll"])]}


def folder_counts(rd: str) -> dict:
    """Files now in each output folder of a run dir (the run's real state)."""
    out = {}
    for st in STATUSES + ("reverify",):
        d = os.path.join(rd, st)
        out[st] = sum(1 for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else 0
    return out


def done_stems(rd: str, include_failed: bool = True) -> set:
    """Stems that already have a result in a run dir (any output folder or TSV); with
    ``include_failed=False`` a file whose only result is failed/ does not count as done."""
    from .details import read_tsv, tsv_path
    out = set()
    for st in STATUSES + ("reverify",):
        if st == "failed" and not include_failed:
            continue
        d = os.path.join(rd, st)
        if os.path.isdir(d):
            out |= {f[:-5] for f in os.listdir(d) if f.endswith(".json")}
        out |= {r.get("file") for r in read_tsv(tsv_path(rd, st))}
    return out


def verbalize_dir(data_dir: str, audio_dir: str, scorer, cfg: ScoringConfig, limit: Optional[int] = None,
                  log=print, resume: bool = False, retry_failed: bool = False) -> dict:
    """Write, under OUT_DIR/<data dir name>/:
        verbalized/<stem>.json  files whose numbers were all decided confidently: the input's own
                                schema with sentence / word / transcription texts verbalized
        verify/<stem>.json      when any number needs review: the file verbalized EXCEPT those numbers,
                                which stay as written (digits) -> decide them with the verifier
        failed/<stem>.json      the INPUT file, unchanged, when it could not be processed
        input/<stem>.json       a copy of the input of each verify file (lets the verifier change
                                already-decided numbers on any machine)
        verbalized.tsv / verify.tsv   one row per number: candidates, scores, model pick, margins,
                                status (auto / pending / verified), final pick (see details.py)
        failed.tsv              one row per failed file: stage, error, status
        summary.json            totals, timing, config;   report.html  interactive report

    ``retry_failed``: like ``resume``, but files in failed/ are processed again.
    ``resume``: skip input files that already have a result in the run dir (an interrupted run, or
    new files added to the data dir); ``limit`` then counts the remaining ones. summary.json always
    counts what is in the folders; the report of a resumed run covers the files it processed
    (report_resume_<time>.html; report.html of the first run is kept)."""
    rd = run_dir(data_dir, cfg)
    srcs = list_sources(data_dir)
    skipped = 0
    resume = resume or retry_failed
    if resume:
        have = done_stems(rd, include_failed=not retry_failed)
        keep = [s for s in srcs if os.path.splitext(os.path.basename(s))[0] not in have]
        skipped = len(srcs) - len(keep)
        srcs = keep
        log(f"resume: {skipped} file(s) already done, {len(srcs)} to go"
            + (" (failed files are retried)" if retry_failed else ""))
    srcs = srcs[:limit]
    for st in STATUSES:
        os.makedirs(os.path.join(rd, st), exist_ok=True)
    from .details import read_tsv as _read_tsv
    hand_edited = {r.get("file") for r in _read_tsv(os.path.join(rd, "sentence_edits.tsv"))}
    import shutil
    import time
    tot = {"files": len(srcs), "verbalized": 0, "verify": 0, "failed": 0, "numbers": 0, "needs_review": 0,
           "word_fallbacks": 0, "word_text_mismatch": 0}
    processed = []
    t_start, audio_s = time.perf_counter(), 0.0
    if hasattr(scorer, "stats"):
        scorer.stats.clear()
    done, timing = 0, {"load_s": 0.0, "score_s": 0.0}
    report = []
    for items, load_s in iter_groups(srcs, audio_dir, cfg, (cfg.unit,)):
        timing["load_s"] += load_s
        ok = [it for it in items if "error" not in it]
        t0 = time.perf_counter()
        try:
            _run(scorer, [p for it in ok for p in it["plans"]], cfg.unit, (cfg.head,))
        except Exception as e:                   # e.g. CUDA OOM: this group fails, the run goes on
            for it in ok:
                it.update(stage="score", error=f"{type(e).__name__}: {e}")
            log(f"scoring failed for a group of {len(ok)} file(s): {type(e).__name__}: {e}")
            _free_gpu()
        timing["score_s"] += time.perf_counter() - t0
        group_rows = {st: [] for st in STATUSES}
        group_files = []
        for it in items:
            done += 1
            src, stem = it["src"], it["stem"]
            name = os.path.basename(src)
            for st in STATUSES + ("input", "reverify"):   # a re-run replaces this file's earlier result
                old = os.path.join(rd, st, name)
                if os.path.exists(old):
                    os.remove(old)
            processed.append(stem)
            group_files.append(stem)
            status, rec = "failed", {"file": stem, "source": os.path.abspath(src)}
            rows = []
            if "error" in it:
                rec.update(stage=it.get("stage"), error=it["error"])
            else:
                try:
                    out, stt, _ = same_document(it["doc"], it["plans"], cfg, scorer)
                    nums = [_number_details(p, n, decide(n, cfg.unit, cfg.head, cfg, scorer), cfg)
                            for p in it["plans"] for n in p["numbers"]]
                    sent_verb = {k: x.get("text") for k, x in enumerate(_sentences(out))}
                    for x in nums:
                        x["text_verbalized"] = sent_verb.get(x["sentence"])
                    reasons = []
                    if stt["needs_review"]:
                        reasons.append(f"{stt['needs_review']} number(s) need review")
                    n_suf = sum(1 for x in nums if x.get("suffix_issue"))
                    n_txt = sum(1 for x in nums if x.get("text_issue"))
                    if n_txt:
                        reasons.append(f"{n_txt} text issue(s)")
                    if n_suf:
                        reasons.append(f"{n_suf} suffix issue(s)")
                    status = "verify" if stt["needs_review"] else "verbalized"
                    if status == "verify":
                        # the verify copy: confident numbers verbalized, uncertain ones left as written
                        keep = {}
                        for p in it["plans"]:
                            if p["numbers"]:
                                keep[p["idx"]] = [
                                    (n.span.text if d["needs_review"] else n.cands[d["best"]].text)
                                    for n, d in ((n, decide(n, cfg.unit, cfg.head, cfg, scorer)) for n in p["numbers"])]
                        out, _, _ = same_document(it["doc"], it["plans"], cfg, choices=keep)
                    rec.update(audio_s=round(it["audio_s"], 2), n_numbers=stt["numbers"],
                               n_needs_review=stt["needs_review"], word_fallbacks=stt["word_fallbacks"],
                               word_text_mismatch=stt["word_text_mismatch"])
                    if reasons:
                        rec["reasons"] = reasons
                    rows = number_rows(stem, it["plans"], nums, status, reasons)
                    audio_s += it["audio_s"]
                    tot["numbers"] += stt["numbers"]
                    tot["needs_review"] += stt["needs_review"]
                    tot["word_fallbacks"] += stt["word_fallbacks"]
                    tot["word_text_mismatch"] += stt["word_text_mismatch"]
                    file_rows = report_rows(stem, it["plans"], cfg, scorer)
                    for r in file_rows:
                        r["file_status"] = status
                    report.extend(file_rows)
                    if status in ("verbalized", "verify"):
                        with open(os.path.join(rd, status, name), "w", encoding="utf-8") as f:
                            json.dump(out, f, ensure_ascii=False, indent=2)
                    if status == "verify":           # the input too, so the verifier can re-edit
                        os.makedirs(os.path.join(rd, "input"), exist_ok=True)   # decided numbers anywhere
                        shutil.copyfile(src, os.path.join(rd, "input", name))
                except Exception as e:           # pragma: no cover - unexpected per-file problem
                    status = "failed"
                    rec = {"file": stem, "source": os.path.abspath(src), "stage": "verbalize",
                           "error": f"{type(e).__name__}: {e}"}
            if status == "failed":
                shutil.copyfile(src, os.path.join(rd, status, name))      # the input, unchanged
            tot[status] += 1
            if status == "failed":
                rows = [{"file": stem, "source": rec["source"], "stage": rec.get("stage"), "error": rec.get("error"),
                         "status": "pending"}]
            group_rows[status].extend(rows)
            msg = rec.get("error") or "; ".join(rec.get("reasons", [])) or f"{rec.get('n_numbers', 0)} numbers"
            if stem in hand_edited:
                msg += " (re-run: its hand-edited sentences in sentence_edits.tsv are not applied)"
            log(f"[{done}/{len(srcs)}] {status:10s} {name}: {msg}")
        merge_details(rd, group_rows, group_files)   # TSVs match the JSON folders after every group
    wall = time.perf_counter() - t_start
    tot["stages_s"] = {k: round(v, 2) for k, v in timing.items()}
    tot["seconds"] = round(wall, 2)
    tot["audio_hours"] = round(audio_s / 3600, 3)
    tot["rtf"] = round(wall / audio_s, 4) if audio_s else None
    if getattr(scorer, "stats", None):
        tot["scorer"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in scorer.stats.items()}
    tot["data_dir"], tot["audio_dir"], tot["out_dir"] = os.path.abspath(data_dir), os.path.abspath(audio_dir), rd
    tot["files_needing_review"] = tot["verify"]
    from .verbalize_report import write_report
    html = "report.html"
    if resume and skipped:
        html = f"report_resume_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    write_report(rd, report, tot, cfg, os.path.basename(os.path.normpath(data_dir)), html_name=html)
    # summary.json counts what is in the folders now (earlier runs, resumes, the verifier's moves)
    this_run = {k: tot[k] for k in ("files", "verbalized", "verify", "failed")}
    counts = folder_counts(rd)
    sp = os.path.join(rd, "summary.json")
    with open(sp, encoding="utf-8") as f:
        summ = json.load(f)
    summ.update(counts, files=sum(counts.values()), this_run=this_run, resumed_skipped=skipped, report=html)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    tot.update(this_run=this_run, totals=counts, resumed_skipped=skipped, report=html)
    return tot


def _free_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                            # pragma: no cover - no torch / no GPU
        pass


def _file_seconds(doc) -> float:
    """Recording length from its last timestamp (audio length is not kept after scoring)."""
    ends = [float(x.get("end_time") or 0) for x in (doc.get("words") or []) + _sentences(doc)]
    return max(ends) if ends else 0.0
