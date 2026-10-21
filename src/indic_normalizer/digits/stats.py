"""Distribution analysis of digits and special characters (SC) in an ASR transcript dir.

Input: a directory like ``eval_frozen`` — one JSON per recording with
    {"transcription": str,
     "words":     [{"word", "start_time", "end_time", "speaker"}, ...],
     "sentences": [{"text", "start_time", "end_time", "speaker"}, ...]}
Audio length per file is read from a same-stem audio file next to the JSON when one exists
(.wav/.flac/.mp3/.m4a/.ogg/.opus, via ffprobe or the stdlib ``wave``); otherwise it is the
last word/sentence end time (leading/trailing silence not counted — the source is recorded
per file in files.tsv and in metrics).

Output: ``<out-root>/<input dir name>/`` (default out-root: ./outputs)
    metrics.json            every metric, machine-readable
    metrics.html            the same, human-readable report (sortable tables, contents)
    files.tsv               one row per file: duration, counts, durations with digits/SC
    sentences.tsv           one row per sentence: timing, flags, numeric classes, text
    numeric_expressions.tsv every number-bearing word in the set, with its class
    sc_chars.tsv            per-character SC frequency (codepoint, name, class, files)
    digits/<file>.tsv       per file: location of every number-bearing word
    sc/<file>.tsv           per file: location of every special character

Definitions
    digit  any Unicode decimal digit (ASCII 0-9, Malayalam ൦-൯, ...).
    SC     any character that is not a letter, combining mark, digit or whitespace:
           punctuation (P*), symbols (S*: % ₹ × + ...), invisible format chars (Cf: ZWJ,
           ZWNJ, BOM), control/private-use/unassigned. Reported in groups:
             punct    sentence punctuation . , ? ! ; : ' " ( ) and dashes
             symbol   S* categories
             format   Cf (zero-width etc.)
             other    everything else
           and separately as "in a numeric expression" (4:00, 2.5, 70%, 8-9) vs elsewhere.
    numeric expression  a whitespace word containing a digit, classified as date, time,
           time_range, range, decimal, percent, currency, ordinal, phone, id_code, year,
           grouped (1,75,000), slash (6/14), alnum (5G, 10K, ഡി3), num_suffix (23-ന്,
           250യിൽ), integer; anything else is other_numeric.

Usage
    indic-digits stats PATH/TO/eval_frozen [--out-root outputs] [--tokenizer tokenizer.model]
    python -m indic_normalizer.digits stats PATH/TO/eval_frozen

    from indic_normalizer.digits.stats import analyse
    out_dir, metrics = analyse("eval_frozen", "outputs")

Standard library only (``sentencepiece`` is imported only when --tokenizer is given; ffprobe
is used only when a non-WAV audio file is found).

Standard library only (``sentencepiece`` is imported only when --tokenizer is given; ffprobe
is used only when a non-WAV audio file is found).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import unicodedata
import wave
from collections import Counter, defaultdict

AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm")

# ----------------------------------------------------------------------------- characters

def is_digit(ch: str) -> bool:
    return unicodedata.category(ch) == "Nd"


def sc_group(ch: str) -> "str | None":
    """SC group of a character, or None if it is a letter/mark/digit/space."""
    if ch.isspace():
        return None
    cat = unicodedata.category(ch)
    if cat[0] in "LMN":                       # letters, combining marks, numbers
        return None
    if cat[0] == "P":
        return "punct"
    if cat[0] == "S":
        return "symbol"
    if cat == "Cf":
        return "format"
    return "other"


def script_of(ch: str) -> "str | None":
    if not unicodedata.category(ch).startswith(("L", "M")):
        return None
    o = ord(ch)
    if 0x0D00 <= o <= 0x0D7F:
        return "malayalam"
    if 0x0B80 <= o <= 0x0BFF:
        return "tamil"
    if 0x0900 <= o <= 0x097F:
        return "devanagari"
    if ch.isascii():
        return "latin"
    try:
        return unicodedata.name(ch).split()[0].lower()
    except ValueError:
        return "unknown"


# ------------------------------------------------------------------- numeric classification

# One classifier for the whole package: the same classes verbalize uses (detect.py).
from .detect import classify as _classify, split_edges as _split_edges  # noqa: E402

_EDGE_PUNCT = re.compile(r"^[\"'“”‘’(\[{«]+|[\"'“”‘’)\]}».,?!;:…—–\-]+$")


def strip_edges(tok: str) -> str:
    """Drop sentence punctuation around a word (keep % / internal punctuation)."""
    t = re.sub(r"^[\"'“”‘’(\[{«]+", "", tok)
    t = re.sub(r"[\"'“”‘’)\]}».,?!;:…—–\-]+$", "", t)
    return t


def to_ascii_digits(s: str) -> str:
    return "".join(str(unicodedata.digit(c)) if is_digit(c) else c for c in s)


def classify_numeric(tok: str) -> str:
    """Class of a number token exactly as verbalize sees it (detect.classify)."""
    c = _classify(_split_edges(tok)[1])
    return "other_numeric" if c == "other" else c


def digit_len_bucket(tok: str) -> str:
    runs = re.findall(r"\d+", to_ascii_digits(tok.replace(",", "")))
    n = max((len(r) for r in runs), default=0)
    if n <= 4:
        return str(n)
    if n <= 7:
        return "5-7"
    return "8+"


# --------------------------------------------------------------------------- audio length

def audio_duration(path: str) -> "float | None":
    try:
        if path.lower().endswith(".wav"):
            with wave.open(path) as w:
                return w.getnframes() / float(w.getframerate())
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return None


def find_audio(json_path: str) -> "str | None":
    stem = os.path.splitext(json_path)[0]
    for ext in AUDIO_EXT:
        for cand in (stem + ext, stem + ext.upper()):
            if os.path.exists(cand):
                return cand
    return None


# ------------------------------------------------------------------------------ helpers

def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "sum": round(sum(xs), 3), "min": round(xs[0], 3),
            "mean": round(statistics.mean(xs), 3), "median": round(statistics.median(xs), 3),
            "p90": round(q(0.9), 3), "max": round(xs[-1], 3)}


def pct(a, b):
    return round(100.0 * a / b, 2) if b else 0.0


def hours(sec):
    return round(sec / 3600.0, 3)


def dur(item):
    try:
        return max(0.0, float(item["end_time"]) - float(item["start_time"]))
    except (KeyError, TypeError, ValueError):
        return None


DUR_BUCKETS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 1e9)]


def dur_bucket(d):
    if d is None:
        return "unknown"
    for lo, hi in DUR_BUCKETS:
        if lo <= d < hi:
            return f"{lo}-{hi}s" if hi < 1e9 else f"{lo}s+"
    return "unknown"


def context(text, i, j, w=25):
    return text[max(0, i - w):i] + "⟦" + text[i:j] + "⟧" + text[j:j + w]


def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t", lineterminator="\n")
        wr.writerow(header)
        wr.writerows(rows)


# ------------------------------------------------------------------------------- main

def analyse(in_dir, out_root, tokenizer=None):
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".json"))
    if not files:
        sys.exit(f"no .json files in {in_dir}")
    name = os.path.basename(os.path.normpath(in_dir))
    out = os.path.join(out_root, name)
    for sub in ("digits", "sc"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    sp = None
    if tokenizer:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tokenizer)

    file_rows, sent_rows, num_rows = [], [], []
    sc_char = Counter()
    sc_char_files = defaultdict(set)
    sc_char_numeric = Counter()
    digit_char = Counter()
    num_class = Counter()
    num_class_files = defaultdict(set)
    num_class_dur = defaultdict(float)
    num_len = Counter()
    script_words = Counter()
    script_sent = Counter()
    oov_chars = Counter()
    speaker_sent = Counter()
    speaker_sent_digit = Counter()
    dur_bucket_all, dur_bucket_digit, dur_bucket_sym = Counter(), Counter(), Counter()
    sent_pos = Counter()                       # where in the sentence the number sits

    tot = Counter()                            # simple counters
    tot_dur = Counter()                        # summed seconds
    per_file_digits, per_file_numtok, per_file_sc, per_file_sym = [], [], [], []
    word_durs = defaultdict(list)              # group -> [word durations]
    sent_durs = defaultdict(list)              # group -> [sentence durations]
    file_durs = defaultdict(list)
    dur_source = Counter()

    for fn in files:
        path = os.path.join(in_dir, fn)
        stem = os.path.splitext(fn)[0]
        d = json.load(open(path, encoding="utf-8"))
        words = d.get("words") or []
        sents = d.get("sentences") or []
        if not words and not sents and d.get("transcription"):
            sents = [{"text": d["transcription"]}]

        # --- file duration
        audio = find_audio(path)
        fdur = audio_duration(audio) if audio else None
        src = "audio" if fdur is not None else "timestamps"
        if fdur is None:
            ends = [float(x["end_time"]) for x in (words or sents) if x.get("end_time") is not None]
            fdur = max(ends) if ends else None
        dur_source[src] += 1

        # --- word level (locations + word durations)
        digit_rows, sc_rows = [], []
        f_digits = f_numtok = f_sc = f_sym = f_punct = f_fmt = 0
        w_digit_dur = 0.0
        # map word index -> sentence index by time (sentences are time-ordered)
        s_bounds = [(float(s.get("start_time", 0) or 0), float(s.get("end_time", 0) or 0)) for s in sents]

        def sent_index(w):
            try:
                st = float(w["start_time"])
            except (KeyError, TypeError, ValueError):
                return ""
            for k, (a, b) in enumerate(s_bounds):
                if a - 1e-3 <= st <= b + 1e-3:
                    return k
            return ""

        items = words if words else [{"word": t, **{k: s.get(k) for k in ("start_time", "end_time", "speaker")}}
                                     for s in sents for t in (s.get("text") or "").split()]
        for wi, w in enumerate(items):
            tok = str(w.get("word", ""))
            wd = dur(w)
            tot["words"] += 1
            si = sent_index(w) if words else ""
            scripts = {script_of(c) for c in tok} - {None}
            for s_ in scripts:
                script_words[s_] += 1
            if sp is not None:
                for c in set(tok):
                    if not c.isspace() and sp.unk_id() in sp.encode(c):
                        oov_chars[c] += 1
            has_digit = any(is_digit(c) for c in tok)
            core = strip_edges(tok)
            if has_digit:
                f_numtok += 1
                cls = classify_numeric(tok)
                nd = sum(is_digit(c) for c in tok)
                f_digits += nd
                for c in tok:
                    if is_digit(c):
                        digit_char[c] += 1
                num_class[cls] += 1
                num_class_files[cls].add(stem)
                num_class_dur[cls] += wd or 0.0
                num_len[digit_len_bucket(tok)] += 1
                tot["words_with_digit"] += 1
                if wd is not None:
                    word_durs["digit"].append(wd)
                    w_digit_dur += wd
                row = [wi, si, w.get("speaker", ""), w.get("start_time", ""), w.get("end_time", ""),
                       round(wd, 3) if wd is not None else "", tok, core, cls, nd, digit_len_bucket(tok)]
                digit_rows.append(row)
                num_rows.append([stem] + row)
            elif wd is not None:
                word_durs["no_digit"].append(wd)
            numeric_ctx = bool(re.search(r"\d", to_ascii_digits(core)))
            w_has = Counter()
            for ci, c in enumerate(tok):
                g = sc_group(c)
                if not g:
                    continue
                inside = numeric_ctx and c in core and 0 < core.find(c) < len(core) - 1 or (numeric_ctx and c == "%")
                f_sc += 1
                w_has[g] += 1
                if g == "symbol":
                    f_sym += 1
                elif g == "punct":
                    f_punct += 1
                elif g == "format":
                    f_fmt += 1
                sc_char[c] += 1
                sc_char_files[c].add(stem)
                if inside:
                    sc_char_numeric[c] += 1
                sc_rows.append([wi, si, w.get("speaker", ""), w.get("start_time", ""), w.get("end_time", ""),
                                round(wd, 3) if wd is not None else "", ci, c, f"U+{ord(c):04X}",
                                unicodedata.name(c, "?"), g, "numeric" if inside else "text", tok])
            if w_has:
                tot["words_with_sc"] += 1
                if wd is not None:
                    word_durs["sc"].append(wd)
                for g in w_has:
                    tot[f"words_with_{g}"] += 1
                    if wd is not None:
                        word_durs[g].append(wd)

        # --- sentence level
        s_digit_dur = s_sc_dur = s_sym_dur = 0.0
        n_s_digit = n_s_sym = 0
        for k, s in enumerate(sents):
            text = s.get("text") or ""
            sd = dur(s)
            tot["sentences"] += 1
            spk = s.get("speaker", "")
            speaker_sent[spk] += 1
            toks = text.split()
            numtoks = [t for t in toks if any(is_digit(c) for c in t)]
            classes = sorted({classify_numeric(t) for t in numtoks})
            groups = Counter(sc_group(c) for c in text if sc_group(c))
            has_d, has_sym = bool(numtoks), bool(groups.get("symbol"))
            has_nonpunct = bool(groups.get("symbol") or groups.get("format") or groups.get("other"))
            scripts = {script_of(c) for c in text} - {None}
            for s_ in scripts:
                script_sent[s_] += 1
            if len(scripts) > 1:
                script_sent["mixed(>1 script)"] += 1
            dur_bucket_all[dur_bucket(sd)] += 1
            if sd is not None:
                sent_durs["all"].append(sd)
            if has_d:
                tot["sentences_with_digit"] += 1
                speaker_sent_digit[spk] += 1
                dur_bucket_digit[dur_bucket(sd)] += 1
                n_s_digit += 1
                if sd is not None:
                    sent_durs["digit"].append(sd)
                    s_digit_dur += sd
                for t in numtoks:
                    i = toks.index(t)
                    sent_pos["only token" if len(toks) == 1 else "first" if i == 0
                             else "last" if i == len(toks) - 1 else "middle"] += 1
                if all(any(is_digit(c) for c in t) or not any(script_of(c) for c in t) for t in toks):
                    tot["sentences_digit_only"] += 1
            elif sd is not None:
                sent_durs["no_digit"].append(sd)
            if groups:
                tot["sentences_with_sc"] += 1
                if sd is not None:
                    sent_durs["sc"].append(sd)
                    s_sc_dur += sd
            if has_nonpunct:
                tot["sentences_with_nonpunct_sc"] += 1
                if sd is not None:
                    sent_durs["nonpunct_sc"].append(sd)
            if has_sym:
                tot["sentences_with_symbol"] += 1
                dur_bucket_sym[dur_bucket(sd)] += 1
                n_s_sym += 1
                if sd is not None:
                    s_sym_dur += sd
            sent_rows.append([stem, k, spk, s.get("start_time", ""), s.get("end_time", ""),
                              round(sd, 3) if sd is not None else "", int(has_d), len(numtoks),
                              ",".join(classes), int(has_sym), int(has_nonpunct),
                              sum(groups.values()), "+".join(sorted(scripts)), text])

        # --- file roll-up
        tot["files"] += 1
        if fdur is not None:
            tot_dur["audio"] += fdur
            file_durs["all"].append(fdur)
        if f_numtok:
            tot["files_with_digit"] += 1
            per_file_digits.append(f_digits)
            per_file_numtok.append(f_numtok)
            if fdur is not None:
                tot_dur["files_with_digit"] += fdur
                file_durs["digit"].append(fdur)
        elif fdur is not None:
            file_durs["no_digit"].append(fdur)
        if f_sc:
            tot["files_with_sc"] += 1
            per_file_sc.append(f_sc)
            if fdur is not None:
                tot_dur["files_with_sc"] += fdur
        if f_sym:
            tot["files_with_symbol"] += 1
            per_file_sym.append(f_sym)
            if fdur is not None:
                tot_dur["files_with_symbol"] += fdur
        tot_dur["words_with_digit"] += w_digit_dur
        tot_dur["sentences_with_digit"] += s_digit_dur
        tot_dur["sentences_with_sc"] += s_sc_dur
        tot_dur["sentences_with_symbol"] += s_sym_dur

        file_rows.append([stem, round(fdur, 3) if fdur is not None else "", src, len(sents), len(items),
                          f_digits, f_numtok, n_s_digit, round(s_digit_dur, 3), round(w_digit_dur, 3),
                          f_sc, f_punct, f_sym, f_fmt, n_s_sym, round(s_sym_dur, 3)])

        hdr = ["word_idx", "sentence_idx", "speaker", "start_time", "end_time", "word_dur_s"]
        if digit_rows:
            write_tsv(os.path.join(out, "digits", stem + ".tsv"),
                      hdr + ["word", "core", "class", "n_digits", "max_digit_run"], digit_rows)
        if sc_rows:
            write_tsv(os.path.join(out, "sc", stem + ".tsv"),
                      hdr + ["char_idx_in_word", "char", "codepoint", "name", "group", "where", "word"],
                      sc_rows)

    # ------------------------------------------------------------------ tables
    write_tsv(os.path.join(out, "files.tsv"),
              ["file", "duration_s", "duration_source", "n_sentences", "n_words", "n_digits",
               "n_numeric_words", "n_sentences_with_digit", "dur_sentences_with_digit_s",
               "dur_words_with_digit_s", "n_sc", "n_punct", "n_symbol", "n_format",
               "n_sentences_with_symbol", "dur_sentences_with_symbol_s"], file_rows)
    write_tsv(os.path.join(out, "sentences.tsv"),
              ["file", "sentence_idx", "speaker", "start_time", "end_time", "duration_s", "has_digit",
               "n_numeric_words", "numeric_classes", "has_symbol", "has_nonpunct_sc", "n_sc",
               "scripts", "text"], sent_rows)
    write_tsv(os.path.join(out, "numeric_expressions.tsv"),
              ["file", "word_idx", "sentence_idx", "speaker", "start_time", "end_time", "word_dur_s",
               "word", "core", "class", "n_digits", "max_digit_run"], num_rows)
    write_tsv(os.path.join(out, "sc_chars.tsv"),
              ["char", "codepoint", "name", "group", "count", "in_numeric", "n_files"],
              [[c if c.isprintable() else "", f"U+{ord(c):04X}", unicodedata.name(c, "?"), sc_group(c),
                n, sc_char_numeric[c], len(sc_char_files[c])] for c, n in sc_char.most_common()])

    # ------------------------------------------------------------------ metrics
    F, W, S, A = tot["files"], tot["words"], tot["sentences"], tot_dur["audio"]
    n_num = sum(num_class.values())
    m = {
        "input_dir": os.path.abspath(in_dir),
        "duration_source": dict(dur_source),
        "totals": {"files": F, "sentences": S, "words": W, "audio_hours": hours(A),
                   "file_duration_s": stats(file_durs["all"]),
                   "sentence_duration_s": stats(sent_durs["all"])},
        "digits": {
            "files_with_digits": F and tot["files_with_digit"],
            "files_with_digits_pct": pct(tot["files_with_digit"], F),
            "audio_hours_files_with_digits": hours(tot_dur["files_with_digit"]),
            "audio_pct_files_with_digits": pct(tot_dur["files_with_digit"], A),
            "digit_chars_per_file (files with digits)": stats(per_file_digits),
            "numeric_words_per_file (files with digits)": stats(per_file_numtok),
            "file_duration_s_with_digits": stats(file_durs["digit"]),
            "file_duration_s_without_digits": stats(file_durs["no_digit"]),
            "words_with_digits": tot["words_with_digit"],
            "words_with_digits_pct": pct(tot["words_with_digit"], W),
            "word_duration_s_with_digits": stats(word_durs["digit"]),
            "word_duration_s_without_digits": stats(word_durs["no_digit"]),
            "audio_hours_words_with_digits": hours(tot_dur["words_with_digit"]),
            "audio_pct_words_with_digits": pct(tot_dur["words_with_digit"], A),
            "sentences_with_digits": tot["sentences_with_digit"],
            "sentences_with_digits_pct": pct(tot["sentences_with_digit"], S),
            "sentences_digit_only (no letters)": tot["sentences_digit_only"],
            "sentence_duration_s_with_digits": stats(sent_durs["digit"]),
            "sentence_duration_s_without_digits": stats(sent_durs["no_digit"]),
            "audio_hours_sentences_with_digits": hours(tot_dur["sentences_with_digit"]),
            "audio_pct_sentences_with_digits": pct(tot_dur["sentences_with_digit"], A),
            "numeric_position_in_sentence": dict(sent_pos.most_common()),
            "digit_char_frequency": {f"{c} (U+{ord(c):04X})": n for c, n in sorted(digit_char.items())},
        },
        "numeric_classes": {
            c: {"count": n, "pct": pct(n, n_num), "files": len(num_class_files[c]),
                "word_audio_s": round(num_class_dur[c], 2)}
            for c, n in num_class.most_common()},
        "numeric_max_digit_run": {k: {"count": v, "pct": pct(v, n_num)}
                                  for k, v in sorted(num_len.items(), key=lambda x: (len(x[0]), x[0]))},
        "special_chars": {
            "files_with_sc": tot["files_with_sc"], "files_with_sc_pct": pct(tot["files_with_sc"], F),
            "audio_hours_files_with_sc": hours(tot_dur["files_with_sc"]),
            "sc_per_file (files with sc)": stats(per_file_sc),
            "files_with_symbol": tot["files_with_symbol"],
            "files_with_symbol_pct": pct(tot["files_with_symbol"], F),
            "audio_hours_files_with_symbol": hours(tot_dur["files_with_symbol"]),
            "symbols_per_file (files with symbols)": stats(per_file_sym),
            "words_with_sc": tot["words_with_sc"], "words_with_sc_pct": pct(tot["words_with_sc"], W),
            "word_duration_s_with_sc": stats(word_durs["sc"]),
            **{f"words_with_{g}": tot[f"words_with_{g}"] for g in ("punct", "symbol", "format", "other")},
            **{f"word_duration_s_with_{g}": stats(word_durs[g]) for g in ("symbol", "format") if word_durs[g]},
            "sentences_with_sc": tot["sentences_with_sc"],
            "sentences_with_sc_pct": pct(tot["sentences_with_sc"], S),
            "sentence_duration_s_with_sc": stats(sent_durs["sc"]),
            "sentences_with_nonpunct_sc": tot["sentences_with_nonpunct_sc"],
            "sentence_duration_s_with_nonpunct_sc": stats(sent_durs["nonpunct_sc"]),
            "sentences_with_symbol": tot["sentences_with_symbol"],
            "audio_hours_sentences_with_symbol": hours(tot_dur["sentences_with_symbol"]),
            "sc_in_numeric_expression": sum(sc_char_numeric.values()),
            "sc_elsewhere": sum(sc_char.values()) - sum(sc_char_numeric.values()),
            "by_group": dict(Counter({sc_group(c): 0 for c in sc_char}) + Counter(
                {g: sum(n for c, n in sc_char.items() if sc_group(c) == g) for g in
                 {sc_group(c) for c in sc_char}})),
            "top_chars": {f"{c if c.isprintable() else '·'} U+{ord(c):04X} {unicodedata.name(c, '?')}": n
                          for c, n in sc_char.most_common(30)},
        },
        "sentence_duration_buckets": {
            "all": dict(sorted(dur_bucket_all.items())),
            "with_digits": dict(sorted(dur_bucket_digit.items())),
            "with_symbols": dict(sorted(dur_bucket_sym.items()))},
        "scripts": {"words_containing": dict(script_words.most_common()),
                    "sentences_containing": dict(script_sent.most_common())},
        "speakers": {spk: {"sentences": n, "with_digits": speaker_sent_digit[spk],
                           "with_digits_pct": pct(speaker_sent_digit[spk], n)}
                     for spk, n in speaker_sent.most_common()},
    }
    if sp is not None:
        m["tokenizer_oov_chars"] = {"tokenizer": os.path.abspath(tokenizer),
                                    "words_affected_by_char": {
                                        f"{c if c.isprintable() else '·'} U+{ord(c):04X} {unicodedata.name(c, '?')}": n
                                        for c, n in oov_chars.most_common()}}

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    from .scoring.html_report import page            # stdlib only, like this module
    with open(os.path.join(out, "metrics.html"), "w", encoding="utf-8") as f:
        f.write(page(f"Digit / SC distribution — {name}", render_md(name, m)))
    return out, m


def render_md(name, m):
    L = [f"# Digit / special-character distribution — `{name}`", ""]
    t, d, sc = m["totals"], m["digits"], m["special_chars"]
    L += [f"Files **{t['files']}** · sentences **{t['sentences']}** · words **{t['words']}** · "
          f"audio **{t['audio_hours']} h** (duration source: {m['duration_source']})", ""]

    def st(s, unit="s"):
        return ("—" if not s.get("n") else
                f"n={s['n']}, sum={s['sum']}{unit}, mean={s['mean']}, median={s['median']}, "
                f"p90={s['p90']}, max={s['max']}")

    L += ["## Digits", "",
          "| Metric | Value |", "|---|---|",
          f"| Files with digits | {d['files_with_digits']} ({d['files_with_digits_pct']}%) |",
          f"| Audio in files with digits | {d['audio_hours_files_with_digits']} h ({d['audio_pct_files_with_digits']}%) |",
          f"| Digit chars per file (files with digits) | {st(d['digit_chars_per_file (files with digits)'], '')} |",
          f"| Numeric words per file | {st(d['numeric_words_per_file (files with digits)'], '')} |",
          f"| Words with digits | {d['words_with_digits']} ({d['words_with_digits_pct']}%) |",
          f"| Word duration, with digits | {st(d['word_duration_s_with_digits'])} |",
          f"| Word duration, without digits | {st(d['word_duration_s_without_digits'])} |",
          f"| Audio in words with digits | {d['audio_hours_words_with_digits']} h ({d['audio_pct_words_with_digits']}%) |",
          f"| Sentences with digits | {d['sentences_with_digits']} ({d['sentences_with_digits_pct']}%) |",
          f"| Sentences that are only numbers | {d['sentences_digit_only (no letters)']} |",
          f"| Sentence duration, with digits | {st(d['sentence_duration_s_with_digits'])} |",
          f"| Sentence duration, without digits | {st(d['sentence_duration_s_without_digits'])} |",
          f"| Audio in sentences with digits | {d['audio_hours_sentences_with_digits']} h ({d['audio_pct_sentences_with_digits']}%) |",
          f"| File duration, with / without digits | {st(d['file_duration_s_with_digits'])} / {st(d['file_duration_s_without_digits'])} |",
          f"| Position of number in sentence | {d['numeric_position_in_sentence']} |",
          "", "### Numeric expression classes", "",
          "| Class | Count | % | Files | Word audio (s) |", "|---|---|---|---|---|"]
    for c, v in m["numeric_classes"].items():
        L.append(f"| {c} | {v['count']} | {v['pct']} | {v['files']} | {v['word_audio_s']} |")
    L += ["", "### Longest digit run per numeric word", "", "| Digits | Count | % |", "|---|---|---|"]
    for k, v in m["numeric_max_digit_run"].items():
        L.append(f"| {k} | {v['count']} | {v['pct']} |")
    L += ["", "## Special characters", "", "| Metric | Value |", "|---|---|",
          f"| Files with any SC | {sc['files_with_sc']} ({sc['files_with_sc_pct']}%) |",
          f"| SC per file | {st(sc['sc_per_file (files with sc)'], '')} |",
          f"| Files with symbols (S*: % ₹ × …) | {sc['files_with_symbol']} ({sc['files_with_symbol_pct']}%), {sc['audio_hours_files_with_symbol']} h |",
          f"| Words with SC | {sc['words_with_sc']} ({sc['words_with_sc_pct']}%) — punct {sc['words_with_punct']}, symbol {sc['words_with_symbol']}, format {sc['words_with_format']}, other {sc['words_with_other']} |",
          f"| Word duration, with SC | {st(sc['word_duration_s_with_sc'])} |",
          f"| Sentences with SC | {sc['sentences_with_sc']} ({sc['sentences_with_sc_pct']}%) |",
          f"| Sentences with non-punctuation SC | {sc['sentences_with_nonpunct_sc']} — duration {st(sc['sentence_duration_s_with_nonpunct_sc'])} |",
          f"| Sentences with symbols | {sc['sentences_with_symbol']} ({sc['audio_hours_sentences_with_symbol']} h) |",
          f"| SC inside a numeric expression / elsewhere | {sc['sc_in_numeric_expression']} / {sc['sc_elsewhere']} |",
          f"| By group | {sc['by_group']} |",
          "", "### Top characters", "", "| Char | Count |", "|---|---|"]
    for k, v in sc["top_chars"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Sentence duration buckets", "", "| Bucket | All | With digits | With symbols |", "|---|---|---|---|"]
    b = m["sentence_duration_buckets"]
    for k in sorted(set(b["all"]) | set(b["with_digits"]), key=lambda x: (x == "unknown", float(re.match(r"[\d.]+", x).group()) if re.match(r"[\d.]+", x) else 0)):
        L.append(f"| {k} | {b['all'].get(k, 0)} | {b['with_digits'].get(k, 0)} | {b['with_symbols'].get(k, 0)} |")
    L += ["", "## Scripts", "", f"Words containing: {m['scripts']['words_containing']}", "",
          f"Sentences containing: {m['scripts']['sentences_containing']}", "",
          "## Speakers", "", "| Speaker | Sentences | With digits | % |", "|---|---|---|---|"]
    for spk, v in m["speakers"].items():
        L.append(f"| {spk} | {v['sentences']} | {v['with_digits']} | {v['with_digits_pct']} |")
    if "tokenizer_oov_chars" in m:
        L += ["", "## Characters the tokenizer can't encode (words affected)", ""]
        for k, v in m["tokenizer_oov_chars"]["words_affected_by_char"].items():
            L.append(f"- {k}: {v}")
    L.append("")
    return "\n".join(L)


def add_arguments(ap):
    ap.add_argument("in_dir", help="directory of per-recording JSON files (e.g. eval_frozen)")
    ap.add_argument("--out-root", default="outputs", help="output root (default: ./outputs)")
    ap.add_argument("--tokenizer", help="optional SentencePiece .model to flag unencodable chars")


def run(a):
    out, m = analyse(a.in_dir, a.out_root, a.tokenizer)
    d = m["digits"]
    print(f"wrote {out}\n  files with digits: {d['files_with_digits']}/{m['totals']['files']} "
          f"({d['files_with_digits_pct']}%), sentences with digits: {d['sentences_with_digits']} "
          f"({d['sentences_with_digits_pct']}%), numeric words: {d['words_with_digits']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("in_dir", help="directory of per-recording JSON files (e.g. eval_frozen)")
    ap.add_argument("--out-root", default="outputs", help="output root (default: ./outputs)")
    ap.add_argument("--tokenizer", help="optional SentencePiece .model to flag unencodable chars")
    a = ap.parse_args(argv)
    out, m = analyse(a.in_dir, a.out_root, a.tokenizer)
    d = m["digits"]
    print(f"wrote {out}\n  files with digits: {d['files_with_digits']}/{m['totals']['files']} "
          f"({d['files_with_digits_pct']}%), sentences with digits: {d['sentences_with_digits']} "
          f"({d['sentences_with_digits_pct']}%), numeric words: {d['words_with_digits']}")


__all__ = ["analyse", "classify_numeric", "sc_group", "main"]


if __name__ == "__main__":
    main()
