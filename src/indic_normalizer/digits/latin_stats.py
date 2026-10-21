"""Distribution of English (Latin-script) characters and words in an ASR transcript dir.

Input: a directory like ``eval_frozen`` — one JSON per recording with
    {"transcription": str,
     "words":     [{"word", "start_time", "end_time", "speaker"}, ...],
     "sentences": [{"text", "start_time", "end_time", "speaker"}, ...]}
Sentences are the unit (``transcription`` is used when a file has no sentences); word timings
come from ``words``. By default the text is analysed as written (``--normalize none``); with
``--normalize train`` / ``eval`` it is first normalized like a training target / scoring text.

Output: ``<out-root>/<input dir name>/latin/`` (default out-root: ./outputs)
    latin_metrics.json   every metric, machine-readable
    latin_metrics.html   the same as a report (contents, sortable tables)
    latin_chars.tsv      one row per Latin character: count, case, files, words
    latin_words.tsv      one row per distinct Latin-bearing word: count, files, kind, shape, example
    latin_tokens.tsv     every Latin-bearing word occurrence: file, sentence, time, speaker, context
    files.tsv            one row per file: counts and audio share of sentences with Latin

Definitions
    Latin char   a letter whose Unicode name starts with LATIN (A-Z, a-z and accented forms
                 like é); fullwidth forms count too. Digits and punctuation are not letters.
    token        a whitespace-separated word, sentence punctuation stripped from its edges.
    kind         english   only Latin letters (inner ' - . allowed: e-mail, don't, U.S.)
                 mixed     Latin letters + Malayalam letters (ATM-ൽ, Wi-Fiയിൽ)
                 alnum     Latin letters + digits, no Malayalam (5G, COVID19, A4)
                 other     Latin letters + anything else
    shape        acronym   2+ letters, all upper case (ATM, KSEB)
                 title     first letter upper, rest lower (Kerala)
                 lower     all lower case (ok, phone)
                 camel     mixed case otherwise (iPhone, YouTube)
                 letter    a single letter (A, b)

Usage
    indic-digits latin PATH/TO/eval_frozen [--out-root outputs] [--normalize none|train|eval]
    python -m indic_normalizer.digits.latin_stats PATH/TO/eval_frozen

Standard library only (the normalizer is imported only with --normalize train/eval).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from .stats import (context, dur, dur_bucket, find_audio, audio_duration, hours, pct, stats,
                    strip_edges, write_tsv)

KINDS = ("english", "mixed", "alnum", "other")
SHAPES = ("acronym", "title", "lower", "camel", "letter")
_ML_LO, _ML_HI = 0x0D00, 0x0D7F
_INNER = set("'’-.&_/")


def is_latin(ch: str) -> bool:
    """A Latin-script letter (A-Z, a-z, accented, fullwidth)."""
    if ch.isascii():
        return ch.isalpha()
    if not ch.isalpha():
        return False
    try:
        return unicodedata.name(ch).startswith(("LATIN", "FULLWIDTH LATIN"))
    except ValueError:
        return False


def is_malayalam_letter(ch: str) -> bool:
    return _ML_LO <= ord(ch) <= _ML_HI and not ch.isdigit()


def token_kind(tok: str) -> str:
    has_ml = any(is_malayalam_letter(c) for c in tok)
    has_digit = any(c.isdecimal() for c in tok)
    if has_ml:
        return "mixed"
    if has_digit:
        return "alnum"
    if all(is_latin(c) or c in _INNER for c in tok):
        return "english"
    return "other"


def token_shape(tok: str) -> str:
    letters = [c for c in tok if is_latin(c)]
    if len(letters) == 1:
        return "letter"
    if all(c.isupper() for c in letters):
        return "acronym"
    if all(c.islower() for c in letters):
        return "lower"
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "title"
    return "camel"


def latin_part(tok: str) -> str:
    """The Latin letters of a token, inner joiners kept (ATM-ൽ -> ATM, Wi-Fiയിൽ -> Wi-Fi)."""
    runs = re.findall(r"[^\W\d_](?:[^\W\d_]|['’\-.&/])*", tok)
    runs = ["".join(c for c in r if is_latin(c) or c in _INNER).strip("".join(_INNER)) for r in runs]
    return " ".join(r for r in runs if any(is_latin(c) for c in r))


def _sentences(doc):
    s = doc.get("sentences")
    if isinstance(s, list) and s:
        return s
    return [{"text": doc.get("transcription") or ""}]


def _normalizer(mode):
    if mode in (None, "", "none"):
        return lambda t: t
    from ..languages.malayalam import normalize
    return lambda t: normalize(t, mode=mode)


def analyse(in_dir, out_root="outputs", normalize="none", top=200):
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".json"))
    if not files:
        sys.exit(f"no .json files in {in_dir}")
    name = os.path.basename(os.path.normpath(in_dir))
    out = os.path.join(out_root, name, "latin")
    os.makedirs(out, exist_ok=True)
    norm = _normalizer(normalize)

    n_sent = n_tok = n_letters = n_latin = n_ml_letters = 0
    char_count, char_files, char_words = Counter(), defaultdict(set), Counter()
    word_count, word_files, word_kind, word_shape, word_example = Counter(), defaultdict(set), {}, {}, {}
    key_count, key_forms = Counter(), defaultdict(Counter)          # case-folded Latin part
    kind_count, shape_count, kind_files = Counter(), Counter(), defaultdict(set)
    tok_len, pos_count, speaker_tok, speaker_sent, speaker_sent_latin = Counter(), Counter(), Counter(), Counter(), Counter()
    sent_dur_latin, sent_dur_other, word_dur_latin, word_dur_other = [], [], [], []
    bucket_all, bucket_latin = Counter(), Counter()
    latin_per_sentence = Counter()
    tok_rows, file_rows = [], []
    sents_latin = sents_only_latin = 0
    total_audio = 0.0
    dur_sources = Counter()

    for fn in files:
        path = os.path.join(in_dir, fn)
        stem = fn[:-5]
        try:
            doc = json.load(open(path, encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            print(f"skip {fn}: {e}", file=sys.stderr)
            continue
        sents = _sentences(doc)
        f_tok = f_latin_tok = f_latin_chars = f_letters = f_sent_latin = 0
        f_dur_latin = 0.0
        for si, s in enumerate(sents):
            text = norm(str(s.get("text") or ""))
            n_sent += 1
            spk = str(s.get("speaker") or "")
            speaker_sent[spk] += 1
            d = dur(s)
            bucket_all[dur_bucket(d)] += 1
            toks = list(re.finditer(r"\S+", text))
            latin_toks = 0
            for k, m in enumerate(toks):
                raw = m.group(0)
                core = strip_edges(raw)
                n_tok += 1
                f_tok += 1
                letters = [c for c in core if c.isalpha()]
                lat = [c for c in core if is_latin(c)]
                n_letters += len(letters)
                f_letters += len(letters)
                n_ml_letters += sum(1 for c in letters if is_malayalam_letter(c))
                if not lat:
                    continue
                latin_toks += 1
                f_latin_tok += 1
                n_latin += len(lat)
                f_latin_chars += len(lat)
                for c in lat:
                    char_count[c] += 1
                    char_files[c].add(stem)
                for c in set(lat):
                    char_words[c] += 1
                kind, shape = token_kind(core), token_shape(core)
                kind_count[kind] += 1
                shape_count[shape] += 1
                kind_files[kind].add(stem)
                tok_len[min(len(lat), 15)] += 1
                speaker_tok[spk] += 1
                pos = "only" if len(toks) == 1 else ("start" if k == 0 else ("end" if k == len(toks) - 1 else "middle"))
                pos_count[pos] += 1
                word_count[core] += 1
                word_files[core].add(stem)
                word_kind[core], word_shape[core] = kind, shape
                word_example.setdefault(core, f"{stem}#{si}: {context(text, m.start(), m.end())}")
                key = latin_part(core).lower()
                key_count[key] += 1
                key_forms[key][core] += 1
                tok_rows.append([stem, si, s.get("start_time"), s.get("end_time"), spk, core, latin_part(core),
                                 kind, shape, context(text, m.start(), m.end())])
            latin_per_sentence[min(latin_toks, 10)] += 1
            if latin_toks:
                sents_latin += 1
                f_sent_latin += 1
                speaker_sent_latin[spk] += 1
                bucket_latin[dur_bucket(d)] += 1
                if latin_toks == len(toks):
                    sents_only_latin += 1
                if d is not None:
                    sent_dur_latin.append(d)
                    f_dur_latin += d
            elif d is not None:
                sent_dur_other.append(d)
        for w in doc.get("words") or []:
            core = strip_edges(norm(str(w.get("word") or "")))
            d = dur(w)
            if d is None or not core:
                continue
            (word_dur_latin if any(is_latin(c) for c in core) else word_dur_other).append(d)
        wav = find_audio(path)
        a_len = audio_duration(wav) if wav else None
        if a_len is None:
            ends = [float(x.get("end_time") or 0) for x in (doc.get("words") or []) + sents
                    if isinstance(x, dict) and x.get("end_time") is not None]
            a_len, src = (max(ends) if ends else 0.0), "timestamps"
        else:
            src = "audio"
        dur_sources[src] += 1
        total_audio += a_len
        file_rows.append([stem, len(sents), f_tok, f_latin_tok, pct(f_latin_tok, f_tok), f_latin_chars,
                          pct(f_latin_chars, f_letters), f_sent_latin, pct(f_sent_latin, len(sents)),
                          round(a_len, 2), round(f_dur_latin, 2), pct(f_dur_latin, a_len), src])

    n_files = len(file_rows)
    files_latin = sum(1 for r in file_rows if r[3])
    folded = Counter()
    for c, n in char_count.items():
        folded[unicodedata.normalize("NFKD", c)[0].lower()] += n
    m = {
        "input": os.path.abspath(in_dir), "normalize": normalize,
        "totals": {"files": n_files, "sentences": n_sent, "tokens": n_tok, "letters": n_letters,
                   "audio_hours": hours(total_audio), "duration_source": dict(dur_sources)},
        "latin": {
            "files_with_latin": files_latin, "files_with_latin_pct": pct(files_latin, n_files),
            "sentences_with_latin": sents_latin, "sentences_with_latin_pct": pct(sents_latin, n_sent),
            "sentences_only_latin": sents_only_latin,
            "tokens_with_latin": sum(kind_count.values()),
            "tokens_with_latin_pct": pct(sum(kind_count.values()), n_tok),
            "latin_chars": n_latin, "latin_chars_pct_of_letters": pct(n_latin, n_letters),
            "malayalam_letters": n_ml_letters, "malayalam_letters_pct_of_letters": pct(n_ml_letters, n_letters),
            "distinct_tokens": len(word_count), "distinct_words_casefolded": len(key_count),
            "uppercase_chars": sum(n for c, n in char_count.items() if c.isupper()),
            "lowercase_chars": sum(n for c, n in char_count.items() if c.islower()),
            "non_ascii_latin_chars": sum(n for c, n in char_count.items() if not c.isascii()),
            "sentence_duration_s_with_latin": stats(sent_dur_latin),
            "sentence_duration_s_without_latin": stats(sent_dur_other),
            "audio_hours_sentences_with_latin": hours(sum(sent_dur_latin)),
            "audio_pct_sentences_with_latin": pct(sum(sent_dur_latin), total_audio),
            "word_duration_s_with_latin": stats(word_dur_latin),
            "word_duration_s_without_latin": stats(word_dur_other),
        },
        "kinds": {k: {"tokens": kind_count[k], "pct": pct(kind_count[k], sum(kind_count.values())),
                      "files": len(kind_files[k])} for k in KINDS},
        "shapes": {k: {"tokens": shape_count[k], "pct": pct(shape_count[k], sum(shape_count.values()))}
                   for k in SHAPES},
        "letters_casefolded": {c: {"count": n, "pct": pct(n, n_latin)} for c, n in sorted(folded.items())},
        "latin_letters_per_token": {("15+" if k == 15 else str(k)): n for k, n in sorted(tok_len.items())},
        "latin_tokens_per_sentence": {("10+" if k == 10 else str(k)): n for k, n in sorted(latin_per_sentence.items())},
        "position_in_sentence": dict(pos_count),
        "sentence_duration_buckets": {b: {"all": bucket_all[b], "with_latin": bucket_latin[b],
                                          "pct": pct(bucket_latin[b], bucket_all[b])}
                                      for b in sorted(bucket_all, key=_bucket_key)},
        "speakers": {spk or "(none)": {"sentences": speaker_sent[spk], "with_latin": speaker_sent_latin[spk],
                                       "pct": pct(speaker_sent_latin[spk], speaker_sent[spk]),
                                       "latin_tokens": speaker_tok[spk]}
                     for spk, _ in speaker_sent.most_common()},
        "top_words_casefolded": [{"word": k, "count": n, "spellings": dict(key_forms[k].most_common(5))}
                                 for k, n in key_count.most_common(top)],
    }

    write_tsv(os.path.join(out, "latin_chars.tsv"),
              ["char", "codepoint", "name", "case", "count", "pct_of_latin", "tokens", "files"],
              [[c, f"U+{ord(c):04X}", unicodedata.name(c, "?"), "upper" if c.isupper() else "lower", n,
                pct(n, n_latin), char_words[c], len(char_files[c])] for c, n in char_count.most_common()])
    write_tsv(os.path.join(out, "latin_words.tsv"),
              ["token", "latin_part", "count", "files", "kind", "shape", "example"],
              [[w, latin_part(w), n, len(word_files[w]), word_kind[w], word_shape[w], word_example[w]]
               for w, n in word_count.most_common()])
    write_tsv(os.path.join(out, "latin_tokens.tsv"),
              ["file", "sentence", "start_time", "end_time", "speaker", "token", "latin_part", "kind", "shape",
               "context"], tok_rows)
    write_tsv(os.path.join(out, "files.tsv"),
              ["file", "sentences", "tokens", "latin_tokens", "latin_tokens_pct", "latin_chars",
               "latin_chars_pct_of_letters", "sentences_with_latin", "sentences_with_latin_pct", "audio_s",
               "audio_s_sentences_with_latin", "audio_pct_sentences_with_latin", "duration_source"], file_rows)
    with open(os.path.join(out, "latin_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    from .scoring.html_report import page            # stdlib only
    with open(os.path.join(out, "latin_metrics.html"), "w", encoding="utf-8") as f:
        f.write(page(f"English characters — {name}", render_md(name, m, file_rows, word_count, word_files,
                                                                word_kind, word_shape)))
    return out, m


def _bucket_key(b):
    try:
        return float(b.split("-")[0].rstrip("s+"))
    except ValueError:
        return 1e9


def _md(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_md(name, m, file_rows, word_count, word_files, word_kind, word_shape, top=100):
    t, l = m["totals"], m["latin"]

    def st(s, unit="s"):
        return ("—" if not s.get("n") else
                f"n={s['n']}, sum={s['sum']}{unit}, mean={s['mean']}, median={s['median']}, "
                f"p90={s['p90']}, max={s['max']}")

    L = [f"# English (Latin) characters — `{name}`", "",
         f"Files **{t['files']}** · sentences **{t['sentences']}** · tokens **{t['tokens']}** · "
         f"letters **{t['letters']}** · audio **{t['audio_hours']} h** · text normalization: "
         f"**{m['normalize']}**", "",
         "Latin char = A–Z / a–z (and accented Latin letters); token = whitespace word without edge "
         "punctuation. Kinds: *english* only Latin letters, *mixed* Latin + Malayalam (ATM-ൽ), *alnum* "
         "Latin + digits (5G), *other*. Shapes: *acronym* ATM, *title* Kerala, *lower* ok, *camel* iPhone, "
         "*letter* A.", "",
         "## Overview", "", "| Metric | Value |", "|---|---|",
         f"| Files with Latin | {l['files_with_latin']} ({l['files_with_latin_pct']}%) |",
         f"| Sentences with Latin | {l['sentences_with_latin']} ({l['sentences_with_latin_pct']}%) |",
         f"| Sentences that are only Latin tokens | {l['sentences_only_latin']} |",
         f"| Tokens with Latin | {l['tokens_with_latin']} ({l['tokens_with_latin_pct']}%) |",
         f"| Distinct tokens / words (case-folded) | {l['distinct_tokens']} / {l['distinct_words_casefolded']} |",
         f"| Latin letters | {l['latin_chars']} ({l['latin_chars_pct_of_letters']}% of all letters) |",
         f"| Malayalam letters | {l['malayalam_letters']} ({l['malayalam_letters_pct_of_letters']}% of all letters) |",
         f"| Upper / lower case Latin letters | {l['uppercase_chars']} / {l['lowercase_chars']} |",
         f"| Non-ASCII Latin letters (é, ﬁ, Ａ …) | {l['non_ascii_latin_chars']} |",
         f"| Audio in sentences with Latin | {l['audio_hours_sentences_with_latin']} h ({l['audio_pct_sentences_with_latin']}%) |",
         f"| Sentence duration, with Latin | {st(l['sentence_duration_s_with_latin'])} |",
         f"| Sentence duration, without Latin | {st(l['sentence_duration_s_without_latin'])} |",
         f"| Word duration, with Latin | {st(l['word_duration_s_with_latin'])} |",
         f"| Word duration, without Latin | {st(l['word_duration_s_without_latin'])} |", ""]
    L += ["## Token kinds", "", "| Kind | Tokens | % | Files |", "|---|---|---|---|"]
    L += [f"| {k} | {v['tokens']} | {v['pct']} | {v['files']} |" for k, v in m["kinds"].items()]
    L += ["", "## Token shapes", "", "| Shape | Tokens | % |", "|---|---|---|"]
    L += [f"| {k} | {v['tokens']} | {v['pct']} |" for k, v in m["shapes"].items()]
    L += ["", "## Letter frequency (case-folded)", "", "| Letter | Count | % of Latin letters |", "|---|---|---|"]
    L += [f"| {_md(c)} | {v['count']} | {v['pct']} |"
          for c, v in sorted(m["letters_casefolded"].items(), key=lambda kv: -kv[1]["count"])]
    L += ["", "## Latin letters per token", "", "| Letters | Tokens |", "|---|---|"]
    L += [f"| {k} | {n} |" for k, n in m["latin_letters_per_token"].items()]
    L += ["", "## Latin tokens per sentence", "", "| Latin tokens | Sentences |", "|---|---|"]
    L += [f"| {k} | {n} |" for k, n in m["latin_tokens_per_sentence"].items()]
    L += ["", "## Position in the sentence", "", "| Position | Tokens |", "|---|---|"]
    L += [f"| {k} | {n} |" for k, n in sorted(m["position_in_sentence"].items(), key=lambda kv: -kv[1])]
    L += ["", "## Sentence duration buckets", "", "| Duration | Sentences | With Latin | % |", "|---|---|---|---|"]
    L += [f"| {b} | {v['all']} | {v['with_latin']} | {v['pct']} |" for b, v in m["sentence_duration_buckets"].items()]
    L += ["", "## Speakers", "", "| Speaker | Sentences | With Latin | % | Latin tokens |", "|---|---|---|---|---|"]
    L += [f"| {_md(s)} | {v['sentences']} | {v['with_latin']} | {v['pct']} | {v['latin_tokens']} |"
          for s, v in m["speakers"].items()]
    L += ["", f"## Top {top} words (case-folded)", "", "| Word | Count | Spellings |", "|---|---|---|"]
    L += [f"| {_md(w['word'])} | {w['count']} | {_md(', '.join(f'{k} ({n})' for k, n in w['spellings'].items()))} |"
          for w in m["top_words_casefolded"][:top]]
    L += ["", f"## Top {top} tokens as written", "", "| Token | Count | Files | Kind | Shape |", "|---|---|---|---|---|"]
    L += [f"| {_md(w)} | {n} | {len(word_files[w])} | {word_kind[w]} | {word_shape[w]} |"
          for w, n in word_count.most_common(top)]
    L += ["", "## Files with the most Latin", "",
          "| File | Sentences | Latin tokens | % tokens | % letters | Sentences with Latin | % audio |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {_md(r[0])} | {r[1]} | {r[3]} | {r[4]} | {r[6]} | {r[7]} | {r[11]} |"
          for r in sorted(file_rows, key=lambda r: -r[3])[:top] if r[3]]
    L += ["", "## Output files", "",
          "- `latin_chars.tsv` — every Latin character: count, case, tokens, files",
          "- `latin_words.tsv` — every distinct Latin-bearing token: count, files, kind, shape, example",
          "- `latin_tokens.tsv` — every occurrence with file, sentence, time, speaker and context",
          "- `files.tsv` — per-file counts and audio share", "- `latin_metrics.json` — all metrics"]
    return "\n".join(L)


def add_arguments(ap):
    ap.add_argument("in_dir", help="directory of per-recording JSON files (e.g. eval_frozen)")
    ap.add_argument("--out-root", default="outputs", help="output root (default: ./outputs) -> <root>/<dir>/latin/")
    ap.add_argument("--normalize", default="none", choices=["none", "train", "eval"],
                    help="analyse the text as written (none, default) or after malayalam.normalize(mode)")
    ap.add_argument("--top", type=int, default=200, help="words kept in the top lists (default 200)")


def run(a):
    out, m = analyse(a.in_dir, a.out_root, a.normalize, a.top)
    l = m["latin"]
    print(f"wrote {out}\n  files with Latin: {l['files_with_latin']}/{m['totals']['files']} "
          f"({l['files_with_latin_pct']}%), sentences: {l['sentences_with_latin']} "
          f"({l['sentences_with_latin_pct']}%), tokens: {l['tokens_with_latin']} ({l['tokens_with_latin_pct']}%), "
          f"Latin letters: {l['latin_chars']} ({l['latin_chars_pct_of_letters']}% of letters)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
