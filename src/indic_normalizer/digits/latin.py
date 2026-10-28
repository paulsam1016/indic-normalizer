"""English number readings in LATIN script, for sentences written in English.

A transcript sentence that is mostly Latin-script ("I was looking for a room for 5 to 6 days")
gets its numbers read in Latin-script English ("five", "twelve thirty", "one thousand two
hundred"), not in Malayalam script. Readings are lower case (train-normalized text is) and
use Indian grouping (one lakh seventy five thousand). Forms are ``latin_<form>``; all are
``lang="en"``. ``latin_share()`` decides whether a sentence counts as English.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from .types import Candidate

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty",
        9: "ninety"}
ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
             "nine": "ninth", "twelve": "twelfth"}
MAX = 9_99_99_99_999
LATIN_MIN_SHARE = 0.5          # a sentence is English when at least half its letters are Latin


def _is_ml(c: str) -> bool:
    """A Malayalam letter or sign (vowel signs and virama count: they are written characters)."""
    return 0x0D00 <= ord(c) <= 0x0D7F and not c.isdecimal()


def latin_share(text: str) -> float:
    """Share of Latin letters among Latin + Malayalam / Devanagari letters (0 when there are none)."""
    lat = ml = 0
    for c in text or "":
        if _is_ml(c) or ("\u0900" <= c <= "\u097f" and (c.isalpha() or unicodedata.category(c).startswith("M"))):
            ml += 1
        elif c.isalpha() and (c.isascii() or unicodedata.name(c, "").startswith("LATIN")):
            lat += 1
    return lat / (lat + ml) if lat + ml else 0.0


def is_english(text: str) -> bool:
    return latin_share(text) >= LATIN_MIN_SHARE


def foreign_share(text: str) -> float:
    """Share of letters that are NOT Malayalam (Latin, Devanagari, ...) among all letters."""
    ml = other = 0
    for c in text or "":
        if _is_ml(c):
            ml += 1
        elif c.isalpha() or unicodedata.category(c) in ("Mn", "Mc"):
            other += 1
    return other / (ml + other) if ml + other else 0.0


def has_letters(text: str) -> bool:
    return any(c.isalpha() for c in text or "")


def sentence_is_english(text: str, default: bool = False) -> bool:
    """is_english for a sentence; one with no letters at all ("0031.", "155.") takes ``default``
    — the language of the whole document."""
    return is_english(text) if has_letters(text) else default


def words(n: int) -> str:
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        t, u = divmod(n, 10)
        return TENS[t] + ("" if u == 0 else " " + ONES[u])
    for unit, name in ((10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand"), (100, "hundred")):
        if n >= unit:
            q, r = divmod(n, unit)
            return f"{words(q)} {name}" + ("" if r == 0 else " " + words(r))
    return ""


def digits(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def doubled(s: str) -> Optional[str]:
    """55 -> double five, 555 -> triple five, 98100 -> nine eight one double zero."""
    runs = re.findall(r"(\d)(\1*)", s)
    if not any(len(rest) for _, rest in runs):
        return None
    out = []
    for d, rest in runs:
        k = 1 + len(rest)
        while k > 0:
            step = 3 if k == 3 else min(k, 2)
            out.append({1: "", 2: "double ", 3: "triple "}[step] + ONES[int(d)])
            k -= step
    return " ".join(out)


def pairs(n: int) -> Optional[str]:
    """428 -> four twenty eight, 405 -> four oh five, 1200 -> twelve hundred."""
    if not (100 <= n <= 9999):
        return None
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{words(hi)} hundred" if hi >= 11 and hi % 10 else None
    return f"{words(hi)} " + (words(lo) if lo >= 10 else "oh " + ONES[lo])


def year(n: int) -> str:
    if 2000 <= n <= 2009:
        return words(n)
    hi, lo = divmod(n, 100)
    return words(hi) + (" " + (words(lo) if lo >= 10 else "oh " + ONES[lo]) if lo else " hundred")


def ordinal(n: int) -> str:
    w = words(n).split()
    last = w[-1]
    if last in ORD_IRREG:
        last = ORD_IRREG[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join(w[:-1] + [last])


def _num(n: int, s: str) -> List[Candidate]:
    out = [Candidate(words(n), "en", "latin_english")]
    for f, form in ((pairs(n), "latin_pairs"),):
        if f:
            out.append(Candidate(f, "en", form))
    if n in (100, 1000):
        out.append(Candidate("hundred" if n == 100 else "thousand", "en", "latin_bare"))
    if 1000 < n < 2000:                       # thousand six hundred
        out.append(Candidate(words(n)[len("one "):], "en", "latin_no_one"))
    if 101 <= n <= 999 and n % 100:          # 155 for 1,55,000 -> one lakh fifty five
        out.append(Candidate(f"{ONES[n // 100]} lakh {words(n % 100)}", "en", "latin_lakh_short"))
    if 1000 <= n < 100_000 and n % 1000 == 0:
        out.append(Candidate(f"{words(n // 1000)} k", "en", "latin_k"))
    if n >= 200_000 and n % 100_000 == 0 and n < 10_000_000:
        out.append(Candidate(f"{words(n // 100_000)} lakhs", "en", "latin_lakhs"))
    if len(s) >= 2 and not re.fullmatch(r"[1-9]0+", s):
        out.append(Candidate(digits(s), "en", "latin_digit_by_digit"))
        d = doubled(s)
        if d:
            out.append(Candidate(d, "en", "latin_doubled"))
    if 1 <= n <= 31:
        out.append(Candidate(ordinal(n), "en", "latin_ordinal"))
    return out


def _int(s: str) -> List[Candidate]:
    s = s.replace(",", "")
    if (len(s) > 1 and s[0] == "0") or int(s) > MAX:
        out = [Candidate(digits(s), "en", "latin_digit_by_digit")]
        d = doubled(s)
        return out + ([Candidate(d, "en", "latin_doubled")] if d else [])
    return _num(int(s), s)


def _decimal(s: str) -> List[Candidate]:
    whole, frac = s.split(".")
    w = int(whole.replace(",", "") or 0)
    head = words(w) if w <= MAX else digits(whole)
    out = [Candidate(f"{head} point {digits(frac)}", "en", "latin_point")]
    if frac.rstrip("0") == "5":
        out.append(Candidate(f"{head} and a half" if w else "half", "en", "latin_half"))
    return out


def _time(s: str) -> List[Candidate]:
    h, m = (int(x) for x in re.split(r"[:.]", s))
    H = words(h)
    if m == 0:
        return [Candidate(f"{H} o clock", "en", "latin_oclock"), Candidate(H, "en", "latin_bare")]
    M = words(m) if m >= 10 else "oh " + ONES[m]
    out = [Candidate(f"{H} {M}", "en", "latin_hh_mm")]
    if m == 30:
        out.append(Candidate(f"half past {H}", "en", "latin_half_past"))
    return out


def _number_any(s: str) -> List[Candidate]:
    s = s.replace(",", "")
    return _decimal(s) if "." in s else _int(s)


def _letters(run: str) -> str:
    return " ".join(run.lower()) if (len(run) <= 3 or run.isupper()) else run.lower()


def _piecewise(s: str) -> List[Candidate]:
    """5G -> five g, A1 -> a one, 3+2 -> three plus two."""
    ws = []
    for p in re.findall(r"[0-9]+(?:\.[0-9]+)?|[A-Za-z]+|\S", s):
        if p[0].isdecimal():
            ws.append(_number_any(p)[0].text)
        elif p.isascii() and p.isalpha():
            ws.append(_letters(p))
        elif p == "%":
            ws.append("percent")
        elif p == "+":
            ws.append("plus")
        elif p == "&":
            ws.append("and")
    text = " ".join(w for w in ws if w)
    return [Candidate(text, "en", "latin_piecewise")] if text else []


def candidates(core: str, cls: str) -> List[Candidate]:
    """Latin-script English readings of one number token (``cls`` from detect.classify), or []
    when the token can't be read this way (e.g. a Malayalam suffix: 23-ന്)."""
    from .detect import to_ascii_digits
    s = to_ascii_digits(core).strip()
    try:
        if cls in ("integer", "grouped", "id_code"):
            return _int(s)
        if cls == "year":
            n = int(s)
            return [Candidate(year(n), "en", "latin_year")] + _num(n, s)
        if cls == "phone":
            d = re.sub(r"\D", "", s)
            plus = "plus " if s.startswith("+") else ""
            out = [Candidate(plus + digits(d), "en", "latin_digit_by_digit")]
            dd = doubled(d)
            return out + ([Candidate(plus + dd, "en", "latin_doubled")] if dd else [])
        if cls == "decimal":
            return _decimal(s)
        if cls == "percent":
            n = [c for c in _number_any(s.rstrip("% ").strip())
                 if not c.form.startswith(("latin_digit_by_digit", "latin_doubled", "latin_lakh", "latin_pairs",
                                           "latin_k", "latin_ordinal"))]
            return [Candidate(f"{c.text} {w}", "en", f"{c.form}+{w}") for c in n for w in ("percent", "percentage")]
        if cls == "currency":
            n = [c for c in _number_any(re.sub(r"(?i)^(₹|rs\.?|inr)\s*|\s*₹$", "", s))
                 if not c.form.startswith(("latin_digit_by_digit", "latin_doubled"))]
            return [Candidate(f"{c.text} rupees", "en", c.form + "+rupees") for c in n] + \
                [Candidate(c.text, "en", c.form + "+nocurrency") for c in n]
        if cls == "time":
            return _time(s)
        if cls in ("time_range", "range"):
            a, b = re.split(r"\s*[-–—~]\s*", s, maxsplit=1)
            part = _time if cls == "time_range" else _number_any
            A, B = part(a)[0].text, part(b)[0].text
            return [Candidate(f"{A} to {B}", "en", "latin_a_to_b"), Candidate(f"{A} {B}", "en", "latin_a_b")]
        if cls == "date":
            d, m, y = re.split(r"[/.\-]", s)
            yy = year(int(y)) if len(y) == 4 else words(int(y))
            return [Candidate(f"{words(int(d))} {words(int(m))} {yy}", "en", "latin_d_m_y"),
                    Candidate(f"{ordinal(int(d))} {words(int(m))} {yy}", "en", "latin_dth_m_y")]
        if cls == "ordinal":
            m = re.match(r"^(\d+)(?:st|nd|rd|th)$", s, re.I)
            return [Candidate(ordinal(int(m.group(1))), "en", "latin_ordinal")] if m else []
        if cls == "slash":
            a, b = (int(x) for x in s.split("/"))
            frac = {(1, 2): "half", (1, 4): "quarter", (3, 4): "three quarters"}.get((a, b))
            out = [Candidate(frac, "en", "latin_fraction")] if frac else []
            return out + [Candidate(f"{words(a)} by {words(b)}", "en", "latin_a_by_b"),
                          Candidate(f"{words(a)} {words(b)}", "en", "latin_a_b")]
        if cls in ("alnum", "other"):
            if re.search(r"[ഀ-ൿ]", s):
                return []
            return _piecewise(s)
    except (ValueError, KeyError, IndexError, AttributeError):
        return []
    return []


# ----------------------------------------------------------------------------- per-sentence language
# Which language a sentence's numbers are read in ("ml": Malayalam script, "en": Latin-script
# English) and whether that choice is doubtful enough for a person to look:
#   1. a file where one language has more than ``dominant`` (80%) of the written characters keeps
#      that language for every sentence (no review);
#   2. otherwise, per sentence:
#      - a sentence without letters ("155.", "0031.") follows the last sentence that has letters
#        (or the next one when it is the first);
#      - a sentence in the file's main language keeps it;
#      - a sentence in the other language keeps its own language when the previous or the next
#        sentence (with letters) is in that language too — a stretch of English;
#      - an isolated switch (both neighbours in another language), or a script that is neither
#        Malayalam nor Latin (Hindi ...), is read in its own language (Malayalam for other
#        scripts) and sent to review.

def script_counts(text: str):
    """(Malayalam characters, Latin letters, other letters / marks)."""
    ml = lat = oth = 0
    for c in text or "":
        if _is_ml(c):
            ml += 1
        elif c.isalpha() and (c.isascii() or unicodedata.name(c, "").startswith("LATIN")):
            lat += 1
        elif c.isalpha() or unicodedata.category(c) in ("Mn", "Mc"):
            oth += 1
    return ml, lat, oth


def _lang_of(counts) -> Optional[str]:
    ml, lat, oth = counts
    if not (ml or lat or oth):
        return None
    return max((("ml", ml), ("en", lat), ("other", oth)), key=lambda kv: kv[1])[0]


def sentence_languages(texts: List[str], dominant: float = 0.8):
    """[(lang "ml" | "en", review reason or None)] per sentence text (rules above)."""
    counts = [script_counts(t) for t in texts]
    tot = [sum(c[i] for c in counts) for i in range(3)]
    total = sum(tot) or 1
    file_lang = _lang_of(tot) or "ml"
    share = max(tot) / total
    if share > dominant and file_lang in ("ml", "en"):
        return [(file_lang, None)] * len(texts)
    raw = [_lang_of(c) for c in counts]
    lettered = [i for i, l in enumerate(raw) if l]
    out = [None] * len(texts)
    for i, lang in enumerate(raw):
        if lang is None:
            continue
        name = {"ml": "Malayalam", "en": "English", "other": "non-Malayalam/English"}[lang]
        if lang == "other":
            out[i] = ("ml", f"{name} script sentence")
            continue
        if lang == file_lang:
            out[i] = (lang, None)
            continue
        k = lettered.index(i)
        nb = [raw[j] for j in (lettered[k - 1] if k else None, lettered[k + 1] if k + 1 < len(lettered) else None)
              if j is not None]
        if lang in nb:
            out[i] = (lang, None)                  # part of a stretch in this language
        else:
            out[i] = (lang, f"isolated {name} sentence in a mixed file "
                            f"({round(100 * tot[0] / total)}% Malayalam / {round(100 * tot[1] / total)}% English)")
    for i in range(len(texts)):                    # digits-only sentences follow the last lettered one
        if out[i] is None:                         #   ... and are reviewed (their language is a guess)
            prev = [j for j in lettered if j < i]
            nxt = [j for j in lettered if j > i]
            src = prev[-1] if prev else (nxt[0] if nxt else None)
            lang = out[src][0] if src is not None else (file_lang if file_lang != "other" else "ml")
            out[i] = (lang, f"sentence without letters in a mixed file (read as "
                            f"{'English' if lang == 'en' else 'Malayalam'}, like the "
                            f"{'previous' if prev else 'next'} sentence)")
    return out
