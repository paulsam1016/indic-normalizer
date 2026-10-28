"""Hindi number readings (Devanagari). First step: English number words written in Devanagari
(वन, टू, थ्री … ट्वेंटी फ़ाइव, फ़ोर हंड्रेड), the twin of the Malayalam-script ``ml.EN_*`` lists.
Native Hindi numbers (एक, दो, पच्चीस, सौ, हज़ार) are not generated yet.

All readings are ``lang="en"``; forms match ``ml`` (english, pairs, english_bare, digit_by_digit,
doubled, english_year, point). Indian grouping: वन लाख सेवेंटी फ़ाइव थाउज़ेंड.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .detect import classify, to_ascii_digits
from .types import Candidate

EN_ONES = ["ज़ीरो", "वन", "टू", "थ्री", "फ़ोर", "फ़ाइव", "सिक्स", "सेवन", "एट", "नाइन"]
EN_TEENS = ["टेन", "इलेवन", "ट्वेल्व", "थर्टीन", "फ़ोर्टीन", "फ़िफ़्टीन", "सिक्सटीन",
            "सेवेंटीन", "एटीन", "नाइनटीन"]
EN_TENS = {2: "ट्वेंटी", 3: "थर्टी", 4: "फ़ोर्टी", 5: "फ़िफ़्टी", 6: "सिक्सटी",
           7: "सेवेंटी", 8: "एटी", 9: "नाइनटी"}
EN_HUNDRED, EN_THOUSAND, EN_LAKH, EN_CRORE = "हंड्रेड", "थाउज़ेंड", "लाख", "करोड़"
EN_OH, EN_DOUBLE, EN_TRIPLE, EN_POINT = "ओ", "डबल", "ट्रिपल", "पॉइंट"
EN_RUPEES, EN_PERCENT = "रुपीज़", "परसेंट"
MAX_NUMBER = 9_99_99_99_999
SPLIT_MIN_DIGITS = 2


def en_words(n: int) -> str:
    if n < 10:
        return EN_ONES[n]
    if n < 20:
        return EN_TEENS[n - 10]
    if n < 100:
        t, u = divmod(n, 10)
        return EN_TENS[t] + ("" if u == 0 else " " + EN_ONES[u])
    for unit, name in ((10_000_000, EN_CRORE), (100_000, EN_LAKH), (1000, EN_THOUSAND), (100, EN_HUNDRED)):
        if n >= unit:
            q, r = divmod(n, unit)
            return f"{en_words(q)} {name}" + ("" if r == 0 else " " + en_words(r))
    return ""


def en_digit_by_digit(s: str) -> str:
    return " ".join(EN_ONES[int(c)] for c in s if c.isdigit())


def en_pairs(n: int) -> Optional[str]:
    """428 -> फ़ोर ट्वेंटी एट, 405 -> फ़ोर ओ फ़ाइव, 2500 -> ट्वेंटी फ़ाइव हंड्रेड (100-9999;
    None when it equals the plain reading)."""
    if not (100 <= n <= 9999):
        return None
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{en_words(hi)} {EN_HUNDRED}" if hi >= 11 and hi % 10 else None
    return f"{en_words(hi)} " + (en_words(lo) if lo >= 10 else f"{EN_OH} " + EN_ONES[lo])


def en_doubled(s: str) -> Optional[str]:
    """55 -> डबल फ़ाइव, 555 -> ट्रिपल फ़ाइव; None when no digit repeats."""
    runs = re.findall(r"(\d)(\1*)", s)
    if not any(len(rest) for _, rest in runs):
        return None
    out = []
    for d, rest in runs:
        k = 1 + len(rest)
        while k > 0:
            step = 3 if k == 3 else min(k, 2)
            out.append({1: "", 2: EN_DOUBLE + " ", 3: EN_TRIPLE + " "}[step] + EN_ONES[int(d)])
            k -= step
    return " ".join(out)


def en_year(n: int) -> str:
    if 2000 <= n <= 2009:
        return en_words(n)
    hi, lo = divmod(n, 100)
    return en_words(hi) + (" " + (en_words(lo) if lo >= 10 else f"{EN_OH} " + EN_ONES[lo]) if lo else " " + EN_HUNDRED)


def _digits(s: str) -> List[Candidate]:
    out = [Candidate(en_digit_by_digit(s), "en", "digit_by_digit")]
    d = en_doubled(s)
    return out + ([Candidate(d, "en", "doubled")] if d else [])


def _int(s: str, year: bool = False) -> List[Candidate]:
    s = s.replace(",", "")
    if (len(s) > 1 and s[0] == "0") or int(s) > MAX_NUMBER:
        return _digits(s)
    n = int(s)
    out = [Candidate(en_year(n), "en", "english_year")] if year and 1000 <= n <= 2999 else []
    out.append(Candidate(en_words(n), "en", "english"))
    p = en_pairs(n)
    if p:
        out.append(Candidate(p, "en", "pairs"))
    if n in (100, 1000):
        out.append(Candidate(EN_HUNDRED if n == 100 else EN_THOUSAND, "en", "english_bare"))
    if 1001 <= n <= 1999:                       # 1600 -> थाउज़ेंड सिक्स हंड्रेड
        out.append(Candidate(en_words(n).replace(f"{EN_ONES[1]} ", "", 1), "en", "english_no_one"))
    if len(s) >= SPLIT_MIN_DIGITS and not re.fullmatch(r"[1-9]0+", s):
        out += _digits(s)
    return out


def _decimal(s: str) -> List[Candidate]:
    whole, frac = s.replace(",", "").split(".")
    w = int(whole or 0)
    head = en_words(w) if w <= MAX_NUMBER else en_digit_by_digit(whole)
    return [Candidate(f"{head} {EN_POINT} {en_digit_by_digit(frac)}", "en", "point")]


def _dedupe(cs: List[Candidate]) -> List[Candidate]:
    seen, out = set(), []
    for c in cs:
        if c.text and c.text not in seen:
            seen.add(c.text)
            out.append(c)
    return out


def candidates(core: str, cls: str = None, style: str = "contracted", latin: bool = False,
               next_word: str = "", alone: bool = False) -> List[Candidate]:
    """Devanagari English readings of one number token core. ``style`` / ``next_word`` are
    accepted for the shared interface and not used yet; ``latin`` gives Latin-script English
    (as for ``ml``). Classes not handled yet fall back to digit by digit."""
    s = to_ascii_digits(core).strip()
    cls = cls or classify(s)
    if latin:
        from . import latin as _latin
        lc = _dedupe(_latin.candidates(s, cls))
        if lc:
            return lc
    try:
        if cls in ("integer", "grouped", "id_code", "year"):
            out = _int(s, year=cls == "year")
        elif cls == "phone":
            plus = "प्लस " if s.startswith("+") else ""
            out = [Candidate(plus + c.text, c.lang, c.form) for c in _digits(re.sub(r"\D", "", s))]
        elif cls == "decimal":
            out = _decimal(s)
        elif cls in ("currency", "percent"):
            amount = re.sub(r"(?i)^(₹|rs\.?|inr)\s*|\s*[₹%]$", "", s).strip()
            n = [c for c in (_decimal(amount) if "." in amount else _int(amount))
                 if c.form not in ("digit_by_digit", "doubled")]
            word = EN_RUPEES if cls == "currency" else EN_PERCENT
            out = [Candidate(f"{c.text} {word}", "en", f"{c.form}+{cls}") for c in n]
            out += [Candidate(c.text, "en", c.form + "+no" + cls) for c in n]
        else:
            out = []
    except (ValueError, KeyError, IndexError):
        out = []
    out = _dedupe(out)
    if not out and re.sub(r"\D", "", s):
        out = _digits(re.sub(r"\D", "", s))
    return out or [Candidate(s, "en", "as_written")]
