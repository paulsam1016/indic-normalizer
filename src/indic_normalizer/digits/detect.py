"""Find and classify number-bearing tokens.

Classes (first match wins, in this order):
    date        17/5/1988, 20-10-20
    time_range  4:00-5:00, 1:30-3:30
    time        6:45, 9:00
    percent     70%, 2.5%
    currency    ₹500, Rs.500, 500₹
    phone       10-13 digits, or +<country code> ...
    range       8-9, 30-40, 19-22
    grouped     1,75,000, 70,000 (comma-grouped integer)
    decimal     2.5, 3.75
    ordinal     21st, 10th, 15-ാം, 2-ാമത്തെ
    year        1900-2099 standalone
    id_code     leading zero (03, 0123) or 8+ digits
    integer     plain digits
    slash       6/14, 1/2
    num_suffix  23-ന്, 250യിൽ, 4-ൽ (numeral + Malayalam suffix)
    alnum       5G, 10K, ഡി3 (letters + digits)
    operator    × between numbers (2950 × 18%)
    other       anything else containing a digit (verbalized piecewise)
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from .types import NumberSpan

CLASSES = ("date", "time_range", "time", "percent", "currency", "phone", "range", "grouped",
           "decimal", "ordinal", "year", "id_code", "integer", "slash", "num_suffix", "alnum",
           "operator", "other")

_D = r"\d"
_ML = "ഀ-ൿ"
_PATTERNS = [
    ("date",       re.compile(rf"^{_D}{{1,2}}([/.\-]){_D}{{1,2}}\1{_D}{{2,4}}$")),
    ("time_range", re.compile(rf"^{_D}{{1,2}}[:.]{_D}{{2}}\s*[-–—]\s*{_D}{{1,2}}[:.]{_D}{{2}}$")),
    ("time",       re.compile(rf"^{_D}{{1,2}}:{_D}{{2}}$")),
    ("percent",    re.compile(rf"^{_D}+(?:[.,]{_D}+)?\s*%$")),
    ("currency",   re.compile(rf"^(?:₹|rs\.?|inr)\s*{_D}[\d,]*(?:\.{_D}+)?$|^{_D}[\d,]*(?:\.{_D}+)?\s*₹$", re.I)),
    ("phone",      re.compile(rf"^\+?{_D}{{10,13}}$|^\+{_D}{{1,3}}[\s-]?{_D}{{6,12}}$")),
    ("range",      re.compile(rf"^{_D}+(?:\.{_D}+)?\s*[-–—~]\s*{_D}+(?:\.{_D}+)?$")),
    ("grouped",    re.compile(rf"^{_D}{{1,3}}(?:(?:,{_D}{{2}})*,{_D}{{3}}|(?:,{_D}{{3}})+)$")),
    ("decimal",    re.compile(rf"^{_D}+\.{_D}+$")),
    ("ordinal",    re.compile(rf"^{_D}+(?:st|nd|rd|th)$|^{_D}+-?ാ(?:ം|മ)[{_ML}]*$", re.I)),
    ("year",       re.compile(r"^(?:19|20)\d{2}$")),
    ("id_code",    re.compile(rf"^0{_D}+$|^{_D}{{8,}}$")),
    ("integer",    re.compile(rf"^{_D}+$")),
    ("slash",      re.compile(rf"^{_D}+/{_D}+$")),
    ("num_suffix", re.compile(rf"^{_D}+(?:,{_D}+)*-?[{_ML}]+$")),
    ("alnum",      re.compile(r"[A-Za-z]|^[^\W\d_]")),
]

_LEAD = "\"'“‘(«[{"
_TRAIL = "\"'”’)»]}.,?!;:…—–-"
_TOKEN = re.compile(r"\S+")


SIGN = r"(?:₹|rs\.?|inr)"                 # currency signs written before / after an amount


def join_currency(text: str) -> str:
    """Glue a currency sign written apart to its amount, so the number is read as money and the
    sign is replaced by the reading: ₹ 70000 -> ₹70000, rs 500 -> rs500, inr 500, 500 ₹ -> 500₹."""
    text = re.sub(rf"(?i)(?<!\S)({SIGN})\s+(?=\d)", r"\1", text or "")
    return re.sub(r"(\d)\s+₹(?!\S)", r"\1₹", text)


def to_ascii_digits(s: str) -> str:
    """Any Unicode decimal digit (Malayalam ൦-൯, ...) -> ASCII."""
    return "".join(str(unicodedata.digit(c)) if unicodedata.category(c) == "Nd" else c for c in s)


def has_digit(s: str) -> bool:
    return any(unicodedata.category(c) == "Nd" for c in s)


def split_edges(tok: str):
    """(lead, core, trail): sentence punctuation around a token, kept out of the number."""
    i, j = 0, len(tok)
    while i < j and tok[i] in _LEAD:
        i += 1
    while j > i and tok[j - 1] in _TRAIL:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


def classify(core: str) -> str:
    """Class of a number-bearing token core (see module docstring)."""
    c = to_ascii_digits(core)
    if c in ("×",):
        return "operator"
    for name, rx in _PATTERNS:
        if rx.search(c):
            return name
    return "other"


def find(text: str) -> List[NumberSpan]:
    """Every whitespace token that contains a digit (plus × between numbers), in order."""
    toks = list(_TOKEN.finditer(text or ""))
    spans = []
    for k, m in enumerate(toks):
        tok = m.group(0)
        lead, core, trail = split_edges(tok)
        if has_digit(core):
            cls = classify(core)
        elif core == "×" and 0 < k < len(toks) - 1 and has_digit(toks[k - 1].group(0)) \
                and has_digit(toks[k + 1].group(0)):
            cls = "operator"
        else:
            continue
        start = m.start() + len(lead)
        spans.append(NumberSpan(text=tok, core=core, start=start, end=start + len(core),
                                token_index=k, cls=cls, lead=lead, trail=trail))
    return spans
