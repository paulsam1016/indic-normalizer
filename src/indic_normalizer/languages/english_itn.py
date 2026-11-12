"""Inverse text normalization (ITN) for English number words in LATIN script.

Spoken -> written numbers, for the Latin-script English in code-mixed Malayalam inference output
(and usable on its own). Malayalam-script English (ട്വന്റി ഫോർ) is read by ``malayalam_itn``;
this module reads ``twenty four`` and writes numbers the same way:

* cardinals: "twenty four" -> 24, "one hundred and five" -> 105, "thousand six hundred" -> 1600,
  "a hundred" -> 100, Indian and Western scales ("three lakh" -> 3,00,000, "two million" ->
  20,00,000; grouped from one lakh up), hyphenated ("twenty-five thousand" -> 25000), the k form
  ("twenty k" -> 20k); a repeated scale starts a new number ("ten thousand fifteen thousand" ->
  10000 15000)
* ordinals: "twenty first" -> 21st, "hundredth" -> 100th
* decimals and percent: "three point one four" -> 3.14, "fifty per cent" -> 50%
* years: "nineteen ninety five" -> 1995, "twenty twenty four" -> 2024, "nineteen oh five" -> 1905
* times: "one o clock" / "five o'clock" -> 1 O’Clock / 5 O’Clock
* digits said one by one: one number when they look like a code (4+ digits, a zero, or "double" /
  "triple": "nine eight four seven" -> 9847, "double oh seven" -> 007), otherwise separate digits
  ("two three" -> 2 3)

Left as words: two adjacent numbers below 100 that do not combine ("twelve thirty", "four twenty
eight": a time, a price or a range); a lone "one" unless a unit / percent / currency word follows
or it ends a range ("one of them", but "one day" -> 1 day); a lone "first" / "second"; a bare
"lakh" / "crore" / "million" with no number after it; plurals ("hundreds of people"). "to" / "too"
/ "for" are never numbers (Latin spells "two" / "four" differently, unlike ടു / ഫോർ), currency
words stay words ("ten dollars" -> 10 dollars), and digits already in the text are left alone.

``spell_small`` (default True here, as in the stt service): English style, where a lone zero..nine
(and a lone small ordinal) stays a word unless a unit follows ("I have five", "the third time").
False reads them like the Malayalam ITN ("I have 5", "the 3rd time"); that is what
``normalize(..., mode="inference")`` and ``itn_numbers`` use unless they are passed
``spell_small=True``. The examples above are with spell_small=False.

Started from the stt service's ``en_itn``; the parser was rewritten to match the Malayalam ITN.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from ..base import indian_grouping

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}
SCALES = {"thousand": 10**3, "lakh": 10**5, "lakhs": 10**5, "lac": 10**5, "lacs": 10**5, "million": 10**6,
          "crore": 10**7, "crores": 10**7, "billion": 10**9, "trillion": 10**12}
BARE_OK = {"thousand"}                           # a scale that means 1 x scale with no count before it
ORD_UNITS = {"zeroth": 0, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
             "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
             "nineteenth": 19}
ORD_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
            "eightieth": 80, "ninetieth": 90}
ORD_SCALES = {"hundredth": 100, "thousandth": 1000, "millionth": 10**6, "billionth": 10**9}
ORD_LONE_WORDS = {"first", "second"}             # alone: words (like ഫസ്റ്റ് / സെക്കൻഡ്)
DIGIT = {k: v for k, v in UNITS.items() if v < 10} | {"oh": 0, "o": 0}
ZERO_LETTERS = ("oh", "o")                       # zero only inside a digit string, never first
REPEAT = {"double": 2, "triple": 3}
K = "k"                                          # twenty k -> 20k
POINT = "point"
CURRENCY = {"dollar", "dollars", "rupee", "rupees", "rs", "euro", "euros", "pound", "pounds"}
PERCENT = {"percent", "percentage"}
CLOCK_ONE = {"o'clock", "o’clock", "oclock"}     # one token
CLOCK_TWO = {"clock", "'clock", "’clock"}        # after a separate "o"
O_CLOCK = "O’Clock"
UNIT_WORDS = {"days", "day", "weeks", "week", "months", "month", "years", "year", "hours", "hour", "minutes",
              "minute", "seconds", "second", "times", "time", "kg", "kilos", "kilograms", "grams", "km",
              "kilometers", "kilometres", "meters", "metres", "liters", "litres", "people", "person",
              "persons", "members", "items", "pages", "am", "pm", "a.m.", "p.m."}
RANGE_WORDS = {"to", "or"}                       # five to six days: context passes across the range
NUMBER_VOCAB = set(UNITS) | set(TENS) | set(SCALES) | {"hundred"} | set(ORD_UNITS) | set(ORD_TENS) | set(ORD_SCALES)

_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)


def fuse_digits(digits: List[str]) -> bool:
    """Digits read one by one are written as one number when they look like a code: 4+ digits
    or a zero. (Same rule as the Malayalam ITN; a run with double / triple always fuses.)"""
    return len(digits) >= 4 or "0" in digits


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


class _Words:
    """Tokens split into lower-cased words. A hyphenated number (twenty-five, twenty-first) is two
    words of one token. Spans may only start at a token's first word and end at its last."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pre, self.post = [], []
        self.w, self.tok, self.first, self.last = [], [], [], []
        for ti, t in enumerate(tokens):
            m = _PUNCT.match(t)
            pre, core, post = m.group(1), m.group(2), m.group(3)
            self.pre.append(pre)
            self.post.append(post)
            parts = core.lower().split("-")
            if len(parts) != 2 or not all(p in NUMBER_VOCAB for p in parts):
                parts = [core.lower()]
            for k, p in enumerate(parts):
                self.w.append(p)
                self.tok.append(ti)
                self.first.append(k == 0)
                self.last.append(k == len(parts) - 1)
        self.n = len(self.w)

    def at(self, i: int) -> str:
        return self.w[i] if 0 <= i < self.n else ""

    def joined(self, i: int) -> bool:
        """Word i follows word i-1 with no punctuation between them."""
        if i <= 0 or i >= self.n:
            return False
        if self.tok[i] == self.tok[i - 1]:
            return True
        return not self.post[self.tok[i - 1]] and not self.pre[self.tok[i]]

    def limit(self, i: int) -> int:
        """End of the punctuation-free stretch starting at word i."""
        j = i + 1
        while j < self.n and self.joined(j):
            j += 1
        return j


def _parse_cardinal(w: List[str]) -> Optional[Tuple[int, int, bool]]:
    """Longest cardinal / ordinal at the start of ``w``: (value, words used, is_ordinal) or None.
    Groups are tracked so a repeated scale or hundred starts a new number instead of adding."""
    total, hundreds, tail = 0, 0, 0                  # tail: tens + units of the current group
    last = None                                      # unit | teen | tens | hundred | scale
    last_scale = 0
    group_start = tail_start = 0                     # where the current group / its tail began
    seen = False
    used = i = 0
    n = len(w)
    while i < n:
        x = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        if x in ("a", "an") and not seen and (nxt == "hundred" or nxt in SCALES):
            tail, last, seen, tail_start = 1, "unit", True, i
            i += 1
            used = i
            continue
        if x == "and" and seen and last in ("hundred", "scale") and (
                nxt in UNITS or nxt in TENS or nxt in ORD_UNITS or nxt in ORD_TENS):
            i += 1
            continue
        if x in UNITS or x in ORD_UNITS:
            v = UNITS.get(x, ORD_UNITS.get(x))
            if last in ("unit", "teen") or (last == "tens" and v >= 10):
                break
            if not tail:
                tail_start = i
            tail += v
            last, seen = ("teen" if v >= 10 else "unit"), True
            i += 1
            used = i
            if x in ORD_UNITS:
                return total + hundreds + tail, used, True
            continue
        if x in TENS or x in ORD_TENS:
            if last in ("unit", "teen", "tens"):
                break
            tail_start = i
            tail += TENS.get(x, ORD_TENS.get(x, 0))
            last, seen = "tens", True
            i += 1
            used = i
            if x in ORD_TENS:
                return total + hundreds + tail, used, True
            continue
        if x in ("hundred", "hundredth"):
            if hundreds or tail >= 100:
                if tail:                                 # two hundred three hundred: new number
                    return total + hundreds, tail_start, False
                break
            if not seen and x == "hundred" and i > 0:
                break
            hundreds, tail = (tail or 1) * 100, 0
            last, seen = "hundred", True
            i += 1
            used = i
            if x == "hundredth":
                return total + hundreds, used, True
            continue
        if x in SCALES or x in ORD_SCALES:
            scale = SCALES.get(x, ORD_SCALES.get(x, 0))
            group = hundreds + tail
            if not group and not seen:
                if x not in BARE_OK and not (nxt in NUMBER_VOCAB and nxt not in SCALES):
                    break                                # a bare lakh / million is a word
            elif not group:
                break
            if last_scale and scale >= last_scale:
                return total, group_start, False         # ten thousand fifteen thousand: 2 numbers
            total += (group or 1) * scale
            hundreds = tail = 0
            last, seen, last_scale = "scale", True, scale
            i += 1
            used = group_start = i
            if x in ORD_SCALES:
                return total, used, True
            continue
        break
    if not seen or used == 0:
        return None
    return total + hundreds + tail, used, False


def _year(w: List[str]) -> Optional[Tuple[int, int]]:
    """'nineteen ninety five' -> 1995; 'twenty twenty four' -> 2024; 'nineteen oh five' -> 1905.
    The first part is fifteen..nineteen or twenty: 'twelve thirty' is a time or a price."""
    if len(w) < 2:
        return None
    first = UNITS[w[0]] if w[0] in UNITS and UNITS[w[0]] >= 15 else (20 if w[0] == "twenty" else None)
    if first is None:
        return None
    rest = w[1:]
    if rest[0] in ZERO_LETTERS and len(rest) > 1 and rest[1] in DIGIT and DIGIT[rest[1]] > 0:
        return first * 100 + DIGIT[rest[1]], 3
    if rest[0] in TENS:
        second = TENS[rest[0]]
        if len(rest) > 1 and rest[1] in UNITS and 0 < UNITS[rest[1]] < 10:
            return first * 100 + second + UNITS[rest[1]], 3
        if first == 20 and second == 20 and len(rest) == 1:
            return None                                  # "twenty twenty" alone is ambiguous
        return first * 100 + second, 2
    if rest[0] in UNITS and 10 <= UNITS[rest[0]] <= 19:
        return first * 100 + UNITS[rest[0]], 2
    return None


def _clock_after(w: List[str]) -> int:
    """Words used by an o'clock right at the start of ``w`` (0 if none)."""
    if w and w[0] in CLOCK_ONE:
        return 1
    if len(w) > 1 and w[0] in ZERO_LETTERS and w[1] in CLOCK_TWO:
        return 2
    return 0


def _digit_string(w: List[str]) -> Optional[Tuple[str, int]]:
    """Digits said one by one at the start of ``w``: (written form, words used) or None. Needs 2+
    digit words or a double / triple; see fuse_digits for one number vs separate digits."""
    out: List[str] = []
    i, rep, repeated = 0, 1, False
    while i < len(w):
        x = w[i]
        if x in REPEAT and i + 1 < len(w) and w[i + 1] in DIGIT:
            rep, repeated = REPEAT[x], True
            i += 1
            continue
        if x in ZERO_LETTERS and _clock_after(w[i:]):
            break                                        # one o clock: that o is not a zero
        if x in DIGIT and (x not in ZERO_LETTERS or out or rep > 1):
            out.append(str(DIGIT[x]) * rep)
            rep = 1
            i += 1
            continue
        break
    if rep > 1:                                          # a trailing double / triple is a word
        i -= 1
    if not (repeated or len(out) >= 2):
        return None
    return ("".join(out) if repeated or fuse_digits(out) else " ".join(out)), i


def itn_tokens(tokens: List[str], *, spell_small: bool = True,
               unit_word: Optional[Callable[[str], bool]] = None) -> List[str]:
    """Rewrite a token list; punctuation glued to a token is kept around the result. Tokens that
    are not English number words (Malayalam, digits, other words) are returned as they are.
    ``unit_word(token)`` may add unit words of another language (the Malayalam ITN passes
    രൂപ / ദിവസം / മിനിറ്റ് ...), so "five മിനിറ്റ്" reads as 5 മിനിറ്റ്."""
    W = _Words(tokens)
    w = W.w

    def unit_at(k: int) -> bool:
        """Word k is a unit / currency / percent / o'clock / k word right after a number."""
        if k >= W.n or not W.joined(k) or not W.first[k]:
            return False
        x = w[k]
        return (x in CURRENCY or x in PERCENT or x == "per" or x in UNIT_WORDS or x == K
                or bool(_clock_after(w[k:W.limit(k)]))
                or bool(unit_word and unit_word(tokens[W.tok[k]].strip())))

    def context(k: int) -> bool:
        """A small number ending before word k is in numeric context: a unit follows, or it is the
        first end of a range whose other end has one (five to six days)."""
        if unit_at(k):
            return True
        if k + 1 < W.n and w[k] in RANGE_WORDS and W.joined(k) and W.joined(k + 1):
            p = _parse_cardinal(w[k + 1:W.limit(k + 1)])
            return bool(p) and unit_at(k + 1 + p[1])
        return False

    out: List[str] = []
    prev_number_end = -1                                 # word index after the last number written
    i = 0
    while i < W.n:
        if not W.first[i]:                               # inside a token already passed on
            i += 1
            continue
        lim = W.limit(i)
        span = w[i:lim]
        text, used = None, 0
        ds = _digit_string(span)
        if ds:
            text, used = ds
        if text is None and w[i] in UNITS | TENS:
            y = _year(span[:3])
            if y:
                text, used = str(y[0]), y[1]
        is_plain = False
        if text is None:
            parsed = _parse_cardinal(span)
            if parsed:
                value, used, ordinal = parsed
                j = i + used
                if not ordinal and j + 1 < lim and w[j] == POINT and w[j + 1] in DIGIT:
                    k, ds2 = j + 1, []                   # decimals: <n> point d d d
                    while k < lim and w[k] in DIGIT and w[k] != "o":
                        ds2.append(str(DIGIT[w[k]]))
                        k += 1
                    text, used = f"{value}.{''.join(ds2)}", k - i
                elif ordinal:
                    text = f"{value}{_ordinal_suffix(value)}"
                else:
                    text, is_plain = indian_grouping(value), True
                j = i + used
                second = _parse_cardinal(w[j:lim]) if is_plain and value < 100 and j < lim else None
                if second and second[0] < 100 and not second[2]:
                    # two numbers below 100 side by side that do not combine (twelve thirty, four
                    # twenty eight): a time, a price or a range; ambiguous, so both stay words
                    end_tok = W.tok[j + second[1] - 1]
                    out.extend(tokens[W.tok[i]:end_tok + 1])
                    i = j + second[1]
                    while i < W.n and not W.first[i]:
                        i += 1
                    prev_number_end = -1
                    continue
                lone = used == 1 or (used == 2 and w[i] in ("a", "an"))
                after_range = (prev_number_end == i - 1 and i >= 1 and w[i - 1] in RANGE_WORDS
                               and W.joined(i))
                if lone and value < 10:
                    if ordinal:
                        keep_word = spell_small or w[i] in ORD_LONE_WORDS
                    else:
                        keep_word = (spell_small or w[i] == "one") and not (context(j) or after_range)
                    if keep_word:
                        text, used = None, 0             # "one of them", "the first time": words
        if text is not None and used:
            end = i + used                               # end must close a token
            if not W.last[end - 1]:
                text = None
        if text is not None and used:
            end = i + used
            if end < lim:
                x, x2 = w[end], (w[end + 1] if end + 1 < lim else "")
                if is_plain and x == K and W.last[end]:
                    text, end = text.replace(",", "") + "k", end + 1
                elif " " not in text and "%" not in text and (x in PERCENT or (x == "per" and x2 == "cent")):
                    text, end = text + "%", end + (1 if x in PERCENT else 2)
                elif is_plain or text.isdigit():
                    c = _clock_after(w[end:lim])
                    if c:
                        text, end = f"{text} {O_CLOCK}", end + c
            out.append(W.pre[W.tok[i]] + text + W.post[W.tok[end - 1]])
            prev_number_end = end
            i = end
            continue
        out.append(tokens[W.tok[i]])
        i += 1
        while i < W.n and not W.first[i]:
            i += 1
        if prev_number_end >= 0 and not (w[i - 1] in RANGE_WORDS and prev_number_end == i - 1):
            prev_number_end = -1
    return out


def itn(text: str, *, spell_small: bool = True,
        unit_word: Optional[Callable[[str], bool]] = None) -> str:
    """English number words (Latin script) in ``text`` -> digits; everything else as is."""
    return " ".join(itn_tokens((text or "").split(), spell_small=spell_small, unit_word=unit_word))
