"""Malayalam spoken forms of numbers: number words and candidate readings per class.

Two vocabularies:
    native   Malayalam number words  (ഇരുപത്തഞ്ച്, അമ്പതിനായിരം, രണ്ട് ലക്ഷം)
    english  English number words in Malayalam script, as code-mixed speech says them
             (ട്വന്റി ഫൈവ്, ഫിഫ്റ്റി തൗസന്റ്, ടു ലാക്). Spellings are the most frequent
             ones in base-model output on the eval set.
Two native compound styles (only differ before a vowel-initial part):
    contracted  ഇരുപത്തഞ്ച്, ഇരുപത്തേഴ്, ഇരുപത്തയ്യായിരം, നൂറ്റമ്പത്   (colloquial; default)
    glide       ഇരുപത്തിയഞ്ച്, ഇരുപത്തിയേഴ്, ഇരുപത്തിയയ്യായിരം, നൂറ്റിയമ്പത്
"""

from __future__ import annotations

import re
from typing import List, Optional

from .detect import classify, to_ascii_digits
from .types import Candidate

VIRAMA = "്"
STYLES = ("contracted", "glide")

ONES = ["പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒമ്പത്"]
TEENS = ["പത്ത്", "പതിനൊന്ന്", "പന്ത്രണ്ട്", "പതിമൂന്ന്", "പതിനാല്", "പതിനഞ്ച്", "പതിനാറ്",
         "പതിനേഴ്", "പതിനെട്ട്", "പത്തൊമ്പത്"]
TENS = {2: "ഇരുപത്", 3: "മുപ്പത്", 4: "നാൽപ്പത്", 5: "അമ്പത്", 6: "അറുപത്", 7: "എഴുപത്",
        8: "എൺപത്", 9: "തൊണ്ണൂറ്"}
HUNDS = {1: "നൂറ്", 2: "ഇരുനൂറ്", 3: "മുന്നൂറ്", 4: "നാനൂറ്", 5: "അഞ്ഞൂറ്", 6: "അറുനൂറ്",
         7: "എഴുനൂറ്", 8: "എണ്ണൂറ്", 9: "തൊള്ളായിരം"}
THOUS_U = {1: "ആയിരം", 2: "രണ്ടായിരം", 3: "മൂവായിരം", 4: "നാലായിരം", 5: "അയ്യായിരം",
           6: "ആറായിരം", 7: "ഏഴായിരം", 8: "എണ്ണായിരം", 9: "ഒമ്പതിനായിരം"}
LAKH, LAKH_C, CRORE = "ലക്ഷം", "ലക്ഷത്തി", "കോടി"

# independent vowel -> dependent sign (അ has none)
V_SIGN = {"അ": "", "ആ": "ാ", "ഇ": "ി", "ഈ": "ീ", "ഉ": "ു", "ഊ": "ൂ", "ഋ": "ൃ",
          "എ": "െ", "ഏ": "േ", "ഐ": "ൈ", "ഒ": "ൊ", "ഓ": "ോ", "ഔ": "ൗ"}
SIGNS = set("ാിീുൂൃെേൈൊോൌൗ")

EN_ONES = ["സീറോ", "വൺ", "ടു", "ത്രീ", "ഫോർ", "ഫൈവ്", "സിക്സ്", "സെവൻ", "എയ്റ്റ്", "നൈൻ"]
EN_TEENS = ["ടെൻ", "ഇലവൻ", "ട്വൽവ്", "തേർട്ടീൻ", "ഫോർട്ടീൻ", "ഫിഫ്റ്റീൻ", "സിക്സ്റ്റീൻ",
            "സെവന്റീൻ", "എയ്റ്റീൻ", "നൈന്റീൻ"]
EN_TENS = {2: "ട്വന്റി", 3: "തേർട്ടി", 4: "ഫോർട്ടി", 5: "ഫിഫ്റ്റി", 6: "സിക്സ്റ്റി",
           7: "സെവന്റി", 8: "എയ്റ്റി", 9: "നൈന്റി"}
# Spellings: തൗസന്റ് (base-model majority), ലാക് (as the reference transcripts write it).
EN_HUNDRED, EN_THOUSAND, EN_LAKH, EN_CRORE = "ഹൺഡ്രഡ്", "തൗസന്റ്", "ലാക്", "ക്രോർ"
# English ordinals as Malayalam speech writes them (base-model output: ട്വന്റിത്തിലോട്ട്,
# ട്വന്റി സെവൻത്തിലോട്ട്, ഫോർട്ടീൻത്തിലോട്ടും, സിക്സ്ത്തിലോട്ടും): -th is ത്ത്.
EN_ORD = {1: "ഫസ്റ്റ്", 2: "സെക്കൻഡ്", 3: "തേർഡ്", 5: "ഫിഫ്ത്ത്", 8: "എയ്റ്റ്ത്ത്", 9: "നൈൻത്ത്",
          12: "ട്വൽത്ത്"}
DATE_MAX = 31                       # numbers <= this also get an English ordinal reading (dates)
LETTERS = {"a": "എ", "b": "ബി", "c": "സി", "d": "ഡി", "e": "ഇ", "f": "എഫ്", "g": "ജി",
           "h": "എച്ച്", "i": "ഐ", "j": "ജെ", "k": "കെ", "l": "എൽ", "m": "എം", "n": "എൻ",
           "o": "ഒ", "p": "പി", "q": "ക്യു", "r": "ആർ", "s": "എസ്", "t": "ടി", "u": "യു",
           "v": "വി", "w": "ഡബ്ല്യു", "x": "എക്സ്", "y": "വൈ", "z": "സെഡ്"}

MAX_NUMBER = 9_99_99_99_999          # up to 999 crore (10 digits); larger is read digit by digit


# --------------------------------------------------------------------------- native words

def combining(w: str) -> str:
    """Form that takes a following part: നൂറ് -> നൂറ്റി, ആയിരം -> ആയിരത്തി, ഇരുപത് -> ഇരുപത്തി."""
    if w.endswith("ം"):
        return w[:-1] + "ത്തി"
    if w.endswith("റ" + VIRAMA):
        return w + "റി"
    if w.endswith("ത" + VIRAMA):
        return w + "തി"
    return w


def join(stem: str, rest: str, style: str = "contracted") -> str:
    """Join a combining stem (ending in ി) to the next part, with sandhi before a vowel.
    style "spaced" (candidate-only) keeps the parts as separate words: ആയിരത്തി എഴുനൂറ്."""
    if style == "spaced" and stem.endswith("ി"):
        return stem + " " + rest
    if rest and rest[0] in V_SIGN and stem.endswith("ി"):
        if style == "glide":
            return stem + "യ" + V_SIGN[rest[0]] + rest[1:]
        return stem[:-1] + V_SIGN[rest[0]] + rest[1:]
    return stem + rest


def _below100(n: int, style: str) -> str:
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    t, u = divmod(n, 10)
    return TENS[t] if u == 0 else join(combining(TENS[t]), ONES[u], style)


def _below1000(n: int, style: str) -> str:
    if n < 100:
        return _below100(n, style)
    h, r = divmod(n, 100)
    return HUNDS[h] if r == 0 else join(combining(HUNDS[h]), _below100(r, style), style)


def thousands_word(x: int, style: str = "contracted") -> str:
    """x thousand for x in 1-99 as one word: 5 അയ്യായിരം, 12 പന്ത്രണ്ടായിരം, 20 ഇരുപതിനായിരം,
    25 ഇരുപത്തയ്യായിരം (glide: ഇരുപത്തിയയ്യായിരം)."""
    if x < 10:
        return THOUS_U[x]
    if x == 10:
        return "പതിനായിരം"
    if x < 20:
        return {15: "പതിനയ്യായിരം", 19: "പത്തൊമ്പതിനായിരം"}.get(x, TEENS[x - 10][:-1] + "ായിരം")
    t, u = divmod(x, 10)
    if u == 0:
        w = TENS[t]
        return w[:-1] + ("ിനായിരം" if w.endswith("ത" + VIRAMA) else "ായിരം")
    return join(combining(TENS[t]), "ഒന്നായിരം" if u == 1 else THOUS_U[u], style)


def _below_lakh(n: int, style: str) -> str:
    if n < 1000:
        return _below1000(n, style)
    th, r = divmod(n, 1000)
    w = thousands_word(th, style)
    return w if r == 0 else join(combining(w), _below1000(r, style), style)


def words(n: int, style: str = "contracted") -> str:
    """0 - 9,99,99,99,999 in native Malayalam words. Up to 99,999 is one fused word; lakhs and
    crores are separate words after a count (രണ്ട് ലക്ഷത്തി അമ്പതിനായിരം, ഒരു കോടി)."""
    if n < 0 or n > MAX_NUMBER:
        raise ValueError(f"{n} out of range for words()")
    if n < 100_000:
        return _below_lakh(n, style)
    if n < 10_000_000:
        lk, r = divmod(n, 100_000)
        count = "ഒരു" if lk == 1 else _below100(lk, style)
        return f"{count} {LAKH}" if r == 0 else f"{count} {LAKH_C} {words(r, style)}"
    cr, r = divmod(n, 10_000_000)
    count = "ഒരു" if cr == 1 else _below1000(cr, style)        # up to 999 crore
    return f"{count} {CRORE}" if r == 0 else f"{count} {CRORE} {words(r, style)}"


def digit_by_digit(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def half(n: int, style: str = "contracted") -> str:
    """n and a half: 0 അര, 1 ഒന്നര, 2 രണ്ടര, 10 പത്തര, 3 lakh -> മൂന്നര ലക്ഷം is n=3."""
    if n == 0:
        return "അര"
    w = words(n, style)
    return w[:-1] + "ര" if w.endswith(VIRAMA) else w + " അര"


CHILLU_BASE = {"ൺ": "ണ", "ൻ": "ന", "ർ": "റ", "ൽ": "ല", "ൾ": "ള", "ൿ": "ക"}


def attach(word: str, suffix: str) -> str:
    """Glue a Malayalam suffix to (the last word of) a number reading, with sandhi:
    ഇരുപത്തിമൂന്ന് + ന് -> ഇരുപത്തിമൂന്നിന്, നാല് + ൽ -> നാലിൽ, പതിനഞ്ച് + ാം -> പതിനഞ്ചാം,
    ഇരുനൂറ്റമ്പത് + യിൽ -> ഇരുനൂറ്റമ്പതിൽ, ഇരുപത്തേഴ് + ഓ -> ഇരുപത്തേഴോ, ആയിരം + ന് -> ആയിരത്തിന്,
    and for English words: സെവൻ + ഓ -> സെവനോ, ഫോർ + ൽ -> ഫോറിൽ, ട്വന്റി + ിൽ -> ട്വന്റിയിൽ."""
    suffix = suffix.lstrip("-")
    if not suffix:
        return word
    head, _, last = word.rpartition(" ")
    pre = head + " " if head else ""
    if suffix[0] == "യ" and len(suffix) > 1 and suffix[1] in SIGNS:   # glide written out
        suffix = suffix[1:]
    if suffix[0] in V_SIGN:                                              # independent vowel
        suffix = V_SIGN[suffix[0]] + suffix[1:]
        if not suffix:
            return word
    vowel_suffix = suffix[0] in SIGNS
    if last.endswith(VIRAMA):                       # നാല് / ഫൈവ്
        stem = last[:-1]
        if suffix.startswith("ക്ക"):                 # dative: അമ്പത് + ക്ക് -> അമ്പതിന്
            return pre + stem + "ിന" + suffix[3:]
        return pre + stem + (suffix if vowel_suffix else "ി" + suffix)
    if last.endswith("ം"):                          # ആയിരം / ലക്ഷം / പതിനഞ്ചാം
        if vowel_suffix:
            return pre + last[:-1] + "മ" + suffix
        if last.endswith("ാം"):                     # ordinal: പന്ത്രണ്ടാം -> പന്ത്രണ്ടാമത്തി...
            return pre + last[:-1] + "മത്തി" + suffix
        return pre + last[:-1] + "ത്തി" + suffix
    if last[-1] in CHILLU_BASE:                     # സെവൻ / ഫോർ / ലാക്ക്-less chillu endings
        stem = last[:-1] + CHILLU_BASE[last[-1]]
        if suffix.startswith("ക്ക"):
            return pre + stem + "ിന" + suffix[3:]
        return pre + stem + (suffix if vowel_suffix else "ി" + suffix)
    if last[-1] in SIGNS:                           # ട്വന്റി / ത്രീ
        if vowel_suffix:
            return pre + last + "യ" + suffix        # ട്വന്റി + ിൽ -> ട്വന്റിയിൽ
        if suffix.startswith("ന്റെ"):
            return pre + last + "യുടെ" + suffix[4:]   # genitive: ത്രീയുടെ
        if suffix[0] == "ന" and len(suffix) > 1 and (suffix[1] in SIGNS or suffix[1] == VIRAMA):
            return pre + last + "ക്ക" + suffix[1:]   # dative: ത്രീ + ന് -> ത്രീക്ക്
        if suffix[0] in CHILLU_BASE or suffix.startswith("ലോ"):
            return pre + last + "യി" + suffix        # ട്വന്റി + ൽ -> ട്വന്റിയിൽ, + ലോട്ട് -> ട്വന്റിയിലോട്ട്
        return pre + last + suffix
    if "\u0d15" <= last[-1] <= "\u0d39":            # inherent -a ending: രണ്ടര, മൂന്നര
        if vowel_suffix:
            return pre + last + "യ" + suffix        # രണ്ടര + ഓ -> രണ്ടരയോ
        if suffix.startswith("ന്റെ"):
            return pre + last + "യുടെ" + suffix[4:]  # genitive
        if suffix.startswith("ക്ക"):
            return pre + last + "യ്" + suffix        # മൂന്നരയ്ക്ക്
        if suffix[0] == "ന" and len(suffix) > 1 and (suffix[1] in SIGNS or suffix[1] == VIRAMA):
            return pre + last + "യ്ക്ക" + suffix[1:]  # dative -ന്: എട്ടര + നുള്ള -> എട്ടരയ്ക്കുള്ള
        return pre + last + "യി" + suffix          # മൂന്നര + ൽ -> മൂന്നരയിൽ
    return pre + last + suffix


def ordinal(n: int, style: str = "contracted") -> str:
    return attach(words(n, style), "ാം")


# -------------------------------------------------------------------------- english words

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
    """English read in pairs, as for codes / room / bus / flight numbers: 428 -> ഫോർ ട്വന്റി എയ്റ്റ്,
    405 -> ഫോർ ഓ ഫൈവ്, 1234 -> ട്വൽവ് തേർട്ടി ഫോർ, 2500 -> ട്വന്റി ഫൈവ് ഹൺഡ്രഡ്. For 100-9999;
    None when it would equal the plain reading (400 -> ഫോർ ഹൺഡ്രഡ്, 2000)."""
    if not (100 <= n <= 9999):
        return None
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{en_words(hi)} {EN_HUNDRED}" if hi >= 11 and hi % 10 else None
    return f"{en_words(hi)} " + (en_words(lo) if lo >= 10 else "ഓ " + EN_ONES[lo])


def en_doubled(s: str) -> Optional[str]:
    """English digits with repeats grouped, as numbers are read out: 55 -> ഡബിൾ ഫൈവ്, 555 ->
    ട്രിപ്പിൾ ഫൈവ്, 9988 -> ഡബിൾ നൈൻ ഡബിൾ എയ്റ്റ്, 98100 -> നൈൻ എയ്റ്റ് വൺ ഡബിൾ സീറോ.
    None when no digit repeats."""
    runs = re.findall(r"(\d)(\1*)", s)
    if not any(len(rest) for _, rest in runs):
        return None
    out = []
    for d, rest in runs:
        k = 1 + len(rest)
        while k > 0:                          # 4 fives -> ഡബിൾ ഫൈവ് ഡബിൾ ഫൈവ്
            step = 3 if k == 3 else min(k, 2)
            out.append({1: "", 2: "ഡബിൾ ", 3: "ട്രിപ്പിൾ "}[step] + EN_ONES[int(d)])
            k -= step
    return " ".join(out)


def en_year(n: int) -> str:
    if 2000 <= n <= 2009:
        return en_words(n)
    hi, lo = divmod(n, 100)
    return en_words(hi) + (" " + (en_words(lo) if lo >= 10 else "ഓ " + EN_ONES[lo]) if lo else " " + EN_HUNDRED)


def en_ordinal(n: int) -> str:
    if n in EN_ORD:
        return EN_ORD[n]
    if n > 100 and n % 100:                 # 112th -> വൺ ഹൺഡ്രഡ് ട്വൽത്ത്
        return en_words(n - n % 100) + " " + en_ordinal(n % 100)
    if n > 20 and n % 10 in EN_ORD and n % 100 not in range(11, 20):
        return en_words(n - n % 10) + " " + EN_ORD[n % 10]
    return en_words(n) + "ത്ത്"        # ട്വന്റിത്ത്, സെവൻത്ത്, ഫോർട്ടീൻത്ത്, സിക്സ്ത്ത്


# ------------------------------------------------------------------------------ candidates

def _dedupe(cands: List[Candidate]) -> List[Candidate]:
    seen, out = set(), []
    for c in cands:
        t = re.sub(r"\s+", " ", c.text).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(Candidate(t, c.lang, c.form))
    return out


def drop_thousands(n: int, style: str = "contracted") -> Optional[str]:
    """Colloquial reading that leaves out "thousand": 38500 -> മുപ്പത്തെട്ട് അഞ്ഞൂറ്,
    13600 -> പതിമൂന്ന് അറുനൂറ്, 2500 -> രണ്ട് അഞ്ഞൂറ്. Only for 2,000-99,999 with a non-zero
    hundreds part (n % 1000 >= 100); None otherwise."""
    if not (2000 <= n < 100_000):
        return None
    th, r = divmod(n, 1000)
    if r < 100:
        return None
    return f"{_below100(th, style)} {_below1000(r, style)}"


def drop_thousands_e(n: int, style: str = "contracted") -> Optional[str]:
    """drop_thousands with the "-ഏ" link: 24500 -> ഇരുപത്തിനാലേ അഞ്ഞൂറ്, 39500 -> മുപ്പത്തൊമ്പതേ
    അഞ്ഞൂറ്. A spelling of the same reading (family drop_thousand)."""
    if not (2000 <= n < 100_000) or n % 1000 < 100:
        return None
    th, r = divmod(n, 1000)
    w = _below100(th, style)
    if not w.endswith(VIRAMA):
        return None
    return f"{w[:-1]}േ {_below1000(r, style)}"


def lakh_drop_thousand(n: int, style: str = "contracted") -> Optional[str]:
    """Lakhs + a whole number of thousands without "ആയിരം": 325000 -> മൂന്ന് ലക്ഷത്തി ഇരുപത്തഞ്ച്,
    150000 -> ഒരു ലക്ഷത്തി അമ്പത്."""
    if not (100_000 <= n < 10_000_000):
        return None
    lk, r = divmod(n, 100_000)
    if r == 0 or r % 1000:
        return None
    count = "ഒരു" if lk == 1 else _below100(lk, style)
    return f"{count} {LAKH_C} {_below100(r // 1000, style)}"


def en_drop_thousands(n: int) -> Optional[str]:
    """English form of drop_thousands: 38500 -> തേർട്ടി എയ്റ്റ് ഫൈവ് ഹൺഡ്രഡ്."""
    if not (2000 <= n < 100_000):
        return None
    th, r = divmod(n, 1000)
    if r < 100:
        return None
    return f"{en_words(th)} {en_words(r)}"


# Other spoken forms of the same number (seen in eval_frozen and base-model hypotheses):
# X8 thousand as ...എട്ടായിരം besides ...എണ്ണായിരം (98000 തൊണ്ണൂറ്റിയെട്ടായിരം, 48500
# നാൽപ്പത്തിയെട്ടായിരത്തിയഞ്ഞൂറ്) and X5 thousand as ...അഞ്ചായിരം besides ...അയ്യായിരം (75000
# എഴുപത്തിയഞ്ചായിരം). These are different words, so the audio can tell them apart. Pure spelling
# variants (നാല്പത്, എമ്പത്, ഇരുന്നൂറ്) are not added: the spelling policy decides those.
# All substitutions applied together give one "<form>_alt" candidate per native form, in the same
# reading family as that form (never a review reason).
ALT_SUBS = [("ണ്ണായിര", "ട്ടായിര"), ("യ്യായിര", "ഞ്ചായിര")]
ALT_FORMS = ("contracted", "glide", "spaced", "drop_thousand", "drop_thousand_spaced")


def alt_spelling(text: str) -> str:
    for a, b in ALT_SUBS:
        text = text.replace(a, b)
    return text


def with_alts(cands: List[Candidate]) -> List[Candidate]:
    """Append the colloquial variant (ALT_SUBS) of each native candidate that has one."""
    extra = []
    for c in cands:
        if c.lang == "ml" and c.form in ALT_FORMS:
            a = alt_spelling(c.text)
            if a != c.text:
                extra.append(Candidate(a, "ml", c.form + "_alt"))
    return cands + extra


def lakh_short(n: int, style: str = "contracted"):
    """A 3-digit amount written in thousands-shorthand for X lakh YY thousand (155 for 1,55,000 —
    fees, packages): [(text, lang, form)] with ഒന്നേ അമ്പത്തഞ്ച് (X-ഏ YY), വൺ ലാക് ഫിഫ്റ്റി ഫൈവ്,
    ഒരു ലക്ഷത്തി അമ്പത്തഞ്ച്. Empty unless 101-999 and not a whole hundred."""
    if not (101 <= n <= 999) or n % 100 == 0:
        return []
    x, yy = divmod(n, 100)
    xe = ONES[x][:-1] + "േ"                          # ഒന്ന് -> ഒന്നേ
    return [(f"{xe} {words(yy, style)}", "ml", "lakh_e"),
            (f"{EN_ONES[x]} {EN_LAKH} {en_words(yy)}", "en", "lakh_short"),
            (f"{'ഒരു' if x == 1 else ONES[x]} {LAKH_C} {words(yy, style)}", "ml", "lakh_short")]


def _num(n: int, style: str) -> List[Candidate]:
    """Readings of an integer, default first: native in ``style``, native in the other style,
    English, native with the compound parts spaced (ആയിരത്തി എഴുനൂറ്), native without
    "thousand" (മുപ്പത്തെട്ട് അഞ്ഞൂറ് / മുപ്പത്തി എട്ട് അഞ്ഞൂറ്), English without "thousand"."""
    other = "glide" if style == "contracted" else "contracted"
    # English sits third so a small max_candidates never cuts it (~1/3 of spoken numbers).
    out = [Candidate(words(n, style), "ml", style), Candidate(words(n, other), "ml", other),
           Candidate(en_words(n), "en", "english"), Candidate(words(n, "spaced"), "ml", "spaced")]
    ls = lakh_short(n, style)
    out += [Candidate(t, lang, form) for t, lang, form in ls]                # 155 -> ഒന്നേ അമ്പത്തഞ്ച്
    d = drop_thousands(n, style)
    if d:
        out.append(Candidate(d, "ml", "drop_thousand"))
        out.append(Candidate(drop_thousands(n, "spaced"), "ml", "drop_thousand_spaced"))
        e = drop_thousands_e(n, style)
        if e:
            out.append(Candidate(e, "ml", "drop_thousand_e"))         # ഇരുപത്തിനാലേ അഞ്ഞൂറ്
    d = lakh_drop_thousand(n, style)
    if d:
        out.append(Candidate(d, "ml", "drop_thousand"))               # മൂന്ന് ലക്ഷത്തി ഇരുപത്തഞ്ച്
    e = en_drop_thousands(n)
    if e:
        out.append(Candidate(e, "en", "drop_thousand"))
    e = en_pairs(n)
    if e:
        out.append(Candidate(e, "en", "pairs"))
    if n in (100, 1000):                      # without "one": ഹൺഡ്രഡ് / തൗസന്റ്
        out.append(Candidate(EN_HUNDRED if n == 100 else EN_THOUSAND, "en", "english_bare"))
    if 1000 < n < 2000:                       # 1600 -> തൗസന്റ് സിക്സ് ഹൺഡ്രഡ് (no "one")
        out.append(Candidate(en_words(n).replace(f"{EN_ONES[1]} ", "", 1), "en", "english_no_one"))
    if 1000 <= n < 100_000 and n % 1000 == 0:               # 15000 -> ഫിഫ്റ്റീൻ കെ
        out.append(Candidate(f"{en_words(n // 1000)} കെ", "en", "english_k"))
    if 200_000 <= n < 10_000_000 and n % 100_000 == 0:     # 800000 -> എയ്റ്റ് ലാക്സ്
        out.append(Candidate(f"{en_words(n // 100_000)} ലാക്സ്", "en", "english_lakhs"))
    tail = []
    if ls:                                    # other spellings of X-ഏ YY (same reading)
        x, yy = divmod(n, 100)
        xe = ONES[x][:-1] + "േ"
        tail = [Candidate(f"{xe} {words(yy, other)}", "ml", "lakh_e_" + other),
                Candidate(f"{xe} {words(yy, 'spaced')}", "ml", "lakh_e_spaced")]
    return with_alts(out) + tail


SPLIT_MIN_DIGITS = 2         # integers with this many digits also get digit-by-digit readings
                             # (94 -> ഒമ്പത് നാല് / നൈൻ ഫോർ; phones are digit-by-digit FIRST)


def _doubled(s: str) -> List[Candidate]:
    e = en_doubled(s)
    return [Candidate(e, "en", "doubled")] if e else []


def _int_or_ids(s: str, style: str) -> List[Candidate]:
    s = s.replace(",", "")
    if (len(s) > 1 and s[0] == "0") or int(s) > MAX_NUMBER:
        return [Candidate(digit_by_digit(s), "ml", "digit_by_digit"),
                Candidate(en_digit_by_digit(s), "en", "digit_by_digit")] + _doubled(s)
    out = _num(int(s), style)
    if len(s) >= 2 and len(set(s)) == 1:      # 55 -> ഡബിൾ ഫൈവ്, 999 -> ട്രിപ്പിൾ നൈൻ
        k = next((i for i, c in enumerate(out) if c.form.endswith("_alt")), len(out))
        out[k:k] = _doubled(s)
    if len(s) >= SPLIT_MIN_DIGITS and not re.fullmatch(r"[1-9]0+", s):   # not 10 / 100 / 1000
        # split digit by digit too (730 -> ഏഴ് മൂന്ന് പൂജ്യം / സെവൻ ത്രീ സീറോ: codes, room / bus /
        # flight numbers), placed before the colloquial _alt spellings so max_candidates keeps them
        k = next((i for i, c in enumerate(out) if c.form.endswith("_alt")), len(out))
        out[k:k] = [Candidate(digit_by_digit(s), "ml", "digit_by_digit"),
                    Candidate(en_digit_by_digit(s), "en", "digit_by_digit")] + _doubled(s)
    return out


def _decimal(s: str, style: str) -> List[Candidate]:
    whole, frac = s.split(".")
    whole = whole.replace(",", "")
    w = int(whole or 0)
    if w > MAX_NUMBER:                       # too long for words: digits, point, digits
        return [Candidate(f"{digit_by_digit(whole)} പോയിന്റ് {digit_by_digit(frac)}", "ml", "point"),
                Candidate(f"{en_digit_by_digit(whole)} പോയിന്റ് {en_digit_by_digit(frac)}", "en", "point")]
    out = []
    if frac.rstrip("0") == "5":
        out.append(Candidate(half(w, style), "ml", "half"))
    if frac.rstrip("0") in ("25", "75"):
        q = frac.startswith("25")
        if w == 0:
            out.append(Candidate("കാൽ" if q else "മുക്കാൽ", "ml", "quarter"))
        else:
            word = words(w, style)
            if word.endswith(VIRAMA):
                out.append(Candidate(word[:-1] + ("േകാൽ" if q else "േമുക്കാൽ"), "ml", "quarter"))
    out.append(Candidate(f"{words(w, style)} പോയിന്റ് {digit_by_digit(frac)}", "ml", "point"))
    out.append(Candidate(f"{en_words(w)} പോയിന്റ് {en_digit_by_digit(frac)}", "en", "point"))
    if len(frac) in (3, 5) and w and int(whole + frac) <= MAX_NUMBER:
        # a dot used as a group separator: 1.70000 = 1,70,000, 2.500 = 2,500
        g = int(whole + frac)
        out += [Candidate(words(g, style), "ml", "grouped_dot"), Candidate(en_words(g), "en", "grouped_dot")]
    if len(frac) == 2 and 1 <= w <= 24 and int(frac) < 60 and whole[:1] != "0":
        # 4.30 / 12.30 may be a time written with a dot: also the time readings (not the default)
        out += [c for c in _time(f"{w}:{frac}", style) if c.form != "bare"]
    return out


def _number_any(s: str, style: str) -> List[Candidate]:
    s = s.replace(",", "")
    return _decimal(s, style) if "." in s else _int_or_ids(s, style)


def _time(s: str, style: str) -> List[Candidate]:
    h, mm = (int(x) for x in re.split(r"[:.]", s))
    H, EH = words(h, style), en_words(h)
    if mm == 0:
        return [Candidate(f"{'ഒരു' if h == 1 else H} മണി", "ml", "oclock"), Candidate(H, "ml", "bare"),
                Candidate(f"{EH} ഓ ക്ലോക്ക്", "en", "oclock")]
    M, EM = words(mm, style), (en_words(mm) if mm >= 10 else "ഓ " + EN_ONES[mm])
    out = []
    if mm == 30:
        out.append(Candidate(half(h, style), "ml", "half"))
    if mm in (15, 45) and H.endswith(VIRAMA):
        out.append(Candidate(H[:-1] + ("േകാൽ" if mm == 15 else "േമുക്കാൽ"), "ml", "quarter"))
    out += [Candidate(f"{H} {M}", "ml", "hh_mm"), Candidate(f"{H} മണി {M}", "ml", "hh_mani_mm"),
            Candidate(f"{EH} {EM}", "en", "hh_mm")]
    return out


def _pair(a: List[Candidate], b: List[Candidate], forms) -> List[Candidate]:
    """Combine two part readings (same language) into range readings."""
    out = []
    for lang in ("ml", "en"):
        A = next((c for c in a if c.lang == lang), None)
        B = next((c for c in b if c.lang == lang), None)
        if A and B:
            for fmt, form in forms:
                if lang == "en" and form == "from_to":
                    continue
                out.append(Candidate(fmt.format(A=A.text, B=B.text), lang, form))
    return out


_RANGE_FORMS = [("{A} {B}", "a_b"), ("{A} മുതൽ {B} വരെ", "from_to"), ("{A} ടു {B}", "a_to_b")]


def _range(s: str, style: str, part) -> List[Candidate]:
    a, b = re.split(r"\s*[-–—~]\s*", s, maxsplit=1)
    return _pair(part(a, style), part(b, style), _RANGE_FORMS)


def _date(s: str, style: str) -> List[Candidate]:
    d, mth, y = re.split(r"[/.\-]", s)
    yi = int(y)
    ny = words(yi, style)
    ey = en_year(yi) if len(y) == 4 else en_words(yi)
    return [Candidate(f"{words(int(d), style)} {words(int(mth), style)} {ny}", "ml", "d_m_y"),
            Candidate(f"{en_words(int(d))} {en_words(int(mth))} {ey}", "en", "d_m_y")]


def _ordinal(s: str, style: str) -> List[Candidate]:
    m = re.match(r"^(\d+)(?:st|nd|rd|th)$", s, re.I)
    if m:
        n = int(m.group(1))
        ya = en_ordinal_ya(n)
        return [Candidate(ordinal(n, style), "ml", "ordinal"), Candidate(en_ordinal(n), "en", "ordinal")] + \
            ([Candidate(ya, "en", "ordinal_ya")] if ya else [])
    m = re.match(r"^(\d+)-?(ാ.*)$", s)
    n, suf = int(m.group(1)), m.group(2)
    return [Candidate(attach(words(n, style), suf), "ml", "ordinal"),
            Candidate(attach(words(n, "glide" if style == "contracted" else "contracted"), suf), "ml", "ordinal"),
            Candidate(en_ordinal(n), "en", "ordinal")]


def en_ordinal_ya(n: int) -> Optional[str]:
    """"-ieth" said with a glide for the tens: 30th തേർട്ടിത്ത് -> തേർട്ടിയത്ത് (30-ന് ->
    തേർട്ടിയത്തിന്), 20th ട്വന്റിയത്ത്. None for other numbers."""
    o = en_ordinal(n)
    return o[:-len("ിത്ത്")] + "ിയത്ത്" if o.endswith("ിത്ത്") else None


def _date_ordinals(digits: str) -> List[Candidate]:
    """English ordinal for a day-of-month-sized number (dates are often said that way:
    ഓഗസ്റ്റ് 27-ലോട്ട് -> ഓഗസ്റ്റ് ട്വന്റി സെവൻത്തിലോട്ട്), plus the glide form of 20th / 30th
    (തേർട്ടിയത്ത്, a spelling variant: form ordinal_th_ya)."""
    if digits.isdigit() and digits[0] != "0" and 1 <= int(digits) <= DATE_MAX:
        out = [Candidate(en_ordinal(int(digits)), "en", "ordinal_th")]
        ya = en_ordinal_ya(int(digits))
        if ya:
            out.append(Candidate(ya, "en", "ordinal_th_ya"))
        return out
    return []


NOSUFFIX = "nosuffix"

# Words written hyphen-joined to a number that are separate words, not case suffixes
# (10-മണിക്ക്, 5-ലക്ഷം, 2-കോടി, 3-കിലോ): read the number, a space, then the word as written.
WORD_STEMS = ("മണി", "ലക്ഷ", "കോടി", "രൂപ", "കിലോ", "ഗ്രാം", "മീറ്റ", "ലിറ്റ", "ശതമാന", "വയസ",
              "ദിവസ", "ആഴ്ച", "മാസ", "വർഷ", "കൊല്ല", "മിനിറ്റ", "മിനുട്ട", "സെക്കൻ", "മണിക്കൂ",
              "പേര", "പേർ", "ആള", "എണ്ണ", "തവണ", "പ്രാവശ്യ", "ഡിഗ്രി", "ഏക്കർ", "സെന്റ", "പവൻ",
              "നമ്പർ", "ക്ലാസ", "നില", "ഓവർ", "റൺ", "വിക്കറ്റ", "ഗോൾ", "പോയിന്റ", "ശതമാനം",
              "ഡോളർ", "പൈസ", "ബില്യ", "മില്യ", "ആയിര", "നൂറ",
              # English words written after a number (3-ഇയർ, 4-സ്റ്റാർ, 1-ഓൺ-1, 3-സ്റ്റെയർ)
              "ഇയർ", "ഇയറ", "സ്റ്റാർ", "സ്റ്റെയർ", "ഷെയർ", "ഓൺ", "ഡേ", "നൈറ്റ്", "റൂം", "പ്ലസ്",
              "കെ", "ലാക്", "ക്രോർ", "അവർ", "വീക്ക്", "മന്ത്", "പേഴ്സൺ", "ടൈം", "സീറ്റ്",
              "ബെഡ്", "പീസ്", "സെറ്റ്", "ജിബി", "എംബി", "ഇഞ്ച്", "ഫീറ്റ്", "ഡോസ്", "ബിഎച്ച്കെ")


def is_word(suffix: str) -> bool:
    """True if the Malayalam text after a number is a separate word (see WORD_STEMS)."""
    return suffix.lstrip("-").startswith(WORD_STEMS)


# English for a unit word written after a number, when the unit itself may be said in English
# (5-മാസം -> ഫൈവ് മന്ത്). Only the bare unit (no case ending) is translated.
UNIT_EN = {"മാസം": "മന്ത്", "വർഷം": "ഇയർ", "കൊല്ലം": "ഇയർ", "ദിവസം": "ഡേ", "ആഴ്ച": "വീക്ക്",
           "മണിക്കൂർ": "അവർ", "മിനിറ്റ്": "മിനിറ്റ്", "മിനുട്ട്": "മിനിറ്റ്", "സെക്കൻഡ്": "സെക്കൻഡ്",
           "ലക്ഷം": "ലാക്", "കോടി": "ക്രോർ", "രൂപ": "റുപ്പീസ്", "ശതമാനം": "പെർസെന്റ്",
           "കിലോ": "കിലോ", "മീറ്റർ": "മീറ്റർ", "ഡിഗ്രി": "ഡിഗ്രി", "തവണ": "ടൈംസ്"}


def en_unit(cands: List[Candidate], word: str) -> List[Candidate]:
    """English number + English unit for the English readings (5-മാസം -> ഫൈവ് മന്ത്): form
    ``<form>+en_unit``. Empty when the word has no English entry or carries a case ending."""
    w = UNIT_EN.get(word.lstrip("-"))
    if not w:
        return []
    return [Candidate(f"{c.text} {w}", "en", f"{c.form}+en_unit") for c in cands
            if c.lang == "en" and f"{c.text} {w}" != glue(c.text, word)]


def glue(reading: str, suffix: str) -> str:
    """Suffix glued with sandhi (attach), or a separate word kept after a space."""
    suffix = suffix.lstrip("-")
    return f"{reading} {suffix}" if is_word(suffix) else attach(reading, suffix)


def _num_suffix(s: str, style: str) -> List[Candidate]:
    """Readings with the written suffix glued on, then the bare number (form ``<form>+nosuffix``):
    the written suffix is not always what was said (27-ന് spoken ഇരുപത്തേഴ്, 55000-യുടെ spoken
    അമ്പത്തയ്യായിരം രൂപയുടെ), so the audio gets to compare with vs without it."""
    m = re.match(r"^([\d,]+)-?(.+)$", s)
    base = _int_or_ids(m.group(1), style)
    base = base + _date_ordinals(m.group(1))
    glued = lambda cs: [Candidate(glue(c.text, m.group(2)), c.lang, c.form) for c in cs]
    if is_word(m.group(2)):                  # 10-മണിക്ക് -> പത്ത് മണിക്ക്: nothing to drop
        base = [c for c in base if not c.form.startswith("ordinal_th")]
        return glued(base) + en_unit(base, m.group(2))      # + 5-മാസം -> ഫൈവ് മന്ത്
    main = [c for c in base if not c.form.endswith("_alt")]
    alts = [c for c in base if c.form.endswith("_alt")]
    hour = []
    num = m.group(1).replace(",", "")
    if m.group(2).lstrip("-") in ("ക്ക്", "ന്") and num.isdigit() and 1 <= int(num) <= 12:
        # 11ക്ക് / 9-ന് as a time: പതിനൊന്ന് മണിക്ക്
        h = int(num)
        hour = [Candidate(f"{'ഒരു' if h == 1 else words(h, style)} മണിക്ക്", "ml", "oclock_manikku")]
    bare = [Candidate(c.text, c.lang, f"{c.form}+{NOSUFFIX}") for c in main
            if c.form != "drop_thousand_spaced"]
    # bare main readings (default style, other style, English) before the colloquial spellings
    # so max_candidates never cuts them
    # the default readings, then their bare forms, then the rest — so max_candidates keeps both
    return glued(main[:3]) + hour + bare[:3] + glued(main[3:]) + glued(alts) + bare[3:]


def suffix_issue(core: str) -> Optional[str]:
    """Why a written suffix looks wrong for its number, or None. A vowel suffix after a number
    ending in ം (55000-യുടെ -> അമ്പത്തയ്യായിരമുടെ, 1000-ഉം) is unusual in speech: usually a noun
    was dropped from the text (അമ്പത്തയ്യായിരം രൂപയുടെ)."""
    s = to_ascii_digits(core).strip()
    m = re.match(r"^([\d,]+)-?([\u0d00-\u0d7f].*)$", s)
    if not m:
        return None
    suf = m.group(2).lstrip("-")
    if suf[:1] == "യ" and len(suf) > 1 and suf[1] in SIGNS:
        suf = suf[1:]
    if not suf or is_word(suf) or not (suf[0] in SIGNS or suf[0] in V_SIGN):
        return None
    try:
        w = words(int(m.group(1).replace(",", "")), "contracted")
    except Exception:
        return None
    if w.endswith("ം") and not w.endswith("ാം"):
        return "vowel suffix after -ം (dropped noun?)"
    return None


SUPERSCRIPT = {"²": "സ്ക്വയർ", "³": "ക്യൂബ്"}       # 5² മീറ്റർ -> അഞ്ച് സ്ക്വയർ മീറ്റർ


def _letters(run: str) -> str:
    if run.isascii() and (len(run) <= 3 or run.isupper()):
        return " ".join(LETTERS.get(ch.lower(), ch) for ch in run)
    return run.lower() if run.isascii() else run


def _piecewise(s: str, style: str) -> List[Candidate]:
    """Fallback for alnum / mixed tokens: read each digit run as a number, spell short Latin
    acronyms, glue a Malayalam suffix onto the number before it (26-27-ഓ -> ... ഇരുപത്തേഴോ),
    read %, and drop other punctuation (it becomes a word break)."""
    pieces = re.findall(r"[0-9]+(?:\.[0-9]+)?|[A-Za-z]+|[\u0d00-\u0d7f]+|\S", s)
    out = []
    for lang in ("ml", "en"):
        ws, prev = [], None                       # prev: value of the number just read
        for p in pieces:
            if p in SUPERSCRIPT:
                ws.append(SUPERSCRIPT[p])
                prev = None
                continue
            if p[0].isdecimal():
                ws.append(next(c.text for c in _number_any(p, style) if c.lang == lang))
                prev = int(p) if p.isdigit() and int(p) <= MAX_NUMBER else None
                continue
            if p in "-–":
                continue                          # keeps prev: 23-ന് still attaches
            if p == "+":
                ws.append("പ്ലസ്")
                prev = None
                continue
            if p == "%":
                ws.append("ശതമാനം" if lang == "ml" else "പെർസെന്റ്")
            elif "\u0d00" <= p[0] <= "\u0d7f":
                if prev is not None and ws:
                    ws[-1] = glue(ws[-1], p)
                else:
                    ws.append(p)
            elif p.isascii() and p.isalpha():
                if prev is not None and p.lower() in ("st", "nd", "rd", "th"):
                    ws[-1] = ordinal(prev, style) if lang == "ml" else en_ordinal(prev)
                    continue                      # a suffix may still follow (12th-ന്റെ)
                ws.append(_letters(p))
            prev = None
        out.append(Candidate(" ".join(w for w in ws if w), lang, "piecewise"))
    return out


_SUFFIXED = re.compile(r"^(\d[\d,.:/%\-]*\d%?|\d)-?([\u0d00-\u0d7f]+)$")


def _phone(s: str, style: str) -> List[Candidate]:
    """Phone numbers: digit by digit first (98765 43210), a leading + as പ്ലസ്; a 10-digit
    number without + also gets its value readings after the split ones."""
    digits = re.sub(r"\D", "", s)
    plus = "പ്ലസ് " if s.startswith("+") else ""
    out = [Candidate(plus + digit_by_digit(digits), "ml", "digit_by_digit"),
           Candidate(plus + en_digit_by_digit(digits), "en", "digit_by_digit")]
    out += [Candidate(plus + c.text, c.lang, c.form) for c in _doubled(digits)]   # ഡബിൾ നൈൻ ...
    if not plus and digits == s and digits[0] != "0" and int(digits) <= MAX_NUMBER:
        out += [c for c in _num(int(digits), style)]
    return out


def _signed(s: str, style: str):
    """-5 / −5 -> മൈനസ് അഞ്ച് (first) and the plain reading; .5 -> പോയിന്റ് അഞ്ച്, അര, ..."""
    m = re.match(r"^[-−]([0-9][0-9,.]*)$", s)
    if m:
        rest = candidates(m.group(1), None, style)
        return [Candidate("മൈനസ് " + c.text, c.lang, c.form + "+minus") for c in rest[:6]] + rest
    m = re.match(r"^\.([0-9]+)$", s)
    if m:
        out = _decimal("0" + s, style)
        return out + [Candidate(f"പോയിന്റ് {digit_by_digit(m.group(1))}", "ml", "point_only"),
                      Candidate(f"പോയിന്റ് {en_digit_by_digit(m.group(1))}", "en", "point_only")]
    return None


# Order rules that must survive max_candidates (see order()): the default stays first and the
# first English reading sits at most third (about a third of spoken numbers are English).
EN_MAX_POS = 2


def order(cands: List[Candidate]) -> List[Candidate]:
    """Enforce the candidate order rules on any list (percent, currency, ranges, ... build their
    lists by language): move the first English reading up to position EN_MAX_POS."""
    k = next((i for i, c in enumerate(cands) if c.lang == "en"), None)
    if k is not None and k > EN_MAX_POS:
        cands = cands[:EN_MAX_POS] + [cands[k]] + cands[EN_MAX_POS:k] + cands[k + 1:]
    return cands


def _digits_only(s: str) -> List[Candidate]:
    d = re.sub(r"[^0-9]", "", s)
    return [Candidate(digit_by_digit(d), "ml", "digit_by_digit")] if d else []


LAKH_WORDS = ("ലക്ഷ", "ലാക്", "lakh", "lac")


def missing_point(s: str, style: str) -> List[Candidate]:
    """2-3 digits before a lakh word, said with a decimal point the text lost: 25 ലക്ഷം ->
    രണ്ടര / രണ്ട് പോയിന്റ് അഞ്ച്, 15 ലാക്സ് -> വൺ പോയിന്റ് ഫൈവ്, 625 -> സിക്സ് പോയിന്റ് ടു ഫൈവ്."""
    if not (s.isdigit() and 2 <= len(s) <= 3 and s[0] != "0"):
        return []
    a, rest = int(s[0]), s[1:]
    out = []
    if rest == "5":
        out.append(Candidate(half(a, style), "ml", "point_missing_half"))
    out += [Candidate(f"{ONES[a]} പോയിന്റ് {digit_by_digit(rest)}", "ml", "point_missing"),
            Candidate(f"{EN_ONES[a]} പോയിന്റ് {en_digit_by_digit(rest)}", "en", "point_missing")]
    return out


def candidates(core: str, cls: str = None, style: str = "contracted", latin: bool = False,
               next_word: str = "") -> List[Candidate]:
    """All plausible spoken readings of one number token core; the first is the default
    (native words in ``style``). Never contains digits or number punctuation.
    ``latin``: the sentence is written in English -> Latin-script English readings only
    (five, twelve thirty); tokens that can't be read that way (23-ന്) keep the usual ones.
    ``next_word``: the word after the number; before a lakh word (25 ലക്ഷം) the readings with a
    lost decimal point are added (രണ്ടര, രണ്ട് പോയിന്റ് അഞ്ച്)."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    s = to_ascii_digits(core).strip()
    cls = cls or classify(s)
    if latin:
        from . import latin as _latin
        lc = _dedupe(_latin.candidates(s, cls))
        if lc:
            return lc
    if cls == "operator":
        return [Candidate("ഇന്റു", "en", "times"), Candidate("ഗുണം", "ml", "times")]
    try:
        out = _candidates(s, cls, style)
        if cls == "integer" and (next_word or "").lower().startswith(LAKH_WORDS):
            out[3:3] = missing_point(s, style)
    except (ValueError, KeyError, AttributeError, StopIteration, IndexError):
        try:
            out = _piecewise(s, style)
        except (ValueError, KeyError, AttributeError, StopIteration, IndexError):
            out = []
    out = order(_dedupe(out))
    return out or _digits_only(s) or [Candidate(s, "ml", "as_written")]


# readings of a number that are never said before "percent" (55% is not "five five percent")
PERCENT_SKIP = ("digit_by_digit", "doubled", "lakh_e", "lakh_short", "pairs", "english_k", "english_lakhs")


def _candidates(s: str, cls: str, style: str) -> List[Candidate]:
    if cls == "phone":
        return _phone(s, style)
    if cls in ("integer", "grouped", "year", "id_code"):
        out = _int_or_ids(s, style)
        if cls == "integer":
            out += _date_ordinals(s)
        if cls == "year":
            out.append(Candidate(en_year(int(s)), "en", "year"))
        if s == "100":                        # "100" written for a spoken "hundred percentage"
            out += [Candidate(f"{EN_HUNDRED} പെർസെന്റേജ്", "en", "percentage_said"),
                    Candidate(f"{EN_HUNDRED} പെർസെന്റ്", "en", "percent_said")]
        return out
    if cls == "decimal":
        return _decimal(s, style)
    if cls == "percent":
        n = [c for c in _number_any(s.rstrip("% ").strip(), style) if not c.form.startswith(PERCENT_SKIP)]
        out = []
        for c in n:          # per reading: ml words + ശതമാനം / പെർസെന്റ്; English + പെർസെന്റ്
            tags = (("ശതമാനം", "shathamanam"), ("പെർസെന്റ്", "percent"), ("പെർസെന്റേജ്", "percentage")) \
                if c.lang == "ml" else (("പെർസെന്റ്", "percent"), ("പെർസെന്റേജ്", "percentage"))
            out += [Candidate(f"{c.text} {w}", c.lang, c.form + "+" + tag) for w, tag in tags]
        return out
    if cls == "currency":
        # ₹ / Rs / INR: the amount with രൂപ / റുപ്പീസ്, then the bare amount (the sign is often not
        # said); never digit by digit
        n = [c for c in _number_any(re.sub(r"(?i)^(₹|rs\.?|inr)\s*|\s*₹$", "", s), style)
             if not c.form.startswith(("digit_by_digit", "doubled"))]
        money = [Candidate(f"{c.text} രൂപ", c.lang, c.form + "+rupa") if c.lang == "ml"
                 else Candidate(f"{c.text} റുപ്പീസ്", "en", c.form + "+rupees") for c in n]
        bare = [Candidate(c.text, c.lang, c.form + "+nocurrency") for c in n if not c.form.endswith("_alt")]
        return money[:4] + bare[:3] + money[4:] + bare[3:]
    if cls == "time":
        return _time(s, style)
    if cls == "time_range":
        return _range(s, style, _time)
    if cls == "range":
        return _range(s, style, _number_any)
    if cls == "date":
        return _date(s, style)
    if cls == "ordinal":
        return _ordinal(s, style)
    if cls == "slash":
        a, b = (int(x) for x in s.split("/"))
        out = []
        frac = {(1, 2): "അര", (1, 4): "കാൽ", (3, 4): "മുക്കാൽ"}.get((a, b))
        if frac:
            out.append(Candidate(frac, "ml", "fraction"))
        return out + _pair(_num(a, style), _num(b, style), [("{A} {B}", "a_b"), ("{A} ബൈ {B}", "a_by_b")])
    if cls == "num_suffix":
        return _num_suffix(s, style)
    sg = _signed(s, style)
    if sg is not None:
        return sg
    mm_ = _SUFFIXED.match(s)
    if mm_ and classify(mm_.group(1)) not in ("other", "alnum"):
        # a known shape + Malayalam suffix: 3:30-ന്, 5:30-ക്ക്, 1.5-ഓ, 2.5-ലക്ഷം -> glue to each reading
        inner_cls = classify(mm_.group(1))
        out = []
        out_src = candidates(mm_.group(1), None, style)
        for c in out_src:
            if inner_cls in ("range", "time_range") and c.form == "from_to":
                continue                     # ... മുതൽ ... വരെ + suffix is not said
            out.append(Candidate(glue(c.text, mm_.group(2)), c.lang, c.form))
        if is_word(mm_.group(2)):            # 2.5-ലക്ഷം -> ടു പോയിന്റ് ഫൈവ് ലാക്
            out += en_unit([c for c in out_src if c.form != "from_to"], mm_.group(2))
        return out
    return _piecewise(s, style)
