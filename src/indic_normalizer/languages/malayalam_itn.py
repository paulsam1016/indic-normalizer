"""Inverse text normalization (ITN) for Malayalam inference output: spoken number words -> digits.

Built by INVERTING the digits module's forward generator (``indic_normalizer.digits.ml``), so
every reading the training targets use reads back: the contracted / glide / spaced spellings
(ഇരുപത്തഞ്ച് / ഇരുപത്തിയഞ്ച് / ഇരുപത്തി അഞ്ച് -> 25), colloquial ones (എട്ടായിരം, പത്തായിരം),
English words in Malayalam script (ട്വന്റി ഫൈവ് -> 25), lakhs and crores (രണ്ട് ലക്ഷത്തി
ഇരുപത്തയ്യായിരം -> 2,25,000), suffixes (ഇരുപത്തേഴിന് -> 27-ന്), decimals and halves
(രണ്ട് പോയിന്റ് അഞ്ച് / രണ്ടര -> 2.5, നാലര മണി -> 4:30 മണി), percent (25%), English digit
forms (ഡബിൾ ഫൈവ് -> 55, ട്വന്റി കെ -> 20k) and English ordinals (ട്വന്റി ഫിഫ്ത്ത് -> 25th).

Lookup, not a hand-written parser: a word is a number only if the generator produces exactly that
word; a word with a suffix only if ``ml.attach(number, suffix)`` rebuilds it exactly (so
ആയിരിക്കും "will be" is never 1000 + a suffix). Below 10,000 all spellings sit in one table (built
on first use); 10,000-99,999 are found by their thousands part and that group's 1,000 values
are generated on demand; lakhs and crores are composed from their parts.

Left as words on purpose: 1-9 with a suffix other than the ordinal ആം (ഒന്നും "anything",
രണ്ടും "both" are idioms), a lone ടു / ഫോർ (English to / for), a lone ഫസ്റ്റ് / സെക്കൻഡ്
(first / a second), and the X-ഏ YY shorthand (ഒന്നേ അമ്പത്തഞ്ച് = 1,55,000 or 1 55?).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Tuple

from ..digits import ml as _d
from . import malayalam as _m

VIRAMA = _d.VIRAMA
SIGNS = _d.SIGNS
SIGN2VOWEL = {v: k for k, v in _d.V_SIGN.items() if v}
CHILLU_OF = {base: ch for ch, base in _d.CHILLU_BASE.items()}       # ന -> ൻ (സെവനിന് -> സെവൻ)
STYLES = ("contracted", "glide")

# ------------------------------------------------------------------------------ lexicons

@lru_cache(maxsize=None)
def _native_lex() -> dict:
    """Every spelling of 0-9,999 -> value (contracted, glide and their colloquial variants)."""
    lex = {}
    for n in range(10000):
        for st in STYLES:
            w = _d.words(n, st)
            for x in (w, _d.alt_spelling(w)):
                lex.setdefault(x, n)
    return lex


@lru_cache(maxsize=None)
def _thousand_stems() -> List[Tuple[str, int]]:
    """'പന്ത്രണ്ടായിര' -> 12 for every spelling of 10-99 thousand, longest first."""
    out = {}
    for x in range(10, 100):
        for st in STYLES:
            w = _d.words(x * 1000, st)
            for t in (w, _d.alt_spelling(w)):
                out.setdefault(t[:-1], x)                    # drop the final ം
    return sorted(out.items(), key=lambda kv: -len(kv[0]))


@lru_cache(maxsize=128)
def _group(x: int) -> dict:
    """Every spelling of x thousand + 0-999 (one group of 1,000 values)."""
    lex = {}
    for r in range(1000):
        for st in STYLES:
            w = _d.words(x * 1000 + r, st)
            for t in (w, _d.alt_spelling(w)):
                lex.setdefault(t, x * 1000 + r)
    return lex


@lru_cache(maxsize=None)
def _fractions() -> dict:
    """Halves and quarters: രണ്ടര 2.5, രണ്ടേകാൽ 2.25, രണ്ടേമുക്കാൽ 2.75 (1-999; അര / കാൽ alone
    are words)."""
    lex = {}
    for n in range(1, 1000):
        for st in STYLES:
            lex.setdefault(_d.half(n, st), n + 0.5)
            w = _d.words(n, st)
            if w.endswith(VIRAMA):
                lex.setdefault(w[:-1] + "േകാൽ", n + 0.25)
                lex.setdefault(w[:-1] + "േമുക്കാൽ", n + 0.75)
    return lex


@lru_cache(maxsize=None)
def _combining() -> frozenset:
    """Non-final words of the spaced spelling (ഇരുപത്തി, നൂറ്റി, ആയിരത്തി, മൂവായിരത്തി,
    പന്ത്രണ്ടായിരത്തി ...): they join with what follows into one number."""
    out = set()
    for n in range(10000):
        w = _d.words(n, "spaced")
        for t in (w, _d.alt_spelling(w)):
            out.update(t.split()[:-1])
    for x in range(10, 100):
        w = _d.words(x * 1000 + 1, "spaced")
        for t in (w, _d.alt_spelling(w)):
            out.update(t.split()[:-1])
    return frozenset(out)


def native_value(word: str) -> Optional[float]:
    """The value of a whole native number word (no suffix), or None."""
    v = _native_lex().get(word)
    if v is not None:
        return v
    if "ായിര" in word:                                       # 10,000-99,999
        for stem, x in _thousand_stems():
            if word.startswith(stem):
                v = _group(x).get(word)
                if v is not None:
                    return v
    return _fractions().get(word)


# English number words (Malayalam script): the digits tables + spellings seen in base-model output
def _en_tables():
    unit = dict(_m._EN_UNIT)
    unit.update({w: i for i, w in enumerate(_d.EN_ONES) if w not in ("ടു", "ഫോർ")})
    teen = dict(_m._EN_TEEN)
    teen.update({w: 10 + i for i, w in enumerate(_d.EN_TEENS)})
    ten = dict(_m._EN_TEN)
    ten.update({w: t * 10 for t, w in _d.EN_TENS.items()})
    mult = {w: 100 for w in _m._EN_HUND | {_d.EN_HUNDRED}}
    mult.update({w: 1000 for w in _m._EN_THOU | {_d.EN_THOUSAND}})
    mult.update({w: 100_000 for w in ("ലാക്", "ലാക്ക്", "ലാക്സ്", "ലാഖ്", "ലാഖ്സ്")})
    mult.update({w: 10_000_000 for w in ("ക്രോർ", "ക്രോഴ്സ്", "ക്രോർസ്")})
    return unit, teen, ten, mult


def _au_variants(table: dict) -> dict:
    """indic-nlp writes the AU sign ൗ as ൌ (തൗസന്റ് -> തൌസന്റ്): accept both."""
    out = dict(table)
    out.update({k.replace("ൗ", "ൌ"): v for k, v in table.items() if "ൗ" in k})
    return out


EN_UNIT, EN_TEEN, EN_TEN, EN_MULT = map(_au_variants, _en_tables())
EN_AMBIG = {"ടു": 2, "ഫോർ": 4}                  # also "to" / "for": need an unambiguous neighbour
EN_REPEAT = {"ഡബിൾ": 2, "ട്രിപ്പിൾ": 3}         # in a digit run: ഡബിൾ ഫൈവ് -> 55
EN_K = "കെ"                                      # ട്വന്റി കെ -> 20k
ORD_LONE_NO = {"ഫസ്റ്റ്", "സെക്കൻഡ്"}            # first / a second: only after a tens word
ORD_NEVER = {"സെക്കൻഡ്"}                         # ട്വന്റി സെക്കൻഡ് = 22nd or 20 seconds


@lru_cache(maxsize=None)
def _en_ordinals() -> dict:
    """Last word of an English ordinal -> the value it adds (ഫിഫ്ത്ത് 5, ട്വന്റിയത്ത് 20)."""
    out = {}
    for n in range(1, 100):
        for o in (_d.en_ordinal(n), _d.en_ordinal_ya(n)):
            if not o:
                continue
            *head, last = o.split()
            hv = sum(EN_TEN.get(h, 0) for h in head)
            if last not in ORD_NEVER:
                out.setdefault(last, n - hv)
    return out


LAKH_WORDS = {_d.LAKH: (100_000, False), _d.LAKH_C: (100_000, True), _d.CRORE: (10_000_000, False)}
POINT = {"പോയിന്റ്"}
PERCENT = {"ശതമാനം", "പെർസെന്റ്", "പെർസെന്റേജ്", "പെഴ്സന്റ്", "പെർസന്റ്"}
TIME_WORD = "മണി"

# ------------------------------------------------------------------------------ suffixes

_NUMBERLIKE = re.compile("^(?:" + "|".join(sorted(map(re.escape, _d.ONES[1:] + _d.EN_ONES[1:]),
                                                     key=len, reverse=True)) + ")")


def _suffix_candidates(rest: str) -> List[str]:
    """Readable forms of the text after a number's stem, cleanest first (ിന് -> ന്, ും -> ഉം,
    ത്തിൽ -> ൽ, ടുവിന് -> ന്); each is checked with ml.attach before it is used."""
    c = []
    if rest.startswith("ക്ക"):
        c.append("ന" + rest[3:])                             # ത്രീക്ക് -> 3-ന്
    if rest.startswith("യ്ക്ക"):
        c.append("ന" + rest[5:])                             # മൂന്നരയ്ക്ക് -> ന്
    if rest.startswith("ത്തി") and len(rest) > 4:
        c.append(rest[4:])                                   # ആയിരത്തിൽ -> ൽ
    if rest[:2] in ("യി", "വി") and len(rest) > 2:
        c.append(rest[2:])                                   # ട്വന്റിയിൽ / ടുവിന് -> ൽ / ന്
    if rest[0] == "ി" and len(rest) > 1:
        c.append(rest[1:])                                   # ിന് -> ന്
    if rest[0] in ("യ", "വ", "മ") and len(rest) > 1 and rest[1] in SIGN2VOWEL:
        c.append(SIGN2VOWEL[rest[1]] + rest[2:])            # വും -> ഉം, മാണ് -> ആണ്
    if rest[0] in SIGN2VOWEL:
        c.append(SIGN2VOWEL[rest[0]] + rest[1:])            # ും -> ഉം, ാം -> ആം
    if rest[0] not in SIGNS and rest[0] != VIRAMA:
        c.append("അ" + rest)                                 # ഇരുപത്താറല്ലേ -> അല്ലേ
    c.append(rest)
    return list(dict.fromkeys(x for x in c if x))


def _heads(prefix: str) -> List[str]:
    out = [prefix, prefix + VIRAMA, prefix + "ം"]
    if prefix and prefix[-1] in CHILLU_OF:
        out.append(prefix[:-1] + CHILLU_OF[prefix[-1]])     # സെവന|ിന് -> സെവൻ
    return out


def _word_info(word: str):
    """('ml', value) / ('frac', value) / ('en', kind, value) / ('lakh', mult, combining) or None."""
    v = native_value(word)
    if v is not None:
        return ("frac", v) if isinstance(v, float) else ("ml", v)
    if word in EN_UNIT:
        return ("en", "unit", EN_UNIT[word])
    if word in EN_TEEN:
        return ("en", "teen", EN_TEEN[word])
    if word in EN_TEN:
        return ("en", "ten", EN_TEN[word])
    if word in EN_AMBIG:
        return ("en", "ambig", EN_AMBIG[word])
    if word in EN_MULT:
        return ("en", "mult", EN_MULT[word])
    if word in _en_ordinals():
        return ("en", "ord", _en_ordinals()[word])
    if word in LAKH_WORDS:
        return ("lakh",) + LAKH_WORDS[word]
    return None


JOIN_MARKS = {"ത്തി", "അത്തി", "ി", "ഇ", "റ്റി"}           # the -ി of a spaced part (ആയിരത്തി, നൂറ്റി)


def split_suffix(core: str):
    """(info, suffix) for a number word with a case suffix glued on, verified with ml.attach;
    None if ``core`` is not one."""
    for i in range(len(core) - 1, 0, -1):
        rest = core[i:]
        for head in _heads(core[:i]):
            info = _word_info(head)
            if info is None or info[0] == "lakh" and info[2]:
                continue
            cands = _suffix_candidates(rest)
            if any(_NUMBERLIKE.match(s) or native_value(s) is not None for s in cands):
                continue                                      # two numbers run together: പത്തിരുപത്
            for s in cands:
                if s in JOIN_MARKS or "കണക്ക" in s:
                    continue                                  # ...ത്തി joins a spaced part; ആയിരക്കണക്കിന്
                if _d.attach(head, s) == core:
                    return info, s
    return None

# ------------------------------------------------------------------------------ tokens

ORDINAL_SFX = ("ആം", "ആമ")


def _show_suffix(s: str) -> str:
    return "ാ" + s[1:] if s.startswith(ORDINAL_SFX) else s   # keep 4-ാം / 2-ാമത്തെ


class _Tok:
    __slots__ = ("raw", "lead", "core", "trail", "info", "sfx")

    def __init__(self, raw):
        self.raw = raw
        self.lead, self.core, self.trail = _m._peel(raw)
        self.info, self.sfx = None, ""
        c = self.core
        if not c or c[0].isascii():
            return
        info = _word_info(c)
        if info is None and c.endswith("ു") and native_value(c[:-1] + VIRAMA) is not None:
            info = ("ml", native_value(c[:-1] + VIRAMA))     # രണ്ടു / നാലു / പത്തു
        if info is None and c not in _combining():
            r = split_suffix(c)
            if r:
                info, self.sfx = r
        if info is None:                                      # run-together parts (ഇരുനൂറ്റിയെണ്ണൂറ്റി...)
            p = _m._parse_leading_number(c)
            if p and p[1] == len(c):
                info = ("ml", p[0])
        self.info = info

    @property
    def kind(self):
        return self.info[0] if self.info else None


def _clean_between(toks, i, j) -> bool:
    """No punctuation inside the span i..j-1 (it may only be around it)."""
    return all(not toks[k].trail for k in range(i, j - 1)) and all(not toks[k].lead for k in range(i + 1, j))

# ------------------------------------------------------------------------------ rendering

def _indian(n: int) -> str:
    s = str(n)
    if n < 100_000:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head] + groups + [tail])


def _fmt(v) -> str:
    if isinstance(v, float):
        if v.is_integer():
            v = int(v)
        else:
            return ("%.2f" % v).rstrip("0").rstrip(".")
    return _indian(v)


def _ord_suffix(n: int) -> str:
    if n % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

# ------------------------------------------------------------------------------ parsing

def _native_part(toks, i):
    """A native number below one lakh starting at i: (end, value, suffix, token) or None. Spaced
    words are joined (ഇരുപത്തി ഒന്ന് -> ഇരുപത്തിയൊന്ന്); ഒരു before നൂറ് is the said "one"."""
    comb = _combining()
    k, parts = i, []
    while k < len(toks) and toks[k].core in comb and not toks[k].sfx:
        parts.append(toks[k].core)
        k += 1
        if k < len(toks) and toks[k].core == "ഒരു" and k + 1 < len(toks) and toks[k + 1].core.startswith("നൂറ"):
            k += 1                                            # നാലായിരത്തി ഒരു നൂറ്
    if parts:
        if k >= len(toks) or not _clean_between(toks, i, k + 1):
            return None
        joined = parts[0]
        for p in parts[1:] + [toks[k].core]:
            joined = _m._join(joined, p)
        t = _Tok(joined)
        if t.kind not in ("ml", "frac"):
            return None
        return k + 1, t.info[1], t.sfx, t
    t = toks[i]
    if t.kind in ("ml", "frac"):
        return i + 1, t.info[1], t.sfx, t
    return None


def _native(toks, i):
    """A native number at i, lakhs and crores composed: (end, value, suffix) or None."""
    if toks[i].core == "ഒരു" and i + 1 < len(toks) and toks[i + 1].core.startswith(TIME_WORD) \
            and not toks[i].trail:
        return i + 1, 1, ""                                   # ഒരു മണി -> 1 മണി
    if toks[i].core == "ഒരു" and i + 1 < len(toks) and toks[i + 1].kind == "lakh":
        part = (i + 1, 1, "", toks[i])
    else:
        part = _native_part(toks, i)
    if not part:
        return None
    end, value, sfx, _ = part
    if sfx or end >= len(toks) or toks[end].kind != "lakh" or not _clean_between(toks, i, end + 1):
        return end, value, sfx
    mult, combining = toks[end].info[1], toks[end].info[2]
    if value >= (100 if mult == 100_000 else 1000):
        return end, value, sfx
    total = int(round(value * mult))
    end += 1
    if toks[end - 1].sfx:
        return end, total, toks[end - 1].sfx
    if end < len(toks) and _clean_between(toks, i, end + 1):
        rest = _native(toks, end)
        if rest and rest[1] < mult and not isinstance(rest[1], float):
            return rest[0], total + rest[1], rest[2]
    if combining:
        return None                                           # ലക്ഷത്തി with nothing after it
    return end, total, ""


def _english(toks, i):
    """A run of English number words at i: (end, rendered) or None. Needs one unambiguous word
    (a lone ടു / ഫോർ is "to" / "for")."""
    j = i
    while j < len(toks) and (toks[j].kind == "en" or toks[j].core in EN_REPEAT or toks[j].core == EN_K) \
            and (j == i or not toks[j].lead) and (j == i or not toks[j - 1].trail):
        j += 1
        if toks[j - 1].sfx or (toks[j - 1].kind == "en" and toks[j - 1].info[1] == "ord"):
            break                                             # a suffix / an ordinal ends the run
    while j > i and (toks[j - 1].core in EN_REPEAT or toks[j - 1].core == EN_K and j - 1 == i):
        j -= 1                                                # ഡബിൾ / കെ at the end: words
    if j < len(toks) and toks[j].core == EN_K and j > i and not toks[j].lead and not toks[j - 1].trail:
        j += 1                                                # ടു കെ (the run stopped at ടു)
    run = toks[i:j]
    nxt = toks[j].core if j < len(toks) else ""
    numeric_next = (nxt in PERCENT or nxt == EN_K
                    or nxt in POINT and j + 1 < len(toks) and _digit_word(toks[j + 1]) is not None
                    or nxt == "ഓ" and j + 1 < len(toks) and toks[j + 1].core.startswith("ക്ലോക്ക"))
    numeric_next = numeric_next or any(t.core == EN_K for t in run[1:])   # ടു കെ -> 2k
    if not run or not (numeric_next or any(t.kind == "en" and t.info[1] != "ambig" for t in run)):
        return None                                           # a lone ടു / ഫോർ is "to" / "for"
    if run[0].core == EN_K or (len(run) == 1 and run[0].kind == "en" and run[0].info[1] == "ord"
                               and run[0].core in ORD_LONE_NO):
        return None
    # digit by digit (codes, phone numbers): units / to / for / double / triple only
    if len(run) >= 2 and all(t.core in EN_REPEAT or t.kind == "en" and t.info[1] in ("unit", "ambig") for t in run):
        out, rep = [], 1
        for t in run:
            if t.core in EN_REPEAT:
                rep = EN_REPEAT[t.core]
                continue
            out.append(str(t.info[2]) * rep)
            rep = 1
        sfx = run[-1].sfx
        return j, "".join(out) + ("-" + _show_suffix(sfx) if sfx else "")
    nums, total, cur, seen, unit_done = [], 0, 0, False, False
    k_flag = ord_flag = False
    last_big = 0                                              # scale of the last thousand / lakh / crore

    def nonlocal_reset():
        nonlocal last_big
        last_big = 0

    def flush():
        nonlocal total, cur, seen, unit_done, k_flag, ord_flag
        if seen:
            v = total + cur
            nums.append((f"{v}k" if k_flag else (f"{v}{_ord_suffix(v)}" if ord_flag else _fmt(v))))
        total, cur, seen, unit_done, k_flag, ord_flag = 0, 0, False, False, False, False
        nonlocal_reset()

    for t in run:
        if t.core in EN_REPEAT:
            continue
        if t.core == EN_K:
            k_flag = True
            continue
        kind, v = t.info[1], t.info[2]
        if kind in ("unit", "ambig"):
            if unit_done or k_flag:
                flush()
            cur += v
            seen = unit_done = True
        elif kind == "teen":
            if unit_done or cur % 100 or k_flag:
                flush()
            cur += v
            seen = unit_done = True
        elif kind == "ten":
            if unit_done or cur % 100 or k_flag:
                flush()
            cur += v
            seen = True
        elif kind == "ord":
            if unit_done or (v >= 10 and cur % 100):
                flush()
            cur += v
            seen, ord_flag = True, True
        elif kind == "mult":
            if v >= 1000 and last_big and v >= last_big:     # ടെൻ തൗസന്റ് ഫിഫ്റ്റീൻ തൗസന്റ്: two numbers
                keep = cur
                cur = 0
                flush()
                cur, seen = keep, True
            if v >= 1000:
                last_big = v
            if v == 100:
                cur = (cur or 1) * 100
            elif v == 1000:
                total += (cur or 1) * 1000
                cur = 0
            else:
                total += (cur or 1) * v
                cur = 0
            seen, unit_done = True, False
    flush()
    if not nums:
        return None
    sfx = run[-1].sfx
    return j, " ".join(nums) + ("-" + _show_suffix(sfx) if sfx else "")


def _digit_word(t) -> Optional[str]:
    if t.sfx:
        return None
    if t.kind == "ml" and t.core in _d.ONES:
        return str(t.info[1])
    if t.kind == "en" and t.info[1] in ("unit", "ambig"):
        return str(t.info[2])
    return None


def itn(text: str) -> str:
    """Number words in ``text`` -> digits (see the module docstring); everything else as is."""
    toks = [_Tok(t) for t in text.split()]
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        # digit-by-digit runs of native single digits (പൂജ്യം പൂജ്യം ഏഴ് -> 007)
        if t.kind == "ml" and t.core in _d.ONES and not t.sfx:
            j = i
            while j < len(toks) and toks[j].kind == "ml" and toks[j].core in _d.ONES and not toks[j].sfx \
                    and (j == i or not toks[j].lead) and (j == i or not toks[j - 1].trail):
                j += 1
            if j - i >= 2:
                out.append(t.lead + "".join(str(x.info[1]) for x in toks[i:j]) + toks[j - 1].trail)
                i = j
                continue
        # X-ഏ YY (ഒന്നേ അമ്പത്തഞ്ച്): ambiguous shorthand, both stay words
        if t.core.endswith("േ") and native_value(t.core[:-1] + VIRAMA) is not None \
                and i + 1 < len(toks) and (toks[i + 1].kind in ("ml", "frac") or toks[i + 1].core in _combining()):
            part = _native_part(toks, i + 1)               # the whole YY, also when spaced
            end = part[0] if part else i + 2
            out += [x.raw for x in toks[i:end]]
            i = end
            continue
        en = _english(toks, i) if (t.kind == "en" or t.core in EN_REPEAT) else None
        if en:
            j, body = en
            j, body = _modifiers(toks, i, j, body, None)
            out.append(t.lead + body + toks[j - 1].trail)
            i = j
            continue
        nat = _native(toks, i) if (t.kind in ("ml", "frac", "lakh") or t.core in _combining() or t.core == "ഒരു") else None
        if nat:
            j, value, sfx = nat
            if sfx and not sfx.startswith(ORDINAL_SFX) and isinstance(value, int) and value < 10:
                out.append(t.raw)                             # ഒന്നും / രണ്ടും: words
                i += 1
                continue
            if sfx and isinstance(value, float):
                if sfx.startswith("ന") and not sfx.startswith("ന്റ"):   # മൂന്നരയ്ക്ക്(ുള്ള) "at half past three"
                    h, m = int(value), int(round((value - int(value)) * 60))
                    out.append(t.lead + f"{h}:{m:02d}-{sfx}" + toks[j - 1].trail)
                    i = j
                    continue
                out.append(t.raw)
                i += 1
                continue
            body = _fmt(value) + ("-" + _show_suffix(sfx) if sfx else "")
            j, body = _modifiers(toks, i, j, body, None if sfx else value)
            out.append(t.lead + body + toks[j - 1].trail)
            i = j
            continue
        out.append(t.raw)
        i += 1
    return " ".join(out)


def _modifiers(toks, i, j, body, value):
    """After a number: പോയിന്റ് + digits (2.5), percent (25%), half / quarter before മണി (4:30)."""
    plain = value is not None or body.isdigit()
    if j < len(toks) and plain and toks[j].core in POINT and not toks[j].lead and not toks[j - 1].trail \
            and not toks[j].trail:
        k, ds = j + 1, []
        while k < len(toks) and _digit_word(toks[k]) is not None and not toks[k].lead \
                and not toks[k - 1].trail:
            ds.append(_digit_word(toks[k]))
            k += 1
        if ds:
            body, j = body.replace(",", "") + "." + "".join(ds), k
            value = None
    if isinstance(value, float) and j < len(toks) and toks[j].core.startswith(TIME_WORD) \
            and not toks[j - 1].trail:
        h, m = int(value), int(round((value - int(value)) * 60))
        return j, f"{h}:{m:02d}"                              # the മണി word stays
    if j < len(toks) and toks[j].core in PERCENT and not toks[j].lead and not toks[j - 1].trail \
            and re.fullmatch(r"[\d.,]+", body):
        return j + 1, body + "%"
    return j, body
