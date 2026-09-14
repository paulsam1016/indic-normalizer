"""
Malayalam ASR text normalization for WER/CER scoring.

Applied identically to reference AND hypothesis, so scores aren't inflated by
convention/encoding differences. normalize() does: mis-encoded chillu-conjunct fix
(ൻ്റ→ന്റ, NFC-invisible), chillu / ZWJ-ZWNJ canonicalization (indic-nlp-library if
present, else a stdlib fallback), Unicode NFC, <unk>('⁇') + punctuation removal,
Indic-digit → ASCII, digit verbalization, English↔Malayalam number canonicalization,
lowercase (for code-mixed English), whitespace collapse. See NORMALIZATION.md for the
full coverage/gap analysis and the design rationale.

Digit verbalization: many ASR tokenizers have NO 0-9 pieces, so a digit left in a
target tokenizes to <unk> and teaches the model to emit '⁇' for spoken numbers. Digits
are expanded to native Malayalam number words ("24" -> "ഇരുപത്തിനാല്", "500" ->
"അഞ്ഞൂറ്"), with standard sandhi joins (glide യ before a vowel-initial part: 21 ->
"ഇരുപത്തിയൊന്ന്"). 1-4 digit runs are read as a number; longer runs and zero-padded
runs (phone/account/OTP-style) are read digit-by-digit ("007" -> "പൂജ്യം പൂജ്യം ഏഴ്").

Number canonicalization (convention-invariant scoring): speakers say numbers in English
or Malayalam, and base models often write English-style numbers in Malayalam script
("ട്വൻറ്റി ഫോർ"). Those are parsed back to a value and re-verbalized through the SAME
Malayalam tables, so "24" / "ട്വൻറ്റി ഫോർ" / "ഇരുപത്തിനാല്" all normalize to the same
form. Ambiguous tokens (ടു = "to", ഫോർ = "for") convert only in a run that also
contains an unambiguous number word.

Zero-width chars are written as \\u escapes so this source stays copy-paste-safe.
"""

import re
import unicodedata

from .. import base

_ZW      = base.ZERO_WIDTH               # ZWNJ, ZWJ, BOM (drop AFTER the chillu fix)
_UNK     = base.UNK
_PUNCT   = base.PUNCT
_SPACE   = base.SPACE
# Malayalam chillu written as consonant + virama + ZWJ(‍) -> atomic chillu (fallback
# when indic-nlp is absent). ZWJ escaped so the source is copy-paste-safe.
_CHILLU  = {
    "ണ്‍": "ൺ",
    "ന്‍": "ൻ",
    "ര്‍": "ർ",
    "ല്‍": "ൽ",
    "ള്‍": "ൾ",
    "ക്‍": "ൿ",
}
# Mis-encoded conjunct fix: an atomic chillu is a "dead" consonant with no following
# vowel, so a chillu directly followed by a virama is not a real chillu — it's the base
# consonant + virama typed with the wrong first codepoint (phones/IMEs emit this for the
# nṯa conjunct: ൻ്റ = CHILLU-N + virama + RRA vs the correct ന്റ = NA + virama + RRA,
# e.g. വിവേകാനന്ദൻ്റെ vs വിവേകാനന്ദന്റെ). NFC does NOT unify them, so they score as a
# silent CER mismatch. Rewrite chillu -> base consonant when a virama follows (lossless:
# a genuine word-final chillu has no following virama, so it is left untouched).
_CHILLU_TO_BASE = {
    "ൺ": "ണ",   # CHILLU NN -> NNA
    "ൻ": "ന",   # CHILLU N  -> NA   (ൻ്റ -> ന്റ)
    "ർ": "ര",   # CHILLU RR -> RA
    "ൽ": "ല",   # CHILLU L  -> LA
    "ൾ": "ള",   # CHILLU LL -> LLA
    "ൿ": "ക",   # CHILLU K  -> KA
}
_VIRAMA = "്"
_CHILLU_CONJ = re.compile("([" + "".join(_CHILLU_TO_BASE) + "])" + _VIRAMA)


def _fix_chillu_conjuncts(text: str) -> str:
    """chillu + virama -> base consonant + virama (e.g. ൻ്റ -> ന്റ)."""
    return _CHILLU_CONJ.sub(lambda m: _CHILLU_TO_BASE[m.group(1)] + _VIRAMA, text)


# Indic digit codepoints -> ASCII (Malayalam ൦-൯).
_DIGITS  = {chr(0x0D66 + d): str(d) for d in range(10)}

# Digit runs -> native Malayalam number words.
_NUM_RE   = re.compile(r"[0-9]+")
_ONES     = ["പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒമ്പത്"]
_TEENS    = ["പത്ത്", "പതിനൊന്ന്", "പന്ത്രണ്ട്", "പതിമൂന്ന്", "പതിനാല്", "പതിനഞ്ച്",
             "പതിനാറ്", "പതിനേഴ്", "പതിനെട്ട്", "പത്തൊമ്പത്"]
# Final vs combining stems (combining stem takes the next lower part, e.g. ഇരുപത്തി+നാല്).
_TENS     = ["ഇരുപത്", "മുപ്പത്", "നാൽപ്പത്", "അമ്പത്", "അറുപത്", "എഴുപത്", "എൺപത്", "തൊണ്ണൂറ്"]
_TENS_C   = ["ഇരുപത്തി", "മുപ്പത്തി", "നാൽപ്പത്തി", "അമ്പത്തി", "അറുപത്തി", "എഴുപത്തി",
             "എൺപത്തി", "തൊണ്ണൂറ്റി"]
_HUNDS    = ["നൂറ്", "ഇരുനൂറ്", "മുന്നൂറ്", "നാനൂറ്", "അഞ്ഞൂറ്", "അറുനൂറ്", "എഴുനൂറ്",
             "എണ്ണൂറ്", "തൊള്ളായിരം"]
_HUNDS_C  = ["നൂറ്റി", "ഇരുനൂറ്റി", "മുന്നൂറ്റി", "നാനൂറ്റി", "അഞ്ഞൂറ്റി", "അറുനൂറ്റി",
             "എഴുനൂറ്റി", "എണ്ണൂറ്റി", "തൊള്ളായിരത്തി"]
_THOUS    = ["ആയിരം", "രണ്ടായിരം", "മൂവായിരം", "നാലായിരം", "അയ്യായിരം", "ആറായിരം",
             "ഏഴായിരം", "എണ്ണായിരം", "ഒമ്പതിനായിരം"]
_THOUS_C  = [t[:-1] + "ത്തി" for t in _THOUS]      # ...ം -> ...ത്തി (ആയിരം -> ആയിരത്തി)
# Independent vowel -> glide യ + dependent vowel sign, for sandhi at the join
# (ഇരുപത്തി + ഒന്ന് -> ഇരുപത്തിയൊന്ന്).
_V_SIGN   = {"അ": "", "ആ": "ാ", "ഇ": "ി", "ഈ": "ീ", "ഉ": "ു", "ഊ": "ൂ",
             "എ": "െ", "ഏ": "േ", "ഐ": "ൈ", "ഒ": "ൊ", "ഓ": "ോ", "ഔ": "ൗ"}


def _join(stem: str, rest: str) -> str:
    if rest[0] in _V_SIGN:
        return stem + "യ" + _V_SIGN[rest[0]] + rest[1:]
    return stem + rest


def _num_words(n: int) -> str:
    """0-9999 as spoken Malayalam: 24 -> 'ഇരുപത്തിനാല്', 2024 -> 'രണ്ടായിരത്തിയിരുപത്തിനാല്'."""
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS[t - 2] if u == 0 else _join(_TENS_C[t - 2], _ONES[u])
    if n < 1000:
        h, r = divmod(n, 100)
        return _HUNDS[h - 1] if r == 0 else _join(_HUNDS_C[h - 1], _num_words(r))
    th, r = divmod(n, 1000)
    return _THOUS[th - 1] if r == 0 else _join(_THOUS_C[th - 1], _num_words(r))


def _verbalize_digits(m: "re.Match[str]") -> str:
    s = m.group(0)
    if len(s) > 4 or (len(s) > 1 and s[0] == "0"):
        return " ".join(_ONES[int(c)] for c in s)   # id-like: digit by digit
    return _num_words(int(s))


# --- English-style number words (Malayalam script, as base models write code-mixed
# numbers) -> value -> the same canonical Malayalam words as digits get. Spelling
# variants collected from real base-model output. --------------------------------------
_GROUP_COMMA = re.compile(r"(?<=[0-9]),(?=[0-9]{3})")   # 70,000 -> 70000 (before _PUNCT splits it)
_EN_UNIT = {"സീറോ": 0, "വൺ": 1, "ത്രീ": 3, "ഫൈവ്": 5, "സിക്സ്": 6, "സെവൻ": 7,
            "എയ്റ്റ്": 8, "എയിറ്റ്": 8, "നൈൻ": 9}
_EN_TEEN = {"ടെൻ": 10, "ഇലവൻ": 11, "ഇലെവൻ": 11, "ട്വെൽവ്": 12, "ട്വൽവ്": 12,
            "തേർട്ടീൻ": 13, "ഫോർട്ടീൻ": 14, "ഫിഫ്റ്റീൻ": 15, "സിക്സ്റ്റീൻ": 16,
            "സെവൻറീൻ": 17, "സെവന്റീൻ": 17, "എയ്റ്റീൻ": 18, "നൈൻറീൻ": 19, "നൈന്റീൻ": 19}
_EN_TEN  = {"ട്വൻറ്റി": 20, "ട്വൻ്റി": 20, "ട്വന്റി": 20, "ട്വെന്റി": 20, "ട്വണ്ടി": 20,
            "തേർട്ടി": 30, "തർട്ടി": 30, "ഫോർട്ടി": 40, "ഫിഫ്റ്റി": 50, "ഫിഫ്ടി": 50,
            "സിക്സ്റ്റി": 60, "സെവൻറ്റി": 70, "സെവന്റി": 70, "സെവൻറി": 70,
            "എയ്റ്റി": 80, "നൈൻറ്റി": 90, "നൈന്റി": 90, "നയന്റി": 90}
_EN_HUND = {"ഹൺഡ്രഡ്", "ഹണ്ട്രഡ്", "ഹൻഡ്രഡ്", "ഹൺട്രഡ്"}
_EN_THOU = {"തൗസൻഡ്", "തൗസന്റ്", "തൗസണ്ട്", "തൗസൻറ്"}
# Also English "to"/"for" in code-mixed speech; convert only beside an unambiguous number word.
_EN_AMBIG = {"ടു": 2, "ഫോർ": 4}
# Unit words swapped verbatim (refs write "2 ലക്ഷം"; base writes "ടു ലാഖ്").
_EN_WORD = {"ലാഖ്": "ലക്ഷം", "ലാക്സ്": "ലക്ഷം", "ലാഖ്സ്": "ലക്ഷം"}
_EN_STRONG = (set(_EN_UNIT) | set(_EN_TEEN) | set(_EN_TEN) | _EN_HUND | _EN_THOU
              | set(_EN_WORD))
_EN_ANY = _EN_STRONG | set(_EN_AMBIG)


def _en_run_to_text(words: list) -> str:
    """One run of English-style number words -> canonical Malayalam text.

    Handles the spoken patterns that occur in calls: bare numbers, tens+units
    ("ട്വൻറ്റി ഫോർ"), X hundred [Y] ("ഫൈവ് ഹൺഡ്രഡ് ഫിഫ്റ്റി"), X thousand
    ("സെവൻറ്റി തൗസൻഡ്"), digit-by-digit ("നൈൻ എയ്റ്റ് സെവൻ" -> three numbers), and lakh
    units. Values >9999 fall back to digit-by-digit words — the same canonical form a
    "70,000" reference reaches via digits.
    """
    out = []
    cur, seen, unit_done = 0, False, False

    def flush():
        nonlocal cur, seen, unit_done
        if seen:
            out.append(_num_words(cur) if cur <= 9999
                       else " ".join(_ONES[int(c)] for c in str(cur)))
        cur, seen, unit_done = 0, False, False

    for w in words:
        if w in _EN_WORD:
            flush()
            out.append(_EN_WORD[w])
        elif w in _EN_UNIT or w in _EN_AMBIG:
            if unit_done:
                flush()
            cur += _EN_UNIT.get(w, _EN_AMBIG.get(w))
            seen, unit_done = True, True
        elif w in _EN_TEEN:
            if unit_done or cur % 100:
                flush()
            cur += _EN_TEEN[w]
            seen, unit_done = True, True
        elif w in _EN_TEN:
            if unit_done or cur % 100:
                flush()
            cur += _EN_TEN[w]
            seen = True
        elif w in _EN_HUND:
            cur = (cur if cur else 1) * 100
            seen, unit_done = True, False
        elif w in _EN_THOU:
            cur = (cur if cur else 1) * 1000
            seen, unit_done = True, False
    flush()
    return " ".join(out)


def _canonicalize_en_numbers(text: str) -> str:
    """Replace runs of English-style number words with canonical Malayalam number words.

    A run converts only if it contains at least one unambiguous number word, so bare
    "ടു"/"ഫോർ" (English to/for) pass through untouched.
    """
    words = text.split()
    out, i = [], 0
    while i < len(words):
        j = i
        while j < len(words) and words[j] in _EN_ANY:
            j += 1
        if j > i and any(words[k] in _EN_STRONG for k in range(i, j)):
            out.append(_en_run_to_text(words[i:j]))
        else:
            out.extend(words[i:j] or [words[i]])
            j = max(j, i + 1)
        i = j
    return " ".join(out)


try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _IND_NORM = IndicNormalizerFactory().get_normalizer("ml")
except Exception:
    _IND_NORM = None

INDIC_NLP_AVAILABLE = _IND_NORM is not None


def normalize(text: str) -> str:
    """Normalize Malayalam ASR text (see module docstring); returns cleaned, lowercased text."""
    text = _fix_chillu_conjuncts(text or "")      # ൻ്റ -> ന്റ etc. (mis-encoded conjunct, NFC-invisible)
    if _IND_NORM is not None:
        text = _IND_NORM.normalize(text)          # canonicalizes chillu / ZWJ properly
    else:
        for seq, atom in _CHILLU.items():
            text = text.replace(seq, atom)        # fallback chillu fix BEFORE dropping ZWJ
    text = unicodedata.normalize("NFC", text)
    text = _UNK.sub(" ", text)
    text = _ZW.sub("", text)
    text = "".join(_DIGITS.get(c, c) for c in text)
    text = _GROUP_COMMA.sub("", text)              # 70,000 -> 70000 before ',' becomes a space
    text = _PUNCT.sub(" ", text)
    text = _canonicalize_en_numbers(text)          # "ട്വൻറ്റി ഫോർ" -> same canonical form as "24"
    text = _NUM_RE.sub(_verbalize_digits, text)    # digits are OOV for the tokenizer -> spell out
    text = text.lower()                            # for code-mixed English tokens
    return _SPACE.sub(" ", text).strip()
