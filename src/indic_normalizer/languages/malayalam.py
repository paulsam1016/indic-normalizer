"""
Malayalam ASR text normalization for WER/CER scoring.

Applied identically to reference AND hypothesis, so scores aren't inflated by
convention/encoding differences. normalize() does: mis-encoded chillu-conjunct fix
(ൻ്റ→ന്റ, NFC-invisible), chillu / ZWJ-ZWNJ canonicalization (via indic-nlp-library, a
hard dependency), Unicode NFC, <unk>('⁇') + punctuation removal, Indic-digit → ASCII,
digit verbalization, English↔Malayalam number canonicalization, lowercase (for
code-mixed English), the CER folds C6 (homoglyph ഠ→0 in a number) and C2 (word-final മ्→ം) in
both modes, plus C1 (word-final samvruthokaram ...ു→...്, eval only), and the opt-in
segmentation fixes W4 (intra-word hyphen join, e-mail→email), W3 (single-letter acronym-run
collapse, എ ടി എം→എടിഎം) and W5 (adjacent-duplicate collapse, വേഗം വേഗം→വേഗം), then whitespace
collapse. All the above are EVAL-mode; inference mode does reader-facing ITN instead (see the
mode selector). See NORMALIZATION.md for the full coverage/gap analysis and the rationale.

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


# ===== CER canonicalization rules C6 / C2 / C1 =======================================
# These reduce character-level noise from spelling/encoding conventions of the SAME word,
# applied symmetrically to ref and hyp. C6 (homoglyph) and C2 (word-final മ्->ം) are
# unambiguous and run in BOTH modes. C1 (samvruthokaram fold) is EVAL-ONLY: it can also fold
# a genuine -u word (see its note), fine for symmetric scoring but wrong for a reader-facing
# transcript. See normalize() and the measured-impact note in NORMALIZATION.md.

# --- C6: visual-lookalike letters/digits ----------------------------------------------
# Homoglyphs that ASR/typists confuse. Kept intentionally tiny and unambiguous: only the
# Latin capital O / zero-glyph and the Malayalam letter ഠ when standing in for the Malayalam
# ZERO ൦/0 inside a digit context. A bare Latin "O" as a word is NOT touched (could be real).
_C6_MAP = {
    "ഠ": "0",   # ഠ (TTHA) — visually the zero glyph; map to ASCII 0 when digit-adjacent
}
_C6_LOOKALIKE = re.compile(r"(?<=[0-9൦-൯])ഠ|ഠ(?=[0-9൦-൯])")


def _fix_homoglyphs(text: str) -> str:
    """ഠ used as zero inside a number -> 0 (digit-adjacent only; standalone ഠ untouched)."""
    return _C6_LOOKALIKE.sub("0", text)


# --- C2: word-final m-nasal (മ് -> ം) --------------------------------------------------
# The m-nasal written as മ് (MA + virama) word-finally is the same sound as the anusvara ം
# and standard orthography uses ം. Canonicalize word-final മ് -> ം. Deliberately does NOT
# touch ന് (that is the chillu-n) — only the labial m. Runs in BOTH modes (unambiguous), so
# the word-end boundary allows a following non-Malayalam-letter (space, end, or punctuation
# that inference keeps, e.g. "വരാമ്." -> "വരാം.").
_C2_FINAL_MA = re.compile("മ" + _VIRAMA + r"(?![ഀ-ൿ])")


def _fold_final_nasal(text: str) -> str:
    """word-final മ് -> ം (labial nasal to anusvara). Leaves ന് alone."""
    return _C2_FINAL_MA.sub("ം", text)


# Train-only C2: never fold a മ് that is itself the second half of a conjunct (ഹ്മ് "hmm" would
# become the malformed ഹ്ം — virama + anusvara). Eval keeps the unguarded symmetric fold.
_C2_FINAL_MA_TRAIN = re.compile("(?<!\u0d4d)" + _C2_FINAL_MA.pattern)


def _fold_final_nasal_train(text: str) -> str:
    return _C2_FINAL_MA_TRAIN.sub("ം", text)


# Train-only AU fix: indic-nlp rewrites the modern AU length mark ൗ (U+0D57, കൗ) as the
# two-part sign ൌ (U+0D4C). Same sound, but a different tokenizer piece — and the sravaani
# tokenizer ranks ൗ as the more frequent one (eval_frozen: 215 ൗ, 0 ൌ). Training targets map
# back to ൗ so they use the token the model actually knows.
def _restore_au_length_mark(text: str) -> str:
    return text.replace("\u0d4c", "\u0d57")


# --- C1: samvruthokaram / final-vowel drift -------------------------------------------
# The "half-u" (samvruthokaram) word ending is written three ways: ...ു് (explicit),
# ...് (bare virama), ...ു (plain u). They are the same ending; canonicalize to the bare
# virama form (...്), which is what most references use. Only fires when the samvruthokaram
# is on a CONSONANT (so we don't strip a real ു vowel). EVAL-ONLY: the final vowel can be
# contrastive, so it is a symmetric scoring fold, not applied to reader-facing output.
_C1_SAMVRUTHU = re.compile("(?<=[ക-ഺ])(?:ു" + _VIRAMA + "|ു)(?=\\s|$)")


def _fold_samvruthokaram(text: str) -> str:
    """word-final consonant + (ു്|ു) -> consonant + ് (canonical samvruthokaram)."""
    return _C1_SAMVRUTHU.sub(_VIRAMA, text)


# Note on numeral + hyphenated suffix (24-ന്, 15-ാം): the linguistically correct fusion
# (24-ന് -> ഇരുപത്തിനാലിന്, 15-ാം -> പതിനഞ്ചാം) was implemented and MEASURED — it made
# WER/CER worse on both real corpora (calls +0.55 WER), because that pattern occurs only in
# the references, never the hypotheses, so fusing it on the ref side only creates new
# mismatches. Convergence beats linguistic correctness here, so it was not shipped. W4 below
# is deliberately digit-safe (leaves digit-adjacent hyphens for E5), which keeps the number
# a clean token E8 can verbalize; the suffix stays a separate token.

# --- W4: intra-word hyphen join (eval-only, WER) --------------------------------------
# A hyphen BETWEEN two non-space, non-digit characters is part of one word ("e-mail",
# "സ്കിൽ-ബേസ്ഡ്", "വൺ-ഓൺ-വൺ"); E5 would turn it into a space and split the token, inflating
# the word count. Remove it so the halves fuse into one token.
# A DIGIT on either side is deliberately EXCLUDED: "24-ന്" / "15-ാം" are numeral + ordinal/
# case-suffix, not a compound word. Joining them fuses the suffix onto the number, which E8
# then verbalizes into one mangled token ("ഇരുപത്തിനാല്ന്") — measured to *hurt* WER on real
# call data. So W4 leaves any digit-adjacent hyphen for E5 to handle as a normal separator.
_W4_HYPHEN = re.compile(r"(?<=[^\s\d-])-(?=[^\s\d-])")


def _join_intra_word_hyphen(text: str) -> str:
    """e-mail -> email, സ്കിൽ-ബേസ്ഡ് -> സ്കിൽബേസ്ഡ് (letter-letter hyphen removed).
    Digit-adjacent hyphens (24-ന്, 5-10) are left for E5 — see the note above."""
    return _W4_HYPHEN.sub("", text)


# --- W3: acronym / letter-run spacing (eval-only, WER) --------------------------------
# "എ ടി എം" / "A T M" (a run of single-LETTER tokens) vs "എടിഎം" / "ATM" are the same
# acronym with different token counts. Collapse a run of >=2 adjacent single-letter tokens
# into one token. A "single letter" is ONE grapheme cluster (Malayalam ടി = ട+ി is two
# codepoints but one letter), or one ASCII letter — never a digit or multi-letter word, so
# a lone real word like "ഒരു" is untouched. Runs on split tokens, so AFTER E5/E10.
def _is_single_letter(tok: str) -> bool:
    """True if tok is exactly one letter (one Malayalam grapheme cluster, or one ASCII letter)."""
    if not tok:
        return False
    if tok.isascii():
        return len(tok) == 1 and tok.isalpha()
    # Malayalam: a base letter (0D00-0D7F) optionally followed by combining vowel signs /
    # virama / au-length mark — all in the same block — with no further base letter.
    base = tok[0]
    if not ("ഀ" <= base <= "ൿ"):
        return False
    for ch in tok[1:]:
        # combining marks that stay part of the same cluster
        if ch in "ാിീുൂൃൄെേൈൊോൌ്ൗൢൣംഃഁ":
            continue
        return False        # a second base letter -> more than one letter
    return True


def _collapse_letter_runs(text: str) -> str:
    """Join runs of >=2 single-letter tokens: 'എ ടി എം' -> 'എടിഎം', 'A T M' -> 'ATM'."""
    toks = text.split()
    out, i, n = [], 0, len(toks)
    while i < n:
        j = i
        while j < n and _is_single_letter(toks[j]):
            j += 1
        if j - i >= 2:                      # a run of >=2 single letters -> one token
            out.append("".join(toks[i:j]))
            i = j
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)


# --- W5: reduplication spacing (eval-only, WER) ---------------------------------------
# Adjacent identical tokens ("വേഗം വേഗം") joined-vs-spaced (or repeat-count) differences
# are boundary noise for WER. Collapse a run of the SAME token to one occurrence. Symmetric
# (applied to ref and hyp alike), so it can only ever under-count a genuine repeat, never
# inflate WER. Runs on split tokens, AFTER E5/E10 (and after W3, so acronym runs are single
# tokens first).
def _collapse_reduplication(text: str) -> str:
    """'വേഗം വേഗം' -> 'വേഗം' (adjacent identical tokens collapsed to one).

    Number words are EXCLUDED: E8 spells id-like digit runs out digit-by-digit, so a
    repeated digit ("007" -> "പൂജ്യം പൂജ്യം ഏഴ്") is meaningful repetition, not
    reduplication — collapsing it would drop a digit. Only non-numeric duplicates fold.
    """
    toks = text.split()
    out = []
    for t in toks:
        if out and out[-1] == t and t not in _ONES:
            continue                    # adjacent duplicate (and not a digit-word) -> drop
        out.append(t)
    return " ".join(out)


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


def _thousands_word(x: int) -> str:
    """x thousand (1-99) as one word: 12 -> പന്ത്രണ്ടായിരം, 20 -> ഇരുപതിനായിരം, 25 ->
    ഇരുപത്തിയഞ്ചായിരം. 1-9 use the irregular _THOUS table (മൂവായിരം, അയ്യായിരം ...)."""
    if x < 10:
        return _THOUS[x - 1]
    if x == 10:
        return "പതിനായിരം"
    w = _num_words(x)
    if w.endswith("ത" + _VIRAMA):          # ...ത് -> ...തിനായിരം (ഇരുപത് -> ഇരുപതിനായിരം)
        return w[:-1] + "ിനായിരം"
    return w[:-1] + "ായിരം"                 # ...് -> ...ായിരം (പന്ത്രണ്ട് -> പന്ത്രണ്ടായിരം)


def _big_num_words(n: int) -> str:
    """0-99,99,999 as spoken Malayalam (Indian grouping). Up to 9999 is _num_words; 10,000-99,999
    fuse the thousands (പന്ത്രണ്ടായിരത്തിയഞ്ഞൂറ്); lakhs are written as a separate word after
    a spaced count, matching how ASR output writes them (രണ്ട് ലക്ഷം, ഒരു ലക്ഷത്തി ...)."""
    if n < 10000:
        return _num_words(n)
    if n < 100000:
        th, r = divmod(n, 1000)
        w = _thousands_word(th)
        return w if r == 0 else _join(w[:-1] + "ത്തി", _num_words(r))
    lk, r = divmod(n, 100000)
    count = "ഒരു" if lk == 1 else _num_words(lk)
    return count + (" ലക്ഷം" if r == 0 else " ലക്ഷത്തി " + _big_num_words(r))


def _verbalize_digits(m: "re.Match[str]") -> str:
    s = m.group(0)
    if len(s) > 7 or (len(s) > 1 and s[0] == "0"):
        return " ".join(_ONES[int(c)] for c in s)   # id-like (phone, account, leading 0): digit by digit
    return _big_num_words(int(s))


# --- Inverse text normalization (ITN): native number words -> digits ------------------
# Built by INVERTING the forward verbalizer, so ITN(verbalize(n)) == n by construction for
# every covered value — no fragile hand-written sandhi parser.
#   _WORD2NUM   fused multi-part forms 0-9999 ("ഇരുപത്തിനാല്" -> 24);
#   _ONE2DIGIT  the single ONES words, for digit-by-digit id-like runs ("ഏഴ്" -> "7").
# These tables back the legacy per-token helpers; the inference pass itself now goes through
# malayalam_itn (_to_digits), which inverts the digits module's generator and also covers
# English words, lakhs/crores, suffixes, halves, percent and the k form directly (no E9 step).
_WORD2NUM = {_num_words(n): n for n in range(0, 10000)}
_ONE2DIGIT = {w: str(i) for i, w in enumerate(_ONES)}

# --- Greedy number-word parser (for fused numeral + suffix, and run-together numbers) ---
# A number in real ASR output is often glued to a following suffix (നാലാം = 4th,
# മുപ്പത്തിയല്ലേ = "thirty, isn't it") or run together (ഇരുനൂറ്റിയെണ്ണൂറ്റി...). The exact
# _WORD2NUM map only matches a whole standalone token, so those slip through. This parser
# walks a token left-to-right, matching number-word UNITS (reusing the forward tables),
# summing their values exactly as _num_words composes them, and stops at the first
# non-number remainder — which is kept as the suffix. Guarded so a token that does NOT begin
# with a number unit (ഫ്രണ്ടിൽ "front", ആയിരിക്കും "will be") is never touched.
_SIGN2VOWEL = {v: k for k, v in _V_SIGN.items() if v}   # vowel-sign -> independent vowel
_ORDINAL_SUFFIX = re.compile("^ാ(?:ം|മ)")               # ordinal ആം / ആമത്(തെ|ായി...)
_VSIGNS = set("ാിീുൂൃെേൈൊോൌ")                            # dependent vowel signs


def _number_units():
    u = {}
    for i, w in enumerate(_ONES):    u[w] = i
    for i, w in enumerate(_TEENS):   u[w] = 10 + i
    for i, w in enumerate(_TENS):    u[w] = (i + 2) * 10
    for i, w in enumerate(_TENS_C):  u[w] = (i + 2) * 10
    for i, w in enumerate(_HUNDS):   u[w] = (i + 1) * 100
    for i, w in enumerate(_HUNDS_C): u[w] = (i + 1) * 100
    for i, w in enumerate(_THOUS):   u[w] = (i + 1) * 1000
    for i, w in enumerate(_THOUS_C): u[w] = (i + 1) * 1000
    return u


_NUM_UNITS = _number_units()
_NUM_UKEYS = sorted(_NUM_UNITS, key=len, reverse=True)   # longest-first greedy


def _match_unit_at(s: str, allow_stem: bool) -> "tuple[int, int, bool] | None":
    """Longest number unit at the start of ``s`` -> (value, chars_consumed_in_s, is_stem), or
    None. Handles the sandhi glide (യ before a vowel-initial unit) and, when ``allow_stem``, a
    unit whose trailing samvruthokaram ് was replaced by a directly-attached vowel sign (a stem
    match ends the numeral: what follows is a suffix)."""
    def _stem(t):                                 # unit whose final ് became an attached vowel sign (നാല് -> നാലാം)
        for k in _NUM_UKEYS:
            if (k.endswith(_VIRAMA) and len(k) >= 2 and t.startswith(k[:-1])
                    and len(t) > len(k) - 1 and t[len(k) - 1] in _VSIGNS):
                return k
        return None

    for k in _NUM_UKEYS:
        if s.startswith(k):
            return _NUM_UNITS[k], len(k), False
    if s.startswith("യ"):                         # glide: യ + (vowel-sign) stood for an അ-/vowel-initial unit
        rest = s[1:]
        if rest and rest[0] in _SIGN2VOWEL:
            cand, glide = _SIGN2VOWEL[rest[0]] + rest[1:], 2
        else:
            cand, glide = "അ" + rest, 1
        for k in _NUM_UKEYS:
            if cand.startswith(k):
                return _NUM_UNITS[k], glide + (len(k) - 1), False
        if allow_stem:                            # glide + stem (ഇരുപത്തി|യൊന്നാം = 21st)
            k = _stem(cand)
            if k:
                return _NUM_UNITS[k], glide + len(k) - 2, True
    if allow_stem:
        k = _stem(s)
        if k:
            return _NUM_UNITS[k], len(k) - 1, True
    return None


def _parse_leading_number(core: str) -> "tuple[int, int] | None":
    """(total_value, chars_consumed) for the numeral at the start of ``core``, or None.
    Greedy additive composition mirroring _num_words. Full/glide matches are always tried
    first (so well-formed numbers round-trip exactly); a stem match is the fallback and ends
    the numeral, leaving the attached vowel (e.g. the ordinal ാം) as the suffix."""
    i = total = 0
    matched = False
    while i < len(core):
        m = _match_unit_at(core[i:], allow_stem=True)
        if not m:
            break
        val, clen, is_stem = m
        total += val
        i += clen
        matched = True
        if is_stem:
            break
    return (total, i) if matched else None


def _unglide(s: str) -> str:
    """Restore a suffix's leading independent vowel if it was written as the glide യ(+sign)."""
    if s.startswith("യ"):
        rest = s[1:]
        if rest and rest[0] in _SIGN2VOWEL:
            return _SIGN2VOWEL[rest[0]] + rest[1:]
        return "അ" + rest
    return s


# Peel leading/trailing punctuation off a token so a number word adjacent to punctuation
# (e.g. "ഇരുപത്തിനാല്," or "(ഏഴ്)") still matches. Inference keeps punctuation, so the
# number passes must look at the token's CORE and reattach the surrounding punctuation.
_LEAD_PUNCT = re.compile(r"^([^\wഀ-ൿ]*)")
_TRAIL_PUNCT = re.compile(r"([^\wഀ-ൿ]*)$")


def _peel(tok: str):
    """(leading_punct, core, trailing_punct) for a token."""
    lead = _LEAD_PUNCT.match(tok).group(1)
    trail = _TRAIL_PUNCT.search(tok).group(1)
    core = tok[len(lead): len(tok) - len(trail)] if (len(lead) + len(trail)) <= len(tok) else ""
    return lead, core, trail


def _itn_digit_by_digit(tokens: list) -> list:
    """Collapse a run of >=2 consecutive single-digit ONES words (by CORE) into one digit
    string (reverses the id-like path: 'പൂജ്യം പൂജ്യം ഏഴ്' -> '007'). A run is only fused
    when the interior tokens carry no punctuation, so '..., ഏഴ്' doesn't glue across commas."""
    cores = [_peel(t) for t in tokens]
    out, i, n = [], 0, len(tokens)
    while i < n:
        j = i
        while j < n and cores[j][1] in _ONE2DIGIT:
            j += 1
        run = tokens[i:j]
        if len(run) >= 2:
            digits = "".join(_ONE2DIGIT[cores[k][1]] for k in range(i, j))
            out.append(cores[i][0] + digits + cores[j - 1][2])   # keep outer punctuation
            i = j
        else:
            out.append(tokens[i]); i += 1
    return out


def _itn_token(tok: str) -> str:
    """One token -> digits; surrounding punctuation preserved. A pure numeral becomes digits
    (ഇരുപത്തിനാല് -> 24); a numeral fused with a suffix becomes ``digits-suffix`` (നാലാം ->
    4-ാം, മുപ്പത്തിയല്ലേ -> 30-അല്ലേ); a token that doesn't start with a number is unchanged."""
    lead, core, trail = _peel(tok)
    if not core:
        return tok
    r = _parse_leading_number(core)
    if r is None:
        return tok
    value, consumed = r
    suffix = _unglide(core[consumed:])
    # The numeral spoken with its own -u ending (രണ്ടു, ആറു, പത്തു) is just the number.
    if suffix in ("ു", "ു" + _VIRAMA):
        suffix = ""
    # ORDINALS ONLY: a fused suffix is converted only when it is the ordinal ആം family
    # (ാം / ാമത് / ാമത്തെ / ാമതായി ...): പത്താം -> 10-ാം, രണ്ടാമത്തെ -> 2-ാമത്തെ.
    # Any other fusion is a word, not a numeral, and is left untouched — measured on the
    # corpus, numeral-prefix idioms dominate: ഒന്നും "anything/nothing", രണ്ടും "both",
    # ഒന്നിച്ച് "together", ഒന്നുകിൽ "either", രണ്ടുമൂന്ന് "two or three".
    if suffix and not _ORDINAL_SUFFIX.match(suffix):
        return tok
    body = str(value) + ("-" + suffix if suffix else "")
    return lead + body + trail


def _to_digits(text: str) -> str:
    """ITN: number words (native in every spelling, English in Malayalam script) -> digits.
    The inference pass; see malayalam_itn (built by inverting the digits module's readings)."""
    from .malayalam_itn import itn
    return itn(text)


# --- English-style number words (Malayalam script, as base models write code-mixed
# numbers) -> value -> the same canonical Malayalam words as digits get. Spelling
# variants collected from real base-model output. --------------------------------------
# Digit-group commas, Western (70,000 / 1,000,000) AND Indian lakh grouping (1,75,000 /
# 12,50,000) -> one digit run, BEFORE _PUNCT splits it. A list like "1,2,3" is not a grouping.
_GROUP_COMMA = re.compile(r"(?<![\d,])\d{1,3}(?:(?:,\d{2})*,\d{3}|(?:,\d{3})+)(?!,?\d)")


def _drop_group_commas(text: str) -> str:
    return _GROUP_COMMA.sub(lambda m: m.group(0).replace(",", ""), text)
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


# indic-nlp-library is a hard dependency (see pyproject.toml). It canonicalizes chillu /
# ZWJ, two-part vowel signs, and colon→visarga.
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

_IND_NORM = IndicNormalizerFactory().get_normalizer("ml")

# Kept for backward compatibility with importers that read this flag; always True now.
INDIC_NLP_AVAILABLE = True


# Accepted mode names. Three behaviours:
#   eval      — WER/CER scoring: canonicalize, and verbalize digits to native Malayalam words
#               (24 -> ഇരുപത്തിനാല്) so digit references meet word hypotheses.
#   train     — training / fine-tune targets: same canonicalization as eval but numbers are
#               NOT rewritten (digit verbalization is done outside this library).
#   inference — reader-facing ITN: keep punctuation and case, number words -> digits.
_EVAL_MODES = {"eval", "evaluation"}
_TRAIN_MODES = {"train", "training", "fine-tune", "finetune"}
_INFER_MODES = {"inference", "infer", "itn"}


def _resolve_mode(mode) -> str:
    m = str(mode or "eval").strip().lower()
    if m in _INFER_MODES:
        return "inference"
    if m in _EVAL_MODES:
        return "eval"
    if m in _TRAIN_MODES:
        return "train"
    raise ValueError(
        f"unknown mode {mode!r}; use 'eval', 'train' (aliases: training, fine-tune) or 'inference'"
    )


# I1 (inference): a standalone ഒന്ന്/ഒന്നു is far more often the softener "just / once"
# (ഒന്ന് നോക്ക് "just look", ഒന്ന് ചെക്ക് ചെയ്യ്, ഒന്ന് പറഞ്ഞു) than the number 1. Convert it
# only in NUMERIC CONTEXT: a neighbouring token is a number (digits, a native or English number
# word), or the next token is a unit/counter (രൂപ, ദിവസം, വർഷം, ഇയർ, ലക്ഷം ...). Otherwise the
# word is held in a placeholder across E9/ITN and restored unchanged.
_ONE_HOLD, _ONEU_HOLD = "\ue010", "\ue011"
_UNIT_PREFIXES = ("രൂപ", "ദിവസ", "മാസ", "വർഷ", "ആഴ്ച", "മണി", "മിനിറ്റ", "മിനിട്ട", "സെക്കൻ",
                  "കിലോ", "ഗ്രാം", "ലിറ്റ", "മീറ്റ", "ലക്ഷ", "ആയിര", "കോടി", "ശതമാന", "പേര",
                  "പേർ", "തവണ", "പ്രാവശ്യ", "എണ്ണ", "ഇയർ", "മന്ത്", "ലാക്", "ലാഖ്", "ഡേയ്",
                  "വീക്ക്", "അവർ", "പെർസ", "തൗസ", "ഹൺഡ്ര", "ഹണ്ട്ര", "മില്യ", "പോയിന്റ")
_UNIT_EXACT = {"ഡേ", "ഡേസ്", "ക്ലാസ്"}


def _is_numberish(core: str) -> bool:
    if not core:
        return False
    if core[0].isdigit() or core in _ONE2DIGIT or core in _WORD2NUM or core in _EN_ANY:
        return True
    from . import malayalam_itn as _itn          # every spelling the digits module writes
    if core in _itn._combining() or _itn._word_info(core) is not None or _itn.split_suffix(core):
        return True
    return core.endswith("ു") and (core[:-1] + _VIRAMA) in _WORD2NUM   # രണ്ടു, ആറു


def _hold_lone_one(text: str) -> str:
    toks = text.split()
    cores = [_peel(t)[1] for t in toks]
    for i, c in enumerate(cores):
        if c not in ("ഒന്ന്", "ഒന്നു"):
            continue
        prev = cores[i - 1] if i > 0 else ""
        nxt = cores[i + 1] if i + 1 < len(cores) else ""
        if (_is_numberish(prev) or _is_numberish(nxt) or nxt in _UNIT_EXACT
                or nxt.startswith(_UNIT_PREFIXES)):
            continue
        toks[i] = toks[i].replace(c, _ONE_HOLD if c == "ഒന്ന്" else _ONEU_HOLD, 1)
    return " ".join(toks)


# Colon -> visarga guard. indic-nlp rewrites ANY ':' that follows a Malayalam letter as the
# visarga ഃ. That is right only INSIDE a word, where ':' is typed for ഃ (ദു:ഖം -> ദുഃഖം). A colon
# followed by a space / end / non-letter is punctuation (ആണ്: ..., പറ:) and must stay ':' —
# eval then strips it, inference keeps it. Such colons are swapped for a private-use placeholder
# around the indic-nlp call so it cannot touch them.
_COLON_PUNCT = re.compile(r":(?![\u0d00-\u0d7f])")

# E13: bracketed annotation tags in references (<unintelligible>, <noise>, <laugh> ...). E5 would
# strip the brackets and leave the tag NAME as a spoken word; drop the whole tag instead.
# (<unk> is E3.) Not applied in train mode, where tags are handled by the data pipeline.
_TAG = re.compile(r"<[A-Za-z][A-Za-z _-]*>")

# T1 (train only): punctuation INSIDE a number is meaning, not punctuation — an external
# verbalizer needs 4:00-5:00 (time range), 2.5 (decimal), 70% (percent), 23-ന് (numeral +
# case suffix), 8-9 (range). Hold those characters in private-use placeholders across E5.
_NUM_PUNCT = re.compile(r"(?<=\d)[:.\-/+](?=\d)|(?<=\d)%|(?<=\d)-(?=[\u0d00-\u0d7f])")   # 3+2 kept too
_NUM_PUNCT_HOLD = {":": "\ue001", ".": "\ue002", "-": "\ue003", "/": "\ue004", "%": "\ue005", "+": "\ue006"}
_NUM_PUNCT_RESTORE = str.maketrans({v: k for k, v in _NUM_PUNCT_HOLD.items()})


def _protect_num_punct(text: str) -> str:
    return _NUM_PUNCT.sub(lambda m: _NUM_PUNCT_HOLD[m.group(0)], text)
_COLON_HOLD = "\ue000"


def normalize(text: str, mode: str = "eval", segmentation: bool = False) -> str:
    """Normalize Malayalam ASR text; see module docstring and NORMALIZATION.md (mode selector).

    mode="eval" (default; alias "evaluation") — WER/CER scoring: strip punctuation, lowercase,
    and verbalize digits to native Malayalam words (24 -> ഇരുപത്തിനാല്). English-style number
    words (ട്വൻറ്റി ഫോർ) are left as written.
    mode="train" (aliases "training", "fine-tune") — training targets: like eval, but numbers
    are NOT rewritten (digit verbalization is handled outside this library), punctuation inside
    a number is kept for that verbalizer (4:00-5:00, 2.5, 70%, 23-ന്, 8-9), the C1 fold is off
    (targets keep their real spelling) and <tags> are left to the data pipeline.
    mode="inference" — reader-facing ITN: keep punctuation and case; number words -> digits
    (ഇരുപത്തിനാല് -> 24, ട്വൻറ്റി ഫോർ -> 24, നാലാം -> 4-ാം); existing digits are left untouched.

    segmentation (eval/train only, default OFF) — apply the boundary rules W4 (intra-word hyphen
    join), W3 (single-letter acronym-run collapse) and W5 (adjacent-duplicate collapse). They
    are OFF by default because on the measured corpora they are net-neutral-to-slightly-
    negative (the boundary variants they target are rare, and each occasionally introduces a
    new mismatch); enable when a corpus is known to benefit. No effect in inference mode.

    The CER folds C6 (homoglyph ഠ->0 in a number) and C2 (word-final മ് -> ം) run in BOTH modes
    (both unambiguous). C1 (word-final samvruthokaram ...ു്/...ു -> ...്) runs in EVAL ONLY — it
    is a symmetric scoring fold that can also touch a genuine -u vowel, so it is not applied to
    reader-facing inference output. Together they measurably help WER/CER (dataset WER -1.09).
    """
    mode = _resolve_mode(mode)

    # --- Shared canonicalization base (identical in both modes) ---
    text = _fix_chillu_conjuncts(text or "")      # E12: ൻ്റ -> ന്റ (mis-encoded conjunct, NFC-invisible)
    text = _COLON_PUNCT.sub(_COLON_HOLD, text)    #      protect punctuation colons (not ദു:ഖം)
    text = _IND_NORM.normalize(text)              # E2:  chillu / ZWJ / two-part vowels / mid-word ':' -> ഃ
    text = text.replace(_COLON_HOLD, ":")
    text = unicodedata.normalize("NFC", text)     # E1
    text = _UNK.sub(" ", text)                    # E3:  <unk>/⁇ removed in BOTH modes
    text = _ZW.sub("", text)                      # E4
    text = "".join(_DIGITS.get(c, c) for c in text)  # E6:  Indic digit -> ASCII (both modes)
    text = _fix_homoglyphs(text)                  # C6:  ഠ used as zero in a number -> 0 (both modes)
    if mode == "train":
        text = _restore_au_length_mark(text)      # T2:  ൌ -> ൗ (undo indic-nlp; train only)
        text = _fold_final_nasal_train(text)      # C2:  word-final മ് -> ം, not after a virama
    else:
        text = _fold_final_nasal(text)            # C2:  word-final മ് -> ം
    text = _drop_group_commas(text)               # E7:  70,000 / 1,75,000 -> one digit run (all modes)
    if mode != "train":
        text = _TAG.sub(" ", text)                # E13: <unintelligible>-style tags (eval + inference)

    if mode in ("eval", "train"):
        if mode == "train":
            text = _protect_num_punct(text)            # T1:  4:00-5:00, 2.5, 70%, 23-ന്, 8-9 survive E5
        if segmentation:
            text = _join_intra_word_hyphen(text)       # W4:  e-mail -> email (BEFORE E5 eats the hyphen)
        text = _PUNCT.sub(" ", text)                   # E5:  strip punctuation
        if mode == "train":
            text = text.translate(_NUM_PUNCT_RESTORE)
        if mode == "eval":                             # E8 (eval only; train leaves digits)
            text = _NUM_RE.sub(_verbalize_digits, text)  #      digits -> native words (24 -> ഇരുപത്തിനാല്)
        text = text.lower()                            # E10: lowercase
        if mode == "eval":
            text = _fold_samvruthokaram(text)          # C1:  word-final ...ു്/...ു -> ...് (eval only: a
                                                       #      training target must keep the real spelling)
        if segmentation:
            text = _collapse_letter_runs(text)         # W3:  "എ ടി എം" -> "എടിഎം" (single-letter runs)
            text = _collapse_reduplication(text)       # W5:  "വേഗം വേഗം" -> "വേഗം" (adjacent identical)
        return _SPACE.sub(" ", text).strip()           # E11

    # --- inference: reader-facing ITN (keep E5 punctuation + E10 case; number words -> digits) ---
    # Existing digits are left exactly as written (4:00-5:00 stays 4:00-5:00). One ITN pass
    # (malayalam_itn) turns every number WORD into digits: native in every spelling the digits
    # module writes, English in Malayalam script, lakhs / crores, suffixes, decimals, percent.
    text = _hold_lone_one(text)                    # I1:  standalone ഒന്ന് = "just/once", not 1
    text = _to_digits(text)                        # ITN: number words -> digits
    text = text.replace(_ONE_HOLD, "ഒന്ന്").replace(_ONEU_HOLD, "ഒന്നു")
    return _SPACE.sub(" ", text).strip()           # E11 (punctuation + case preserved)
