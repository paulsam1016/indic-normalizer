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


def _verbalize_digits(m: "re.Match[str]") -> str:
    s = m.group(0)
    if len(s) > 4 or (len(s) > 1 and s[0] == "0"):
        return " ".join(_ONES[int(c)] for c in s)   # id-like: digit by digit
    return _num_words(int(s))


# --- Inverse text normalization (ITN): native number words -> digits ------------------
# Built by INVERTING the forward verbalizer, so ITN(verbalize(n)) == n by construction for
# every covered value — no fragile hand-written sandhi parser.
#   _WORD2NUM   fused multi-part forms 0-9999 ("ഇരുപത്തിനാല്" -> 24);
#   _ONE2DIGIT  the single ONES words, for digit-by-digit id-like runs ("ഏഴ്" -> "7").
# In inference mode English-style number words are first resolved to native words upstream
# (E9 still runs), so by the time _to_digits sees the text EVERY number is a native fused
# form or an ASCII run — one place converts them all to digits.
_WORD2NUM = {_num_words(n): n for n in range(0, 10000)}
_ONE2DIGIT = {w: str(i) for i, w in enumerate(_ONES)}


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
    """One token -> digits if its core is a (fused) native number word, else unchanged;
    surrounding punctuation is preserved."""
    lead, core, trail = _peel(tok)
    if core in _WORD2NUM:
        return lead + str(_WORD2NUM[core]) + trail
    if core in _ONE2DIGIT:
        return lead + _ONE2DIGIT[core] + trail
    return tok


def _to_digits(text: str) -> str:
    """ITN: rewrite native Malayalam number words back to digits.

    Handles fused numeric words (24 <- ഇരുപത്തിനാല്), spaced digit-by-digit id runs
    (007 <- പൂജ്യം പൂജ്യം ഏഴ്), and number words adjacent to preserved punctuation
    (24, <- ഇരുപത്തിനാല്,); non-number tokens are left untouched.
    """
    tokens = _itn_digit_by_digit(text.split())
    return " ".join(_itn_token(t) for t in tokens)


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


# indic-nlp-library is a hard dependency (see pyproject.toml). It canonicalizes chillu /
# ZWJ, two-part vowel signs, and colon→visarga.
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

_IND_NORM = IndicNormalizerFactory().get_normalizer("ml")

# Kept for backward compatibility with importers that read this flag; always True now.
INDIC_NLP_AVAILABLE = True


# Accepted mode names. Two behaviours: eval (model-facing, verbalize) and inference
# (reader-facing, ITN). "training"/"fine-tune" are aliases of eval.
_EVAL_MODES = {"eval", "evaluation", "training", "train", "fine-tune", "finetune"}
_INFER_MODES = {"inference", "infer", "itn"}


def _resolve_mode(mode) -> str:
    m = str(mode or "eval").strip().lower()
    if m in _INFER_MODES:
        return "inference"
    if m in _EVAL_MODES:
        return "eval"
    raise ValueError(
        f"unknown mode {mode!r}; use 'eval' (aliases: training, fine-tune) or 'inference'"
    )


def normalize(text: str, mode: str = "eval", segmentation: bool = False) -> str:
    """Normalize Malayalam ASR text; see module docstring and NORMALIZATION.md (mode selector).

    mode="eval" (default; aliases "training"/"fine-tune") — model-facing: strip punctuation,
    lowercase, and verbalize every number to native Malayalam words (24 -> ഇരുപത്തിനാല്).
    mode="inference" — reader-facing ITN: keep punctuation and case, and render every number
    as digits (ഇരുപത്തിനാല് -> 24, ട്വൻറ്റി ഫോർ -> 24).

    segmentation (eval only, default OFF) — apply the boundary rules W4 (intra-word hyphen
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
    text = _IND_NORM.normalize(text)              # E2:  chillu / ZWJ / two-part vowels / visarga
    text = unicodedata.normalize("NFC", text)     # E1
    text = _UNK.sub(" ", text)                    # E3:  <unk>/⁇ removed in BOTH modes
    text = _ZW.sub("", text)                      # E4
    text = "".join(_DIGITS.get(c, c) for c in text)  # E6:  Indic digit -> ASCII (both modes)
    text = _fix_homoglyphs(text)                  # C6:  ഠ used as zero in a number -> 0 (both modes)
    text = _fold_final_nasal(text)                # C2:  word-final മ് -> ം (both modes)
    text = _GROUP_COMMA.sub("", text)             # 70,000 -> 70000 (both: keeps a number one token)

    if mode == "eval":
        if segmentation:
            text = _join_intra_word_hyphen(text)       # W4:  e-mail -> email (BEFORE E5 eats the hyphen)
        text = _PUNCT.sub(" ", text)                   # E5:  strip punctuation
        text = _canonicalize_en_numbers(text)          # E9:  "ട്വൻറ്റി ഫോർ" -> native words
        text = _NUM_RE.sub(_verbalize_digits, text)    # E8:  digits -> native words (OOV-safe targets)
        text = text.lower()                            # E10: lowercase
        text = _fold_samvruthokaram(text)              # C1:  word-final ...ു്/...ു -> ...് (eval only)
        if segmentation:
            text = _collapse_letter_runs(text)         # W3:  "എ ടി എം" -> "എടിഎം" (single-letter runs)
            text = _collapse_reduplication(text)       # W5:  "വേഗം വേഗം" -> "വേഗം" (adjacent identical)
        return _SPACE.sub(" ", text).strip()           # E11

    # --- inference: reader-facing ITN (keep E5 punctuation + E10 case; numbers -> digits) ---
    # E9 still resolves English-style number words to native words, so the single ITN pass
    # below turns EVERY number (native or English-derived) into digits.
    text = _canonicalize_en_numbers(text)          # E9:  "ട്വൻറ്റി ഫോർ" -> "ഇരുപത്തിനാല്"
    text = _NUM_RE.sub(_verbalize_digits, text)    # normalize any stray ASCII digits to canonical words...
    text = _to_digits(text)                        # E8⁻¹: ...then ITN everything back to digits
    return _SPACE.sub(" ", text).strip()           # E11 (punctuation + case preserved)
