"""Shared normalization primitives for Indic-script ASR scoring.

These are the language-agnostic building blocks every normalizer reuses, so the
per-language modules only add their script-specific rules on top. All steps are
meaning-preserving and safe to apply to BOTH reference and hypothesis (see
NORMALIZATION.md for the design rationale and the Whisper "vanishing vowels" pitfall
this deliberately avoids — we remove an explicit punctuation set, never a Unicode
category, so vowel signs / virama are preserved).

Pure stdlib; no third-party dependency.
"""

import re
import unicodedata

# ``⁇`` / ``<unk>`` OOV markers (Soniox and some base models emit these).
UNK = re.compile(r"⁇|<unk>")

# Invisible formatting chars: ZWNJ, ZWJ, BOM, U+FFFE, word joiner, soft hyphen. Drop AFTER any
# chillu/ZWJ fix that relies on the ZWJ still being present.
ZERO_WIDTH = re.compile("[\u200c\u200d\ufeff\ufffe\u2060\u00ad]")
# Zero-width SPACE marks a word boundary: it becomes a space, not nothing.
ZWSP = "\u200b"

# Punctuation + symbols to strip, ASCII AND Indic (danda ।॥, curly quotes, dashes,
# ellipsis). An EXPLICIT set — never a Unicode Mark/Symbol/Punctuation *category*
# filter, which would delete Indic vowel signs and the virama.
PUNCT = re.compile(r"[.,!?;:\"'`~^*_=+/\\|<>(){}\[\]@#$%&…।॥“”‘’«»\-–—]")

SPACE = re.compile(r"\s+")

# An apostrophe between two Latin letters joins the word (it's -> its, can't -> cant,
# o'clock -> oclock) instead of splitting it in two (it s). Straight, curly, modifier-letter and
# backtick forms. Any other apostrophe (quotes, students') is punctuation.
_LATIN = "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f"
WORD_APOSTROPHE = re.compile(f"(?<=[{_LATIN}])['\u2019\u2018\u02bc`](?=[{_LATIN}])")

# Digit-grouping commas, Western and Indian lakh (70,000 / 1,75,000 -> one digit run). A
# 1,2,3 list is not a grouped number and is left alone.
GROUP_COMMA = re.compile(r"(?<![\d,])\d{1,3}(?:(?:,\d{2})*,\d{3}|(?:,\d{3})+)(?!,?\d)")

# Mode names shared by every normalizer (see NORMALIZATION.md "Mode selector").
EVAL_MODES = {"eval", "evaluation"}
TRAIN_MODES = {"train", "training", "fine-tune", "finetune"}
INFER_MODES = {"inference", "infer", "itn"}


def resolve_mode(mode) -> str:
    """'eval' / 'train' / 'inference' for a mode name or alias; ValueError otherwise."""
    m = str(mode or "eval").strip().lower()
    if m in INFER_MODES:
        return "inference"
    if m in EVAL_MODES:
        return "eval"
    if m in TRAIN_MODES:
        return "train"
    raise ValueError(
        f"unknown mode {mode!r}; use 'eval', 'train' (aliases: training, fine-tune) or "
        "'inference' (aliases: infer, itn)"
    )


def to_nfc(text: str) -> str:
    """Unicode NFC — one canonical codepoint sequence per grapheme."""
    return unicodedata.normalize("NFC", text or "")


def strip_unk(text: str) -> str:
    """Replace ``⁇`` / ``<unk>`` markers with a space (removed from both sides)."""
    return UNK.sub(" ", text)


def strip_zero_width(text: str) -> str:
    """Zero-width space -> space; remove ZWNJ / ZWJ / BOM / word joiner / soft hyphen."""
    return ZERO_WIDTH.sub("", text.replace(ZWSP, " "))


def indian_grouping(n: int) -> str:
    """How ITN writes a whole number: plain below one lakh (1000, 50000), Indian grouping from
    one lakh up (1,00,000 / 1,20,000 / 1,00,00,000)."""
    s = str(n)
    if n < 100_000:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head] + groups + [tail])


def western_grouping(n: int) -> str:
    """How ITN writes a number said with million / billion / trillion: 2,000,000 / 1,200,000."""
    return f"{n:,}"


def drop_group_commas(text: str) -> str:
    """70,000 / 1,75,000 -> 70000 / 175000 (a 1,2,3 list is untouched)."""
    return GROUP_COMMA.sub(lambda m: m.group(0).replace(",", ""), text)


def fold_digits(text: str) -> str:
    """Map every Unicode *decimal* digit to ASCII 0-9 (Devanagari, Malayalam, …).

    Uses the Unicode decimal property, so it covers all scripts uniformly and leaves
    non-digit characters untouched.
    """
    out = []
    for ch in text:
        d = unicodedata.decimal(ch, None)
        out.append(str(d) if d is not None else ch)
    return "".join(out)


# A currency sign with its amount (eval): ₹500 / Rs. 500 / INR 500 / $24 / €10 / £5, and 500₹.
# Group commas are already dropped (E7) when this runs.
_AMOUNT = r"(\d+(?:\.\d+)?)"
CURRENCY_BEFORE = re.compile(r"(?i)(?<![\w.])(₹|rs\.?|inr|\$|€|£)\s?" + _AMOUNT + r"(?![\d.]*\d)")
CURRENCY_AFTER = re.compile(_AMOUNT + r"\s?₹(?!\w)")
CURRENCY_SIGN = re.compile("[₹€£¥]")             # a sign left without an amount
PERCENT_SIGN = re.compile(_AMOUNT + r"\s?%")      # 20% / 2.5 % (eval: -> 20 percent-word)
CURRENCY_KEY = {"₹": "inr", "rs": "inr", "rs.": "inr", "inr": "inr", "$": "usd", "€": "eur", "£": "gbp"}


def currency_to_words(text: str, names: dict) -> str:
    """Sign + amount -> amount + word, for eval (both sides then meet a spoken "500 രൂപ" /
    "500 rupees" / "20 ശതമാനം"). ``names[key]`` is (singular, plural) for key inr / usd / eur /
    gbp, and pct for a percent sign; the singular is used for an amount of exactly 1."""
    def word(key, amount):
        one, many = names[key]
        return one if amount in ("1", "1.0", "1.00") else many

    text = CURRENCY_BEFORE.sub(lambda m: f"{m.group(2)} {word(CURRENCY_KEY[m.group(1).lower()], m.group(2))}", text)
    text = CURRENCY_AFTER.sub(lambda m: f"{m.group(1)} {word('inr', m.group(1))}", text)
    if "pct" in names:                                 # 20% -> 20 ശതമാനം / 20 percent
        text = PERCENT_SIGN.sub(lambda m: f"{m.group(1)} {names['pct'][1]} ", text)
    return text


def join_apostrophes(text: str) -> str:
    """it's -> its, can't -> cant: an apostrophe inside a Latin word is removed, not spaced."""
    return WORD_APOSTROPHE.sub("", text)


def strip_punct(text: str) -> str:
    """Replace ASCII + Indic punctuation/symbols with a space."""
    return PUNCT.sub(" ", text)


def collapse_space(text: str) -> str:
    """Collapse runs of whitespace to a single space and trim."""
    return SPACE.sub(" ", text).strip()
