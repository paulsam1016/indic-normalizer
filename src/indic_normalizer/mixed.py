"""Code-mixed text: other scripts inside text whose main language is ``lang``.

A normalizer has three tiers:

* shared  — script-agnostic steps every language runs (``base``: NFC, invisible characters,
  ``⁇`` / ``<unk>``, digit folding, group commas, whitespace);
* main    — the rules of ``lang`` itself, for everything written in its script (for Malayalam
  that includes English loanwords in Malayalam script, ട്വന്റി ഫോർ: Malayalam text as written);
* mixed   — one handler per OTHER script found in the text, with its own options (this module).

Handlers do not run on their own: the main normalizer calls them at the right step and passes
its context (e.g. its unit words, so "five മിനിറ്റ്" reads as 5 മിനിറ്റ്).

``mixed`` (the argument of ``normalize``) is:

* ``True`` (default) — every handler, default options;
* ``False`` / ``None`` — none: other scripts get the shared steps only;
* a dict per handler — ``{"en": {"spell_small": True}}`` sets options, ``{"en": False}`` turns
  one off (handlers not named keep their defaults).

Handlers today: ``en`` (English in Latin script).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from typing import Callable, Dict, Optional

UnitWord = Optional[Callable[[str], bool]]


@dataclass(frozen=True)
class EnglishOptions:
    # inference ITN: English style for lone small numbers ("I have five" stays); False reads
    # them like the main language's ITN ("I have 5")
    spell_small: bool = False
    # eval: canonical forms (contractions expanded, fillers dropped, OK -> okay, & -> and)
    canonical: bool = True


_A = "['\u2019\u2018\u02bc`]"                     # an apostrophe, any form
_CONTRACTIONS = [(re.compile(p, re.I), r) for p, r in (
    (rf"\bcan{_A}t\b", "can not"), (r"\bcannot\b", "can not"),
    (rf"\bwon{_A}t\b", "will not"), (rf"\bshan{_A}t\b", "shall not"),
    (rf"\b([a-z]+)n{_A}t\b", r"\1 not"),              # don't, isn't, didn't, couldn't ...
    (rf"\blet{_A}s\b", "let us"),
    # 's after these pronouns is "has" before been / got (it's been), else "is"; after any other
    # word it is a possessive (John's) and is left to the apostrophe join
    (rf"\b(it|that|what|there|here|he|she|who|where|how|this){_A}s(\s+(?:been|got)\b)", r"\1 has\2"),
    (rf"\b(it|that|what|there|here|he|she|who|where|how|this)\s?{_A}s\b", r"\1 is"),
    (rf"\b([a-z]+){_A}re\b", r"\1 are"), (rf"\b([a-z]+){_A}ll\b", r"\1 will"),
    (rf"\b([a-z]+){_A}ve\b", r"\1 have"), (rf"\bi{_A}m\b", "i am"),
    (rf"\b([a-z]+){_A}d\b", r"\1 would"),
    (r"\bgonna\b", "going to"), (r"\bwanna\b", "want to"), (r"\bgotta\b", "got to"),
)]
# um / uh / hmm / mm (and longer spellings) are hesitations; "uh huh" (yes) and "mhm" are kept
_FILLER = re.compile(r"(?i)\b(?:u+m+|u+h+|h+m+|m{2,})\b(?![\s-]+huh\b)")
_OKAY = re.compile(r"(?i)\b(?:ok|okey|o\.k\.?)(?![\w.])")
_AND = re.compile(r"\s*&\s*")


def _canonical(text: str) -> str:
    for rx, rep in _CONTRACTIONS:
        text = rx.sub(rep, text)
    text = _FILLER.sub(" ", text)
    return _OKAY.sub("okay", text)


class English:
    """English in Latin script. Inference: number words -> digits (english_itn). Eval, in a
    mostly-English sentence (or all of the text, ``every=True``, when English is the main
    language): numbers are spelled out the way the digits module writes training targets
    (24 -> twenty four, 2.5 -> two point five, 21st -> twenty first, 4:05 -> four oh five,
    1995 -> nineteen ninety five), currency signs and percent become words (₹500 -> five hundred
    rupees), and the canonical forms apply (can't -> can not, um dropped, OK -> okay, & -> and);
    other sentences are left to the main language."""

    name = "en"
    Options = EnglishOptions

    def itn(self, text: str, opts: EnglishOptions, unit_word: UnitWord = None) -> str:
        from .languages.english_itn import itn
        return itn(text, spell_small=opts.spell_small, unit_word=unit_word)

    # a sentence ends at . ! ? । ॥ followed by a space or the end (not inside 2.5 / 2.0.1), or a newline
    _SENTENCE = re.compile(r".*?(?:[.!?।॥]+(?=\s|$)|\n|$)", re.S)

    def _per_english_sentence(self, text: str, fn, every: bool = False) -> str:
        if every:
            return fn(text)
        from .digits import latin
        out = []
        for m in self._SENTENCE.finditer(text):
            seg = m.group(0)
            out.append(fn(seg) if seg and latin.is_english(seg) else seg)
        return "".join(out)

    @staticmethod
    def _spell(seg: str) -> str:
        from .digits import verbalize
        return verbalize(seg, latin=True)

    def verbalize(self, text: str, opts: EnglishOptions, every: bool = False) -> str:
        """Eval: numbers -> English words, as the digits module writes them (first reading)."""
        return self._per_english_sentence(text, self._spell, every)

    CURRENCY = {"inr": ("rupee", "rupees"), "usd": ("dollar", "dollars"), "eur": ("euro", "euros"),
                "gbp": ("pound", "pounds"), "pct": ("percent", "percent")}

    def currency(self, text: str, opts: EnglishOptions, every: bool = False) -> str:
        """Eval: ₹500 / $24 / 20% in an English sentence -> 500 rupees / 24 dollars / 20 percent."""
        from .base import currency_to_words
        return self._per_english_sentence(text, lambda seg: currency_to_words(seg, self.CURRENCY), every)

    def canonical(self, text: str, opts: EnglishOptions, every: bool = False) -> str:
        """Eval: contractions, fillers and OK are Latin words, so they are folded wherever they
        are; & -> and only in an English sentence (elsewhere it is dropped with the punctuation)."""
        if not opts.canonical:
            return text
        text = _canonical(text)
        return self._per_english_sentence(text, lambda seg: _AND.sub(" and ", seg), every)


HANDLERS = {"en": English()}


def resolve(mixed=True, spell_small: Optional[bool] = None) -> Dict[str, object]:
    """{handler name: options} for the handlers that run. ``spell_small`` is a shortcut for
    ``{"en": {"spell_small": ...}}`` (None = leave as ``mixed`` sets it)."""
    if mixed is True:
        active = {name: h.Options() for name, h in HANDLERS.items()}
    elif mixed is False or mixed is None:
        active = {}
    elif isinstance(mixed, dict):
        unknown = set(mixed) - set(HANDLERS)
        if unknown:
            raise ValueError(f"unknown code-mixed handler(s) {sorted(unknown)}; known: {sorted(HANDLERS)}")
        active = {}
        for name, h in HANDLERS.items():
            cfg = mixed.get(name, True)
            if cfg is False or cfg is None:
                continue
            if cfg is True:
                active[name] = h.Options()
            elif isinstance(cfg, dict):
                known = {f.name for f in fields(h.Options)}
                bad = set(cfg) - known
                if bad:
                    raise ValueError(f"unknown option(s) {sorted(bad)} for {name!r}; known: {sorted(known)}")
                active[name] = h.Options(**cfg)
            else:
                raise ValueError(f"mixed[{name!r}] must be True, False or a dict of options, not {cfg!r}")
    else:
        raise ValueError(f"mixed must be True, False or a dict, not {mixed!r}")
    if spell_small is not None and "en" in active:
        active["en"] = replace(active["en"], spell_small=bool(spell_small))
    return active


def itn(text: str, active: Dict[str, object], unit_word: UnitWord = None) -> str:
    """Inference ITN of every active handler (after the main language's own ITN)."""
    for name, opts in active.items():
        text = HANDLERS[name].itn(text, opts, unit_word)
    return text


def canonical(text: str, active: Dict[str, object]) -> str:
    """Eval: each active handler's canonical forms (English: can't -> can not, um, OK, &)."""
    for name, opts in active.items():
        text = HANDLERS[name].canonical(text, opts)
    return text


def currency(text: str, active: Dict[str, object]) -> str:
    """Eval: currency sign + amount in sentences of an active handler's language -> amount +
    that language's currency word (before the main language rewrites the rest)."""
    for name, opts in active.items():
        text = HANDLERS[name].currency(text, opts)
    return text


def verbalize(text: str, active: Dict[str, object]) -> str:
    """Eval: digits in sentences of an active handler's language -> that language's words
    (before the main language spells out the rest)."""
    for name, opts in active.items():
        text = HANDLERS[name].verbalize(text, opts)
    return text
