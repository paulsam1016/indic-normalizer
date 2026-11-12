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


class English:
    """English in Latin script. Inference: number words -> digits (english_itn). Eval: digits in a
    mostly-English sentence are spelled out in English (I worked 24 hours -> twenty four hours),
    so they meet a spoken "twenty four hours", and a currency sign becomes its English word
    (₹500 -> five hundred rupees); other sentences are left to the main language."""

    name = "en"
    Options = EnglishOptions

    def itn(self, text: str, opts: EnglishOptions, unit_word: UnitWord = None) -> str:
        from .languages.english_itn import itn
        return itn(text, spell_small=opts.spell_small, unit_word=unit_word)

    # a sentence ends at . ! ? । ॥ followed by a space or the end (not inside 2.5 / 2.0.1), or a newline
    _SENTENCE = re.compile(r".*?(?:[.!?।॥]+(?=\s|$)|\n|$)", re.S)
    _DIGITS = re.compile(r"[0-9]+")

    @staticmethod
    def _words(m: "re.Match[str]") -> str:
        from .digits import latin
        s = m.group(0)
        if len(s) > 7 or (len(s) > 1 and s[0] == "0"):      # id-like: digit by digit (as E8)
            return f" {latin.digits(s)} "
        return f" {latin.words(int(s))} "

    def _per_english_sentence(self, text: str, fn) -> str:
        from .digits import latin
        out = []
        for m in self._SENTENCE.finditer(text):
            seg = m.group(0)
            out.append(fn(seg) if seg and latin.is_english(seg) else seg)
        return "".join(out)

    def verbalize(self, text: str, opts: EnglishOptions) -> str:
        return self._per_english_sentence(text, lambda seg: self._DIGITS.sub(self._words, seg))

    CURRENCY = {"inr": ("rupee", "rupees"), "usd": ("dollar", "dollars"), "eur": ("euro", "euros"),
                "gbp": ("pound", "pounds"), "pct": ("percent", "percent")}

    def currency(self, text: str, opts: EnglishOptions) -> str:
        """Eval: ₹500 / $24 / 20% in an English sentence -> 500 rupees / 24 dollars / 20 percent."""
        from .base import currency_to_words
        return self._per_english_sentence(text, lambda seg: currency_to_words(seg, self.CURRENCY))


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
