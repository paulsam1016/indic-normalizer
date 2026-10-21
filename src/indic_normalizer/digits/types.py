"""Data types shared by the digits module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """One way a number can be spoken, written out in words.

    ``lang``  "ml" (Malayalam number words), "en" (English number words in Malayalam script)
              or "mixed".
    ``form``  how it is read, e.g. "contracted" (ഇരുപത്തഞ്ച്), "glide" (ഇരുപത്തിയഞ്ച്),
              "half" (രണ്ടര), "point" (രണ്ട് പോയിന്റ് അഞ്ച്), "digit_by_digit", "from_to"
              (A മുതൽ B വരെ) ...
    """

    text: str
    lang: str = "ml"
    form: str = "contracted"

    def __str__(self) -> str:  # so a Candidate can be used where a string is expected
        return self.text


@dataclass(frozen=True)
class NumberSpan:
    """A whitespace token that contains a digit, located in the source text.

    ``start``/``end`` are character offsets of ``core`` (the token without surrounding
    sentence punctuation) in the original text; ``token_index`` is the token's position
    among whitespace-separated tokens — the same index as the word list in word-timestamped
    transcripts, so a chooser can look up the audio span of the number.
    """

    text: str            # the full token as written
    core: str            # token minus leading/trailing sentence punctuation
    start: int
    end: int
    token_index: int
    cls: str             # date, time, percent, ... (see detect.CLASSES)
    lead: str = ""
    trail: str = ""
    context: dict = field(default_factory=dict, compare=False, hash=False)
