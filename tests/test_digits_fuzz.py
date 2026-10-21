"""Every class of input: candidates() never raises, never leaves digits, keeps English early."""
import itertools
import re

import pytest

from indic_normalizer.digits import find, ml, verbalize

NUMS = ["0", "5", "12", "25", "100", "730", "1000", "2024", "38500", "55000", "1,75,000",
        "12345678", "999999999", "9876543210", "12345678901", "007"]
SHAPES = ["{n}", "{n}%", "₹{n}", "{n}₹", "{n}.5", "{n}.25", "{n}.30", ".{n}", "-{n}", "{n}-ന്",
          "{n}-യുടെ", "{n}യിൽ", "{n}-ഉം", "{n}-മണിക്ക്", "{n}-ലക്ഷം", "{n}th", "{n}-ാം", "{n}-{n}",
          "{n}-{n}-ന്", "{n}/4", "{n}²", "{n}K", "ഡി{n}", "+91 {n}", "{n}:30", "4:00-{n}:30",
          "{n}/5/2020", "{n}-ൽ", "({n})", "{n}.", "൧൨"]
CASES = sorted({s.format(n=n) for n, s in itertools.product(NUMS, SHAPES)})


@pytest.mark.parametrize("style", ["contracted", "glide"])
def test_candidates_never_fail(style):
    bad = []
    for t in CASES:
        for sp in find(t):
            try:
                cs = ml.candidates(sp.core, sp.cls, style)
            except Exception as e:           # noqa: BLE001
                bad.append((t, repr(e)))
                continue
            if not cs or any(re.search(r"[0-9൦-൯]", c.text) for c in cs):
                bad.append((t, "digits left / empty"))
            k = next((i for i, c in enumerate(cs) if c.lang == "en"), None)
            if k is not None and k > ml.EN_MAX_POS:
                bad.append((t, f"first English at {k}"))
        try:
            verbalize(t, style=style)
        except Exception as e:               # noqa: BLE001
            bad.append((t, "verbalize " + repr(e)))
    assert not bad, bad[:20]
