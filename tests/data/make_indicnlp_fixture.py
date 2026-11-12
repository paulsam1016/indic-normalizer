"""Regenerate tests/data/indicnlp_092_ml.json: outputs of the REAL indic-nlp-library 0.92
Malayalam normaliser, which tests/test_indicnlp_port.py checks the built-in port against.

Not needed for the test suite (the fixture is committed); run it only to widen the fixture:

    pip install "indic-nlp-library==0.92"
    python tests/data/make_indicnlp_fixture.py

Inputs: the port test's edge cases plus every string literal containing Malayalam in the
package source and the tests, each as raw, NFC and NFD.
"""

import ast
import importlib.metadata
import json
import pathlib
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from test_indicnlp_port import EDGE_CASES  # noqa: E402
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory  # noqa: E402

VERSION = importlib.metadata.version("indic-nlp-library")
assert VERSION == "0.92", f"the fixture records indic-nlp-library 0.92, not {VERSION}"


def malayalam_literals():
    out = set()
    for p in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").glob("test_*.py")]):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and any("ഀ" <= c <= "ൿ" for c in node.value)):
                out.add(node.value)
    return out


def main():
    ref = IndicNormalizerFactory().get_normalizer("ml")
    texts = set(EDGE_CASES) | malayalam_literals()
    inputs = sorted({v for t in texts for v in (t, unicodedata.normalize("NFC", t),
                                                unicodedata.normalize("NFD", t))})
    pairs = [[s, ref.normalize(s)] for s in inputs]
    path = pathlib.Path(__file__).with_name("indicnlp_092_ml.json")
    path.write_text(json.dumps({"indic_nlp_library": VERSION, "pairs": pairs},
                               ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"{len(pairs)} pairs -> {path}")


if __name__ == "__main__":
    main()
