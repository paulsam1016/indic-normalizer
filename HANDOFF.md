# indic-normalizer — Project Handoff

Handoff for a fresh session to pick up the `indic-normalizer` library without losing
context. Project root: `C:\Users\Paul\Projects\PythonProjects\indic-normalizer`
(import name `indic_normalizer`, distribution `indic-normalizer`).

---

## Objective

A standalone, installable Python library providing **meaning-preserving text
normalization for Indic-script ASR scoring (WER/CER)**, extracted from AetherTranscribe
so it can be reused across projects. Malayalam-first, with a framework designed so other
Indic languages drop in as plugins.

The core invariant (why this library exists): when scoring ASR you normalize **both**
reference and hypothesis, so the normalizer must be **stable** (same input → same output)
and **meaning-preserving** (never collapse a real distinction). It deliberately avoids the
Whisper "vanishing vowels" pitfall — see `NORMALIZATION.md` → *Background & external
validation* for the full rationale, the ACL-2024 grounding, and the head-to-head tool
comparison.

---

## Current Status & Context

**Working, installed, and green.** The library was scaffolded and ported from
AetherTranscribe's `scripts/normalizers/` this session. `pip install -e ".[dev]"` succeeds
and **20/20 tests pass**, verified on BOTH code paths:
- stdlib fallback (no `indic-nlp-library`), and
- with `indic-nlp-library` installed (the fuller Malayalam rule set).

Smoke-tested public API (all confirmed working):
```python
from indic_normalizer import get_normalizer, normalize, script_of
fn, src = get_normalizer("ml")     # src == "indic_normalizer.malayalam"
fn("24") == fn("ട്വൻറ്റി ഫോർ") == fn("ഇരുപത്തിനാല്")   # True
fn("വിവേകാനന്ദൻ്റെ") == fn("വിവേകാനന്ദന്റെ")           # True  (ൻ്റ→ന്റ fix)
script_of("ഇതൊരു")                 # "ml"  (Unicode-block routing, NOT language ID)
normalize("Hello, WORLD!")         # "hello world"  (universal fallback)
```

**File layout:**
```
pyproject.toml                     # hatchling, src layout; deps=[]; extras: [indic],[dev]
README.md
NORMALIZATION.md                   # authoritative design/coverage/gap doc (cite this)
HANDOFF.md                         # this file
src/indic_normalizer/
  __init__.py                      # get_normalizer, normalize(text,lang), script_of, __version__
  base.py                          # shared primitives: NFC, zero-width, punct, digit-fold, whitespace
  universal.py                     # language-agnostic normalizer
  registry.py                      # _ALIASES, _lang_code, _load (lazy), get_normalizer, script_of + _SCRIPT_BLOCKS
  languages/
    __init__.py
    malayalam.py                   # chillu, ൻ്റ fix, digit verbalization, EN↔ML numbers, indic-nlp optional
tests/
  test_universal.py                # 7 tests
  test_malayalam.py                # 13 tests (E12/E8/E9 pins + vanishing-vowels guardrail)
```

**Relationship to AetherTranscribe:** the code was ported FROM
`AetherTranscribe/scripts/normalizers/` (`malayalam.py`, `universal.py`, `__init__.py`).
Those originals **still exist and are still in use** by AetherTranscribe — the wiring to
consume this library has NOT been done yet (see Immediate Next Steps). The
`NORMALIZATION.md` here is a copy of the one merged into AetherTranscribe this session;
they will drift unless one becomes canonical.

---

## Completed Tasks

1. **Package scaffolded** — `pyproject.toml` (PEP 621, hatchling backend, src layout, zero
   required deps, `[indic]`=indic-nlp-library / `[dev]`=pytest extras), `README.md`.
2. **`base.py`** — shared framework: `to_nfc`, `strip_unk`, `strip_zero_width`,
   `fold_digits`, `strip_punct`, `collapse_space`, plus the shared `UNK`/`ZERO_WIDTH`/
   `PUNCT`/`SPACE` regexes. Explicit punctuation set (incl. Indic danda `।॥`), NEVER a
   Unicode Mark/Symbol/Punctuation *category* filter (that is the vanishing-vowels bug).
3. **`universal.py`** — language-agnostic normalizer, refactored onto `base`.
4. **`registry.py` + `__init__.py`** — `get_normalizer(lang) -> (fn, source)` with alias
   table (`ml`/`mal`/`malayalam`), lazy per-language import, universal fallback; plus
   `script_of(text)` Unicode-block script routing and a `normalize(text, lang=None)`
   convenience.
5. **`languages/malayalam.py`** — full port: `_fix_chillu_conjuncts` (ൻ്റ→ന്റ, runs
   first), indic-nlp chillu/ZWJ canonicalization with stdlib fallback, digit
   verbalization (native Malayalam number words + sandhi glide യ), English↔Malayalam
   number canonicalization. `INDIC_NLP_AVAILABLE` flag preserved.
6. **Tests** — 20 tests incl. the crown-jewel regression pins and the ACL-2024-style
   self-normalization identity guardrail (word count + vowel-sign preservation).
7. **`NORMALIZATION.md`** copied in as the design doc.
8. **Verified** — installed editable; 20/20 pass with and without indic-nlp-library.
9. **Bug fixed during verification** — removed invalid trove classifier
   `Natural Language :: Malayalam` from `pyproject.toml` (not a real PyPI classifier; it
   blocked the hatchling build).

---

## Immediate Next Steps

Ordered; the first is the reason the library was extracted.

1. **Wire AetherTranscribe to consume the library (decided approach: pip editable).**
   - Add `indic-normalizer` to AetherTranscribe deps (editable local path for now:
     `pip install -e ../indic-normalizer`, later a pinned wheel/git ref).
   - Turn `AetherTranscribe/scripts/normalizers/` into a **thin shim** that re-exports
     `get_normalizer`/`normalize` from `indic_normalizer` (keeps all existing call sites —
     `benchmarks/test.py`, `scripts/eval_metrics.py`, etc. — working unchanged), OR update
     call sites directly. Confirm which with the user.
   - Delete the now-duplicated `AetherTranscribe/scripts/normalizers/NORMALIZATION.md`
     (this repo's copy becomes canonical), or leave a pointer.
   - Re-run AetherTranscribe's normalizer/eval tests to confirm parity.

2. **Repo hygiene before any publish:** add `.gitignore` (`__pycache__/`, `*.egg-info/`,
   `.pytest_cache/`, `build/`, `dist/`), add a `LICENSE` file (MIT is declared in
   `pyproject.toml` but no file exists), `git init` + initial commit.

3. **Design-doc backlog items** (from `NORMALIZATION.md` → *Recommended order*), highest
   leverage first — these are the substantive normalizer improvements, not packaging:
   - **C7 (fallback parity)** — port colon→visarga and two-part vowel-sign composition
     into the stdlib fallback path so scoring is identical with/without indic-nlp. Cite
     `NORMALIZATION.md` row C7 for exact rules + guard cases.
   - **Guardrail test** already present here (`test_no_vanishing_vowels`); consider
     expanding the sentence basket.
   - **C1 (samvruthokaram / final-vowel drift)** and **W1 (sandhi/postposition
     split-merge)** are the biggest CER/WER levers respectively — see `NORMALIZATION.md`.
   - Note `NORMALIZATION.md` recommends **adopting spacing-insensitive CER as the headline
     Malayalam metric** first; that change lives in AetherTranscribe's `eval_metrics`, not
     this library.

4. **Optional publish path:** `python -m build`, TWINE to PyPI/private index once (2) done
   and the name is confirmed available.

---

## Known Constraints

- **No language auto-detection by design.** `get_normalizer` requires an explicit language
  code. Script ≠ language (Devanagari is shared by Hindi/Marathi/Sanskrit/Nepali), and
  reliable language-ID needs a model. `script_of(text)` provides Unicode-**block** script
  routing only. For ASR scoring the language is known (it's the benchmark language). See
  `README.md` → *Language auto-detection?* and `NORMALIZATION.md`.
- **Do NOT add opinionated spelling/typo correction.** External tools offer it (libindic's
  "common mistakes" corrector on-by-default; indic-nlp's `do_correct_geminated_T`
  `റ്റ→ട്ട`; "alternate forms"). It changes spelling, not encoding, and for scoring is
  double-edged (can hide real errors, can collapse legitimate contrasts). Explicit
  decision — see `NORMALIZATION.md` → *Recommended order* note. Keep off.
- **Symmetry rule.** Every normalization MUST be applied identically to reference and
  hypothesis, or it biases the score instead of fixing it (`NORMALIZATION.md`).
- **Never strip by Unicode category** (`M`/`S`/`P`). Indic vowel signs (mātras) and the
  virama are Mark-class; category stripping deletes them (the Whisper "vanishing vowels"
  bug — silently *lowers* WER). Use the explicit `PUNCT` set in `base.py`. Guarded by
  `test_no_vanishing_vowels`.
- **`ൻ്റ→ന്റ` (E12) is unique to us** — no external library (indic-nlp, libindic,
  whisper_normalizer) handles this NFC-invisible, high-frequency conjunct. It must run
  **before** indic-nlp normalization. Pinned by `test_chillu_conjunct_fix`; don't break
  the ordering.
- **indic-nlp-library is optional, not required.** Core must stay pure-stdlib (zero deps).
  The Malayalam path has a stdlib fallback; keep both paths behaviourally equivalent
  (this is exactly what C7 closes).
- **Python ≥ 3.9** (declared in `pyproject.toml`). Sandbox verification ran on 3.10.
- **Number tables are Malayalam-specific and hand-curated** from real base-model output
  (spelling variants of English-in-Malayalam-script numbers). Extending to another
  language means new tables, not reuse.
- **Not yet a git repo / not published.** No `.gitignore`, no `LICENSE` file, no VCS
  history yet (see Next Steps 2).

---

## Quick reference — run it

```bash
cd C:\Users\Paul\Projects\PythonProjects\indic-normalizer
pip install -e ".[indic,dev]"      # or ".[dev]" to test the stdlib fallback path
pytest -q                          # expect: 20 passed
```
