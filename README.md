# indic-normalizer

Meaning-preserving text normalization for **Indic-script ASR scoring** (WER/CER),
Malayalam-first.

When you score ASR you normalize **both** the reference and the hypothesis so the metric
penalizes real mis-transcriptions — not casing, punctuation, or encoding/convention
differences. The normalization must be **stable** (same input → same output) and
**meaning-preserving** (never collapse a real distinction). This library provides exactly
that, and deliberately avoids the well-known [Whisper "vanishing vowels"
pitfall](https://arxiv.org/abs/2409.02449) where category-based Mark-stripping deletes the
vowel signs of Brahmic scripts. See [`NORMALIZATION.md`](./NORMALIZATION.md) for the full
design, coverage/gap analysis, and sources.

## Install

```bash
pip install indic-normalizer            # core, pure stdlib, zero deps
pip install "indic-normalizer[indic]"   # + indic-nlp-library for fullest Malayalam rules
```

Editable, from a local checkout:

```bash
pip install -e ".[indic,dev]"
```

## Usage

```python
from indic_normalizer import get_normalizer, normalize

# Explicit language (recommended for scoring — you know the benchmark language):
fn, source = get_normalizer("ml")        # aliases: ml / mal / malayalam
fn("ട്വൻറ്റി ഫോർ")                        # -> 'ഇരുപത്തിനാല്'
fn("24") == fn("ട്വൻറ്റി ഫോർ") == fn("ഇരുപത്തിനാല്")   # True

# One-shot:
normalize("ഇ–മെയിൽ", lang="ml")
normalize("hello, WORLD!")               # no lang -> universal normalizer

# Script routing (NOT language ID — see below):
from indic_normalizer import script_of
script_of("ഇതൊരു")                       # -> 'ml'
```

`get_normalizer(lang)` returns `(normalize_fn, source_label)`. A language with a dedicated
normalizer (Malayalam today) uses it; everything else falls back to the language-agnostic
**universal** normalizer (NFC, punctuation/zero-width/`⁇` removal, digit-fold, lowercase,
whitespace collapse).

## What the Malayalam normalizer does

Applied identically to reference and hypothesis:

- mis-encoded chillu-conjunct fix (`ൻ്റ → ന്റ`, NFC-invisible — **no other library does
  this**)
- chillu / ZWJ-ZWNJ canonicalization (indic-nlp-library if installed, else a stdlib
  fallback)
- Unicode NFC, `<unk>`/`⁇` removal, ASCII + Indic punctuation (incl. danda) removal
- Indic-digit → ASCII, then **digit verbalization** (`24 → ഇരുപത്തിനാല്`) with sandhi joins
- **English↔Malayalam number canonicalization** (`ട്വൻറ്റി ഫോർ`/`24`/`ഇരുപത്തിനാല്` all
  collapse to one form)
- lowercase (code-mixed English), whitespace collapse

## Language auto-detection?

**No** — `get_normalizer` takes an explicit language. Script ≠ language
(Hindi/Marathi/Sanskrit all share Devanagari), and reliable language-ID needs a model.
`script_of(text)` offers trivial, reliable Unicode-**block** routing when script alone is
enough. For ASR scoring you already know the language (it's the benchmark language), so
pass it.

## Adding a language

1. Drop `src/indic_normalizer/languages/<lang>.py` exposing `normalize(text) -> str`
   (reuse `indic_normalizer.base` for the shared steps).
2. Register it in `src/indic_normalizer/registry.py`: add aliases to `_ALIASES` and a
   branch in `_load`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
