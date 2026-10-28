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
pip install indic-normalizer            # pulls indic-nlp-library (required)
```

Editable, from a local checkout:

```bash
pip install -e ".[dev]"
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
- chillu / ZWJ-ZWNJ canonicalization, two-part vowel signs, colon→visarga (via
  indic-nlp-library, a required dependency)
- Unicode NFC, `<unk>`/`⁇` removal, ASCII + Indic punctuation (incl. danda) removal
- Indic-digit → ASCII, then **digit verbalization** (`24 → ഇരുപത്തിനാല്`) with sandhi joins
- **English↔Malayalam number canonicalization** (`ട്വൻറ്റി ഫോർ`/`24`/`ഇരുപത്തിനാല്` all
  collapse to one form)
- lowercase (code-mixed English), whitespace collapse

## Digits module (`indic_normalizer.digits`)

A separate part of the package for **digit normalization**: it turns numbers in text into the
words that are spoken, e.g. for training targets (the core normalizer leaves digits alone in
`train` mode). Core behaviour is unchanged by it.

```python
from indic_normalizer import digits

digits.verbalize("ഫീസ് 1,75,000 ആണ്, 4:00-5:00 വരെ")
# 'ഫീസ് ഒരു ലക്ഷത്തി എഴുപത്തയ്യായിരം ആണ്, നാല് മണി അഞ്ച് മണി വരെ'

digits.find("70% ആയി")          # [NumberSpan(core='70%', cls='percent', token_index=0, ...)]
digits.candidates("2.5")         # രണ്ടര / രണ്ട് പോയിന്റ് അഞ്ച് / ടു പോയിന്റ് ഫൈവ്
```

It detects and classifies numbers (date, time, time range, percent, currency, phone, range,
grouped, decimal, ordinal, year, ID, numeral + suffix, alphanumeric) and lists the plausible
readings for each: native Malayalam (contracted `ഇരുപത്തഞ്ച്` by default, or glide
`ഇരുപത്തിയഞ്ച്` with `style="glide"`), English words in Malayalam script (`ട്വന്റി ഫൈവ്`), and
class-specific forms (`രണ്ടര`, `ആറേമുക്കാൽ`, `ഇരുപത്തിമൂന്നിന്`). A digit doesn't record how it
was said, so `verbalize(text, chooser=fn)` lets you pick per number — e.g. score each candidate
with an ASR model on that word's audio (`span.token_index` matches word-timestamp lists).
Without a chooser the first (native) reading is used. Output never contains digits or number
punctuation.

### Choosing readings from the audio (optional extra)

```bash
pip install "indic-normalizer[digits] @ git+https://github.com/paulsam1016/indic-normalizer"
# torch / NeMo: install the builds matching your CUDA if pip's defaults don't fit
```

A plain `pip install indic-normalizer` installs only the core dependencies; the text-only digits
functions above work without the extra. The extra adds numpy, PyYAML, soundfile, torch and
NeMo for scoring each candidate reading against the audio with a NeMo hybrid RNN-T (TDT) / CTC
model (negative log-likelihood of the candidate text; lowest wins).

```bash
cp examples/digits_scoring.yaml my.yaml        # set model_path, cuda, compute_dtype, batch_size, ...
# compare the 4 methods (tdt|ctc head x segment|word unit) on a dir, optionally vs base-model hyps
indic-digits check   --config my.yaml DATA_DIR AUDIO_DIR      # model sanity check, run first
indic-digits compare --config my.yaml DATA_DIR AUDIO_DIR --hyps-dir base_hypotheses-streaming --hyps-dir base_hypotheses-segment
# verbalize -> OUT_DIR/<data dir name>/ (out_dir in the config, or --out-dir):
#   verbalized/<stem>.json  input schema (transcription / words / sentences), texts verbalized
#   verify/<stem>.json      confident numbers verbalized, uncertain ones still digits (verify.tsv: pending)
#   failed/<stem>.json      input, unchanged: could not be processed     (details: failed.tsv)
#   verbalized.tsv, summary.json, report.html
# then check the uncertain numbers in the browser: indic-verifier (see "Verifier" below)
indic-digits verbalize --config my.yaml DATA_DIR AUDIO_DIR
# speed: change batch_size / joint_chunk_frames / encoder_batch_s / group_audio_s one at a time,
# run compare for each, report time / peak GPU memory / changed picks -> outputs/sweep/<dir>/sweep.html
indic-digits sweep --config my.yaml --sweep-config examples/digits_sweep.yaml DATA_DIR AUDIO_DIR
indic-digits sweep --config my.yaml --sweep-config examples/digits_sweep_dtype.yaml DATA_DIR AUDIO_DIR  # f32 vs bf16
indic-digits stats DATA_DIR                     # digit / special-character distribution report
indic-digits text "ഫീസ് 1,75,000 ആണ്" --candidates
```

`DATA_DIR` holds one JSON per recording (`sentences` with `text`/`start_time`/`end_time`, and
`words` with timestamps, like eval_frozen); `AUDIO_DIR` holds `<same name>.wav`.

### Verifier

A local browser app (`indic_normalizer.verifier`, standard library only: no GPU, torch or numpy;
Windows or Linux) for a verbalize run dir:

```bash
indic-verifier OUT_DIR/<data dir> --audio-dir AUDIO_DIR --data-dir DATA_DIR   # = indic-digits verifier
```

- **Numbers**: check the ones verbalize was unsure about (verify/, reverify/, failed/) one at a
  time with audio, then save; change numbers already decided, also in finished files
  (file list: *verbalized (auto)* / *verified*). Changing a decided number rebuilds the file from
  the original input, so `--data-dir` (or the page's Input folder) must point at it.
- **Sentences** (✎ Sentences tab): fix the text of any sentence of the file. Saved at once and
  logged in `sentence_edits.tsv`; the sentence's words are re-split with timings spread over its
  span (approximate). Hand edits are re-applied whenever the file is rebuilt (a decided number
  changed, `reverify`), and `reverify` leaves the numbers of hand-edited sentences alone.
- Logs for study: `verifier_decisions.tsv` (every reading saved), `typed_readings.tsv`
  (readings typed by hand), `sentence_edits.tsv`.

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
