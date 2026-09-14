# Malayalam normalization — coverage & gaps (WER vs CER)

> **Contents:** [Background & external validation](#background--external-validation) ·
> [How normalization affects the two metrics](#how-normalization-affects-the-two-metrics) ·
> [Already handled](#already-handled) · [Missing — WER](#missing--primarily-affects-wer-segmentation--boundary) ·
> [Missing — CER](#missing--primarily-affects-cer-character-level-residue) ·
> [Recommended order](#recommended-order) · [Tool comparison](#appendix-external-tool-comparison) ·
> [Sources](#sources)

---

## Background & external validation

This section grounds the design in the ASR-normalization literature and confirms our
approach against the standard Indic tools. The governing principle throughout: **a
normalization is legitimate only if it maps two surface forms a human would call "the same
words" onto one form, and never merges forms that differ in meaning.** Because we normalize
**both reference and hypothesis identically**, our bar differs from a training normalizer's
— we need a form that is **stable** and **meaning-preserving**, not the "linguistically
correct" one.

### The Whisper "vanishing vowels" pitfall — why we do NOT strip by Unicode category

The default multilingual normalizer (OpenAI Whisper's `BasicTextNormalizer`, also in
Meta MMS, Seamless, AssemblyAI Conformer, and HuggingFace `transformers`) is **actively
broken for Indic scripts**, and it silently *lowers* reported WER — worse than being noisy.
For non-English text it does NFKC, then **replaces every character in Unicode category
`M`/`S`/`P` with a space**. In Brahmic scripts the **vowel signs (mātras) and the virama
are Mark (`M`) characters**, so this strips the vowels, not punctuation:
`ഇതൊരു ഉദാഹരണമാണ്` → `ഇത ര ഉദ ഹരണമ ണ` (a meaningless consonant skeleton).

Measured impact (Manohar et al., ACL 2024, on FLEURS): splitting words at every vowel sign
inflates the reference word count, so **Malayalam WER fell 152 points (287%→135%)** on the
baseline model purely from the normalizer; METEOR similarity between a Malayalam sentence
and its own normalized form is **0.00**. English/Finnish survive only because their
diacritics compose under NFKC into single Letter-class codepoints; Indic vowel signs have
no composed Letter form, so NFKC can't save them.

**Implication for our code:** never apply a Mark-class or "strip all combining marks"
filter to Indic text. Our `universal.py` correctly removes an *explicit punctuation set*
(E5), not a Unicode category — this is the single most important thing we get right, and it
must stay that way. A future "clean up marks" change would silently reintroduce this bug;
guard against it with a self-normalization identity test (see [C7](#c7)).

### Why chillu / `ൻ്റ` mismatches exist (and NFC won't fix them)

Malayalam chillu letters have an atomic codepoint (Unicode 5.1+) but the same glyph is also
writable as `consonant + virama + ZWJ` (legacy) or `consonant + virama` (bare). Unicode
deliberately gave the atomic and ZWJ forms **no formal canonical decomposition** (to avoid
destabilizing existing data), so **NFC/NFNC leave them distinct** — a reference in atomic
chillu and a hypothesis in the ZWJ sequence mismatch on every chillu. Handled by E2.

The `ൻ്റ`-vs-`ന്റ` conjunct (E12) is subtler and **no external tool handles it**: IMEs emit
atomic chillu-n + virama + `റ` for the very common *nṯa* conjunct instead of the correct
`ന` + virama + `റ`. NFC doesn't unify them, and the reference libraries' chillu rules don't
either (they compose *toward* atomic chillu or decompose *word-final* chillu; this is a
chillu wrongly sitting *before* a virama). E12 is our own fix and our biggest single win.

### How our approach maps to the literature

- **Manohar, Pillai, Sherly — "What is Lost in Normalization?" (ACL 2024)** proves the
  vanishing-vowels harm and mandates **linguistically-informed, language-specific
  normalizers** (it deliberately proposes no algorithm). Our per-language module in
  `scripts/normalizers/malayalam.py` is exactly that approach.
- **SN-WER (ACL 2026 MeLLM)** proposes a script-insensitive companion metric for
  cross-script / romanized settings. Not needed for single-script Malayalam scoring, but
  the relevant direction if we ever score against romanized references — report it
  *alongside* WER/CER, not instead.
- **Unicode primary sources** (L2/05-308; PR-66; Thottingal's Malayalam Unicode report) are
  the authority on why chillu/ZWJ mismatches exist and why NFC won't fix them.

Verdict against the standard tools (full table in the
[appendix](#appendix-external-tool-comparison)): **our normalizer is the strongest of the
set for ASR scoring.** It does three things no off-the-shelf tool does — the `ൻ്റ→ന്റ` fix
(E12), digit verbalization (E8), and English↔Malayalam number canonicalization (E9) — and
it avoids both known traps (Mark-class stripping, and opinionated typo/spelling
"correction" — see the note under [Recommended order](#recommended-order)).

---

## How normalization affects the two metrics

How normalization affects the two metrics:

- **WER** is driven by anything that changes **how the string splits into words** — the
  word-count / boundary problem. One boundary disagreement flips two tokens at once (a
  substitution *plus* a deletion or insertion), so segmentation variants inflate WER even
  when every character is correct. This is the "word-segmentation variant" effect that
  dominated the Malayalam v0/v2 gap.
- **CER** is driven by residual **character-level** mismatches that survive normalization
  (spelling drift, encoding variants, final-vowel forms). Boundaries matter far less to CER,
  so CER is the fairer view for agglutinative Malayalam.

Rule of thumb for prioritization: fix the WER-side (segmentation) items to stop *overstating*
error; fix the CER-side items to reduce *genuine* residual error. Adopting a spacing-insensitive
CER as the headline Malayalam metric neutralizes most WER-side segmentation noise at once.

---

## Already handled

Applied in `scripts/normalizers/malayalam.py` (`normalize()` order: indic-nlp/chillu fix →
NFC → `<unk>`/`⁇` → zero-width → Indic-digit→ASCII → group-comma → punctuation →
English-number canonicalization → digit verbalization → lowercase → whitespace collapse).

| # | Item | Effect | Notes |
|---|------|--------|-------|
| E1 | Unicode NFC | CER | Canonical composition. |
| E2 | Chillu / ZWJ-ZWNJ canonicalization | CER | indic-nlp when present; stdlib chillu map fallback (`ന്‍`→`ൻ`, etc.). |
| E3 | `⁇` / `<unk>` removal | CER + WER | Soniox/base-model OOV marker; removed from both sides. |
| E4 | Zero-width (ZWNJ/ZWJ/BOM) drop | CER | After the chillu fix. |
| E5 | ASCII + Indic punctuation removal | CER + WER | Includes danda `।` `॥`, quotes, dashes. |
| E6 | Indic-digit → ASCII (`൦-൯`→`0-9`) | CER | Feeds verbalization. |
| E7 | Digit-grouping commas (`70,000`→`70000`) | CER + WER | Before `,` becomes a space, so both surface forms meet. |
| E8 | Native digit verbalization (`24`→`ഇരുപത്തിനാല്`) | WER + CER | 1–4 digit runs read as a number; longer/zero-padded read digit-by-digit. Sandhi glide `യ` at joins. |
| E9 | English-style number words → canonical Malayalam (`ട്വൻറ്റി ഫോർ`→`ഇരുപത്തിനാല്`) | WER + CER | Convention-invariant scoring; ambiguous `ടു`/`ഫോർ` convert only beside an unambiguous number word. Lakh units mapped. |
| E10 | Lowercasing | CER | For code-mixed Latin tokens. |
| E11 | Whitespace collapse + trim | WER | Multiple spaces → one. |
| E12 | Mis-encoded chillu conjunct (`ൻ്റ`→`ന്റ`; also `ൺ ർ ൽ ൾ ൿ` before virama) | CER | A chillu + virama is not a real chillu — it's the base consonant typed with the wrong codepoint (`വിവേകാനന്ദൻ്റെ` vs `വിവേകാനന്ദന്റെ`), NFC-invisible. Rewritten to the base-consonant form; genuine word-final chillu (no following virama) untouched. First step in `malayalam.normalize()` (`_fix_chillu_conjuncts`). |

---

## Missing — primarily affects **WER** (segmentation / boundary)

Highest leverage, because this is where the observed gap lives.

| # | Item | Why it hurts WER | Suggested fix |
|---|------|------------------|---------------|
| W1 | **Sandhi / compound & postposition split-merge** (`വീട്ടിലേക്ക്` vs `വീട്ടില് ലേക്ക്`; fused vs spaced `-ല്`, `-ിന്`, `-ോട്`) | Same phrase, different word count; the single biggest boundary contributor for agglutinative Malayalam. | Curated split/merge map for common postpositions/function morphemes, **or** adopt spacing-insensitive CER as the primary metric. |
| W2 | **English code-mixed token spacing** (`ഫോൺ നമ്പർ` vs `ഫോൺനമ്പർ`) | Inconsistent join of borrowed compounds → word-count mismatch. | Small join/split lexicon for frequent borrowed compounds. |
| W3 | **Acronym / letter-sequence spacing** (`എ ടി എം` vs `എടിഎം` vs `ATM`; `ഒ ടി പി` vs `OTP`) | Each convention is a different token count. | Collapse spaced single-letter runs; map common Latin↔Malayalam-script acronyms. |
| W4 | **Intra-word hyphen handling** (`covid-19` currently → `covid 19`) | Removing hyphen *to a space* splits one token into two; `covid-19`/`covid19`/`covid 19` land in three word counts. | Collapse intra-word hyphens to nothing (or a fixed rule) instead of to a space. |
| W5 | **Reduplication / echo words** (`വേഗം വേഗം`, `പെട്ടെന്ന് പെട്ടെന്ന്`) joined vs spaced | Boundary-only word-count mismatch. | Low priority; normalize spacing of known reduplications if it shows up. |

---

## Missing — primarily affects **CER** (character-level residue)

Reduces genuine residual error; most-per-line-of-code wins first.

| # | Item | Why it hurts CER | Suggested fix |
|---|------|------------------|---------------|
| C1 | **Samvruthokaram / final-vowel drift** (`നാല്` vs `നാലു` vs `നാല`; final `ു്`/`്`/`ു`) | Hits nearly every word ending → large, broad CER contributor. | Canonicalize word-final samvruthokaram to one form (align with the form the number tables already emit). |
| C2 | **Anusvara vs nasal spelling** (`കാര്യം` vs `കാര്യന്`; `ം` vs `ന്`/`മ്`) | Per-character substitution surviving current pipeline. | Rule to canonicalize word-final nasal representation. |
| C3 | **Conjunct / reph / `്യ`·`്വ` ordering variants** not unified by NFC | Multiple valid encodings of the same conjunct; stdlib fallback misses them. | Rely on indic-nlp when present; add targeted rules for the common ones in the stdlib path. |
| C4 | **Long-vowel / gemination spelling drift** (`ാ` vs `ാാ`, `ക്ക` vs `ക`) | Casual-transcription doubling/undoubling. | Optional collapse of obvious duplicates; be conservative to avoid changing meaning. |
| C5 | **Foreign-word transliteration variants** (`ഫോൺ`/`ഫോണ്`, `ബസ്`/`ബസ്സ്`) | Same borrowed word, different spelling. | Small canonical-spelling lexicon for frequent borrowings. |
| C6 | **Visual-lookalike letters/digits** (`ഠ`/`൦`/`0`/`O`) | Rare homoglyph substitutions. | Low priority; map if observed. |
| <a name="c7"></a>C7 | **indic-nlp fallback parity gap** — colon→visarga (`ദു:ഖം`→`ദുഃഖം`) and two-part vowel-sign composition (` െ`+`ാ`→`ൊ`, au forms) fire **only when indic-nlp-library is installed**; the stdlib fallback path skips them. | Same input scores differently with vs without the optional dependency — a silent, environment-dependent CER difference on both sides. | Port these two rules (from indic-nlp's `MalayalamNormalizer`) into the stdlib fallback, **or** make indic-nlp a hard dependency for the `ml` path. Overlaps C3. Guard: `ദു:ഖം`↔`ദുഃഖം`, `െ`+`ാ`↔`ൊ`. |
| <a name="c8"></a>C8 | **Bare (ZWJ-less) word-final chillu** — `consonant + virama` with no ZWJ may not unify with atomic chillu, since indic-nlp keys on the ZWJ. | Legacy/IME chillu without ZWJ mismatches an atomic-chillu reference. | Audit the eval corpus for bare word-final `+virama`; if present, add the mapping to the stdlib chillu table. |

---

## Guardrail test (prevents regressions)

<a name="c7-guard"></a>Add a **self-normalization identity test** per the ACL-2024 METEOR
check: for a basket of real Malayalam sentences, assert `normalize(s)` preserves word count
and non-trivial character overlap with `s`. This is a cheap tripwire against a future
well-meaning change that reintroduces the vanishing-vowels behavior (Mark-class stripping).
Pair it with pinned regression cases for the crown-jewel rules: `വിവേകാനന്ദൻ്റെ` and
`വിവേകാനന്ദന്റെ` → same output (E12); `ട്വൻറ്റി ഫോർ` / `24` / `ഇരുപത്തിനാല്` → same output
(E8/E9).

---

## Recommended order

0. **Add the guardrail test (above) + C7 fallback parity** — cheap, and they protect the
   wins already banked. C7 removes an environment-dependent scoring difference; the identity
   test prevents silently reintroducing the vanishing-vowels bug. Do these alongside step 1.
1. **Adopt spacing-insensitive CER** as the headline Malayalam metric (low-risk in
   `eval_metrics`) — directly quantifies how much of the gap is W1/W3 segmentation. Do this first.
2. **W1 + W3** (postposition split-merge + acronym/letter-sequence collapse) — biggest WER-side wins.
3. **C1 + C2** (samvruthokaram + anusvara) — biggest CER-side wins per line of code, since they
   touch most word endings.
4. Remaining items (W2, W4, W5, C3–C6, C8) as they surface in error analysis.

> **Symmetry.** Every rule must be applied **symmetrically to hypothesis and reference** (as
> the current normalizer already is), or it will bias the score instead of fixing it.
>
> **Do NOT add opinionated spelling/typo correction.** The external tools offer this
> (libindic's "common mistakes" corrector, on by default; indic-nlp's `do_correct_geminated_T`
> = `റ്റ→ട്ട`; "alternate forms"). These change *spelling*, not encoding, and for scoring they
> are double-edged: they can **hide** a real mis-transcription (if the model's error happens to
> be one the corrector "fixes") and can **collapse** legitimate spelling contrasts. Our job is
> meaning-preserving canonicalization, not orthographic opinion — keep these off. This is a
> deliberate decision, distinct from C1/C5 which canonicalize *encoding/inflectional* variants,
> not lexical spelling.

---

## Appendix: external tool comparison

The reference Malayalam normalizers and what each actually does, evaluated for the
**ASR-scoring** use case (symmetric ref+hyp, meaning-preserving, stable).

| capability | ours | indic-nlp-library | libindic/normalizer | whisper_normalizer |
|---|---|---|---|---|
| NFC (E1) | ✅ | ➖ does own | ➖ | ➖ |
| chillu ZWJ-seq → atomic (E2) | ✅ via indic-nlp | ✅ default | ✅ | ✅ |
| **`ൻ്റ→ന്റ` conjunct (E12)** | ✅ **unique** | ❌ | ❌ | ❌ |
| two-part vowel signs (`ൊ ോ ൌ`) | ✅ *(indic-nlp only — see C7)* | ✅ | ✅ | ✅ |
| colon → visarga | ✅ *(indic-nlp only — see C7)* | ✅ | ✅ | ✅ |
| Indic danda `।॥` removed (E5) | ✅ | ➖ | ❌ ASCII-only | ➖ |
| Indic digit → ASCII (E6) | ✅ | ❌ | ❌ | ❌ |
| **digit verbalization (E8)** | ✅ **unique** | ❌ | ❌ | ❌ |
| **English↔Malayalam number canon. (E9)** | ✅ **unique** | ❌ | ❌ | ❌ |
| Mark-class stripping (the bug) | ❌ correctly avoids | ❌ correctly avoids | ❌ correctly avoids | ⚠️ in non-Indic base path |
| opinionated typo/spelling "fix" | ❌ correctly avoids | ➖ opt-in | ⚠️ **on by default** | ➖ |

Notes:
- **indic-nlp-library** (Anoop Kunchukuttan, MIT) — the de-facto standard and our optional
  backend. Its `MalayalamNormalizer` composes legacy ZWJ chillu → atomic (default on),
  does two-part vowels / poorna-virama / colon→visarga (default on), and `റ്റ→ട്ട` +
  reverse chillu decomposition (default **off**). Does **not** handle E12.
- **libindic/normalizer** (Santhosh Thottingal / SMC, LGPL) — YAML-rule driven; removes
  **only ASCII punctuation** (misses danda/curly quotes) and ships an opinionated typo
  corrector flagged "needs thorough review". Not suitable as-is for scoring.
- **whisper_normalizer** (Kurian Benoy, 5M+ downloads) — wraps indic-nlp's Malayalam
  normalizer for the `ml` path but defaults to the broken `BasicTextNormalizer` for the
  non-Indic path; evidence that indic-nlp is the community base, not a distinct algorithm.
- **AI4Bharat / IndicNLP** — uses indic-nlp-library for normalization; not a separate
  algorithm.

## Sources

- Kavya Manohar, [Indian Languages and Text Normalization: Part 1](https://kavyamanohar.com/post/indic-normalizer/)
- Manohar, Pillai, Sherly, [*What is Lost in Normalization? Exploring Pitfalls in Multilingual ASR Model Evaluations*](https://arxiv.org/abs/2409.02449) (ACL 2024)
- Pattnayak et al., [*SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation*](https://arxiv.org/abs/2606.02548) (ACL 2026 MeLLM)
- [indic-nlp-library — `indic_normalize` source](https://indic-nlp-library.readthedocs.io/en/latest/_modules/indicnlp/normalize/indic_normalize.html) (Anoop Kunchukuttan, MIT)
- [libindic/normalizer](https://github.com/libindic/normalizer) (Santhosh Thottingal / SMC)
- [kurianbenoy/whisper_normalizer](https://github.com/kurianbenoy/whisper_normalizer)
- Ross O'Connell, [Breaking Brahmic: How OpenAI's Text Cleaning Hides Whisper's True WER for South Asian Languages](https://deepgram.com/learn/how-openai-s-text-normalization-hides-whisper-s-true-word-error-rate-for-south-asian-and-southeast-asian-languages) (Deepgram, 2023)
- Unicode primary sources: [Problems of Malayalam Encoding (L2/05-308)](https://www.unicode.org/L2/L2005/05308-malayalam-probs.pdf) · [PR-66 Encoding Chillu Forms](https://www.unicode.org/review/pr-66.html) · [Thottingal — Report on Malayalam Unicode Issues](https://thottingal.in/documents/ReportonMalayalamUnicodeIssues.pdf)
- [AI4Bharat IndicNLP](https://indicnlp.ai4bharat.org/home/) · [indic-punct: punctuation + ITN for Indic](https://arxiv.org/html/2203.16825v1)
