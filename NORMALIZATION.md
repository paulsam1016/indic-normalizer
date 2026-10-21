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

The **Complexity** column rates what each rule took to *build* (same ★ scale as the gap
tables — **★ trivial** library call/one substitution … **★★★★ hard** bespoke logic with
guards). Rows keep pipeline order, not difficulty order, because the sequence is load-bearing
(e.g. E6→E7→E8 must run before verbalization, E12 must run first).

| # | Item | Effect | Complexity | Notes |
|---|------|--------|:----------:|-------|
| E1 | Unicode NFC | CER | ★ trivial | Canonical composition — stdlib `unicodedata.normalize`. |
| E2 | Chillu / ZWJ-ZWNJ canonicalization | CER | ★ trivial | One call into indic-nlp-library (a required dependency); the library does the work. Punctuation colons are shielded from its colon→visarga rewrite (see Mode selector, colon guard). |
| E3 | `⁇` / `<unk>` removal | CER + WER | ★ trivial | OOV marker; fixed-string removal from both sides. |
| E4 | Zero-width (ZWNJ/ZWJ/BOM) drop | CER | ★ trivial | Fixed codepoint set stripped, after the chillu fix. |
| E5 | ASCII + Indic punctuation removal | CER + WER | ★ trivial | One character-class regex; includes danda `।` `॥`, quotes, dashes. |
| E6 | Indic-digit → ASCII (`൦-൯`→`0-9`) | CER | ★ trivial | Fixed 10-codepoint translation table. Feeds verbalization. |
| E7 | Digit-grouping commas, Western **and Indian lakh** (`70,000`, `1,75,000`→`175000`) | CER + WER | ★★ easy | Regex over the whole grouped number (`1,2,3` lists untouched); must fire before `,`→space so both surface forms meet — ordering-sensitive. The old 3-digit-only rule split `1,75,000` into `1 75000` (24× on eval_frozen). |
| E8 | Native digit verbalization (`24`→`ഇരുപത്തിനാല്`) | WER + CER | ★★★★ hard | Bespoke number-to-Malayalam-words logic: runs up to 7 digits are read as a number — thousands fused (`12000`→`പന്ത്രണ്ടായിരം`, `20000`→`ഇരുപതിനായിരം`, `12500`→`പന്ത്രണ്ടായിരത്തിയഞ്ഞൂറ്`), lakhs as a spaced word the way ASR writes them (`200000`→`രണ്ട് ലക്ഷം`, `175000`→`ഒരു ലക്ഷത്തി എഴുപത്തിയഞ്ചായിരം`); 8+ digits or a leading zero (phones, IDs) read digit-by-digit; sandhi glide `യ` at joins. **`eval` mode only.** |
| E9 | English-style number words → canonical Malayalam (`ട്വൻറ്റി ഫോർ`→`ഇരുപത്തിനാല്`) | WER + CER | ★★★★ hard | Convention-invariant scoring: a spelling-tolerant word parser, with ambiguous `ടു`/`ഫോർ` converted only beside an unambiguous number word, and lakh units mapped. **`inference` mode only** (→ digits); not in `eval` or `train`. |
| E10 | Lowercasing | CER | ★ trivial | `str.lower()`, for code-mixed Latin tokens. |
| E11 | Whitespace collapse + trim | WER | ★ trivial | One regex: multiple spaces → one. |
| E13 | Annotation tags (`<unintelligible>`, `<noise>`) | WER | ★ trivial | Whole tag dropped; E5 alone would leave the tag name as a fake word (123 in dataset refs). **`eval` + `inference`**; `train` leaves tags to the data pipeline. |
| E12 | Mis-encoded chillu conjunct (`ൻ്റ`→`ന്റ`; also `ൺ ർ ൽ ൾ ൿ` before virama) | CER | ★★★ moderate | Small closed map, but built on a real insight and guarded: a chillu + virama is not a real chillu — it's the base consonant typed with the wrong codepoint (`വിവേകാനന്ദൻ്റെ` vs `വിവേകാനന്ദന്റെ`), NFC-invisible. Rewritten to the base-consonant form; genuine word-final chillu (no following virama) left untouched. First step in `malayalam.normalize()` (`_fix_chillu_conjuncts`). |

---

## Mode selector — `eval` / `train` / `inference`

`normalize()` takes a **`mode`** argument with **three behaviours**:

- **`eval`** (**default**; alias `evaluation`) — **WER/CER scoring.** Aggressive, symmetric
  canonicalization applied to reference *and* hypothesis so only real recognition errors count.
  Digits are verbalized to **native Malayalam words** (`24` → `ഇരുപത്തിനാല്`) so digit references
  meet word hypotheses: call references write numbers as digits (1,659 digit runs) while the ASR
  always writes words (0 digit runs) — without E8 calls WER rises 26.1 → 29.6. English-style
  number words (`ട്വൻറ്റി ഫോർ`) are **left as written** in eval (E9 dropped; measured cost +0.21
  WER calls, +0.10 dataset).
- **`train`** (aliases `training`, `fine-tune`, `finetune`) — **training / fine-tune targets.**
  Like `eval`, but numbers are not rewritten (no E8) — digit verbalization for targets is
  handled outside this library — and three things are kept for that external verbalizer and
  for spelling fidelity:
  - **T1 number-internal punctuation survives E5**: `4:00-5:00`, `2.5`, `70%`, `1/2`, `23-ന്`,
    `8-9` (eval_frozen: 77 times, 23 decimals, 31 percents, 179 numeral-suffixes, 30 ranges,
    11 slashes). Ordinary punctuation is still stripped.
  - **No C1 fold** — a target keeps its real spelling (`ഒരു`, `പറഞ്ഞു`); C1 used to rewrite
    4–5.6% of target words.
  - **Tags are not removed** (handled upstream).
  - **T2 AU length mark kept**: indic-nlp rewrites `ൗ` (U+0D57, `അക്കൗണ്ട്`) as the two-part
    `ൌ` (U+0D4C); train maps it back. Same sound, but the sravaani tokenizer ranks `ൗ` as the
    more frequent piece (eval_frozen: 215 `ൗ`, 0 `ൌ`). Eval is unaffected (symmetric).
  - **C2 guarded**: word-final `മ്`→`ം` is skipped right after a virama, so the filler `ഹ്മ്`
    is not turned into the malformed `ഹ്ം`.
  (Note: the sravaani tokenizer has no `0–9`, `-`, `%` pieces, so anything the external
  verbalizer leaves unverbalized becomes `<unk>`.)
- **`inference`** (aliases `infer`, `itn`) — **reader-facing output (ITN).** Keeps punctuation
  and case; number *words* become digits (`ഇരുപത്തിനാല്`→`24`, `ട്വൻറ്റി ഫോർ`→`24`,
  ordinals `നാലാം`→`4-ാം`); digits already present are left exactly as written
  (`4:00-5:00`, `2950 × 18%`). **I1:** a standalone `ഒന്ന്`/`ഒന്നു` is usually the softener
  "just/once" (`ഒന്ന് നോക്ക്`, `ഒന്ന് ചെക്ക് ചെയ്യ്`), so it becomes `1` only in numeric context —
  a neighbouring number, or a following unit/counter (`രൂപ`, `ദിവസം`, `വർഷം`, `ഇയർ`, `ലാക്`,
  `മിനിറ്റ്` …). On the call hypotheses this cut lone `1`s from 535 to 60.

A shared **canonicalization base** runs identically in all three modes.

| Step | `eval` | `train` | `inference` |
|------|--------|------------|-------------|
| E12 chillu-conjunct fix (`ൻ്റ`→`ന്റ`) | ✓ | ✓ | ✓ |
| E2 chillu / ZWJ canonicalization (indic-nlp)¹ | ✓ | ✓ | ✓ |
| E1 NFC · E3 `<unk>`/`⁇` drop · E4 zero-width drop | ✓ | ✓ | ✓ |
| E13 annotation tags (`<unintelligible>`) | drop | — (kept) | drop |
| E6 Indic-digit → ASCII (`൨൪`→`24`) | ✓ | ✓ | ✓ |
| C6 homoglyph `ഠ`→`0` inside a number | ✓ | ✓ | ✓ |
| T2 AU length mark `ൌ`→`ൗ` (undo indic-nlp) | — | ✓ | — |
| C2 word-final `മ്`→`ം` | ✓ | ✓ (not after a virama) | ✓ |
| E7 digit-grouping comma (`70,000`, `1,75,000`→`175000`) | ✓ | ✓ | ✓ |
| **T1 number-internal punctuation** (`4:00-5:00`, `2.5`, `70%`, `23-ന്`) | — (stripped) | **kept** | kept |
| **E5 punctuation removal** | strip | strip | **keep** |
| **E9 English-number words** | — (left as written) | — (left as written) | **→ digits** (via native words + ITN) |
| **E8 digits** | **→ native words** (≤7 digits as a number incl. thousands/lakhs; 8+ or leading 0 digit-by-digit) | — (left as digits) | — (left as digits) |
| **ITN number words → digits** | — | — | **✓** (ordinals only for fused suffixes; lone `ഒന്ന്` only in numeric context) |
| **E10 lowercase** | ✓ | ✓ | **keep case** |
| **C1 samvruthokaram fold** (`നാലു`→`നാല്`) | ✓ | — (target keeps real spelling) | — (would strip a real `-u` for a reader) |
| W3/W4/W5 segmentation (`segmentation=True`, default off) | opt-in | opt-in | — |
| E11 whitespace collapse | ✓ | ✓ | ✓ |

¹ **Colon guard.** indic-nlp rewrites any `:` after a Malayalam letter as the visarga `ഃ`. That
is only right *inside* a word (`ദു:ഖം` → `ദുഃഖം`). A colon followed by a space / end / non-letter
is punctuation (`ആണ്: …`, `പറ:`) and is protected from indic-nlp, so eval/train strip it and
inference keeps it. On the eval_frozen set this removed 12 spurious visargas (12 → 0).

**Inference ITN details.** The parser composes number-word units greedily (round-trips all
values 0–9999). A numeral fused with a suffix is converted **only for the ordinal family**
(`ാം`, `ാമത്`, `ാമത്തെ`, …: `പത്താം`→`10-ാം`, `ഇരുപത്തിയൊന്നാം`→`21-ാം`); other fusions are
idioms, not numbers (`ഒന്നും` "anything", `രണ്ടും` "both", `ഒന്നിച്ച്` "together") and are left
as words. A numeral spoken with its own `‑ു` ending is just the number (`രണ്ടു`→`2`). Words that
merely contain a number substring are never touched (`ഫ്രണ്ടിൽ` "front", `ആയിരിക്കും`).

`<unk>`/`⁇` is stripped in every mode; inference could instead map it to a marker such as `[?]`
if model uncertainty should be shown to the reader — an open option, not implemented.

---

## Missing — primarily affects **WER** (segmentation / boundary)

Highest leverage, because this is where the observed gap lives.

**Fix difficulty** (how mechanically tractable): **★ trivial** = one fixed regex/substitution,
no lexicon, no ambiguity · **★★ easy** = a small closed-set regex or table, few guard cases ·
**★★★ moderate** = curated whitelist/map + guards against real minimal pairs · **★★★★ hard**
= needs a sizeable lexicon or morphological analysis, real risk of collapsing distinctions.
Rows are ordered easiest-first within each table.

| # | Item | Why it hurts WER | Fix difficulty | Status / suggested fix |
|---|------|------------------|:--------------:|---------------|
| W4 | **Intra-word hyphen handling** (`covid-19` → `covid 19`) | Removing hyphen *to a space* splits one token into two; `covid-19`/`covid19`/`covid 19` land in three word counts. | **★ trivial** | **Implemented (opt-in, `segmentation=True`).** `_join_intra_word_hyphen`: `(?<=[^\s\d-])-(?=[^\s\d-])` → `` (letter–letter only; **digit-adjacent hyphens excluded** — see the numeral-suffix note below). |
| W3 | **Acronym / letter-sequence spacing** (`എ ടി എം` vs `എടിഎം` vs `ATM`) | Each convention is a different token count. | **★★ easy** | **Implemented (opt-in).** `_collapse_letter_runs`: collapse a run of ≥2 single-*grapheme* tokens into one (`എ ടി എം`→`എടിഎം`). Cross-script acronym map not added (no measured need). |
| W5 | **Reduplication / echo words** (`വേഗം വേഗം`) joined vs spaced | Boundary-only word-count mismatch. | **★★ easy** | **Implemented (opt-in).** `_collapse_reduplication`: collapse adjacent identical tokens to one; **number-words excluded** so id-like digit runs (`007`→`പൂജ്യം പൂജ്യം ഏഴ്`) are not eaten. |
| W2 | **English code-mixed token spacing** (`ഫോൺ നമ്പർ` vs `ഫോൺനമ്പർ`) | Inconsistent join of borrowed compounds → word-count mismatch. | **★★★ moderate** | *Gap.* Small join/split lexicon for frequent borrowed compounds — needs a curated list, no clean regex. |
| W1 | **Sandhi / compound & postposition split-merge** (`വീട്ടിലേക്ക്` vs `വീട്ടില് ലേക്ക്`) | Same phrase, different word count; the single biggest boundary contributor for agglutinative Malayalam. | **★★★★ hard** | *Gap.* Sandhi rewrites characters at the join, so splitting needs real morpheme boundaries (a wrong split corrupts genuine single words). Curated split/merge map, built + guarded carefully. |

> **W3/W4/W5 are OFF by default** (`normalize(..., segmentation=False)`). They are implemented,
> tested, and available behind the `segmentation` flag (and `analysis --segmentation`), but
> **measured on the real call + dataset corpora they are net-neutral-to-slightly-negative**
> (calls +0.25 WER / +0.09 CER, dataset +0.06 WER / +0.11 CER) — the boundary variants they
> target are rare here and each occasionally introduces a new mismatch. With the flag off the
> pipeline is byte-for-byte identical to before. Per-rule on calls: W5 ≈ −0.16→+0.03 (best),
> W3 ≈ neutral, W4 ≈ +0.10 (worst). Enable when a corpus is known to benefit.
>
> **Numeral + hyphenated suffix (tried, not shipped).** The linguistically correct fusion
> `24-ന്` → `ഇരുപത്തിനാലിന്`, `15-ാം` → `പതിനഞ്ചാം` (verbalize the number, attach the `-ന്`/`-ാം`
> suffix with sandhi) was implemented and measured — it made **WER/CER worse on both corpora**
> (calls +0.55 WER), because that pattern occurs only in the *references*, never the
> hypotheses, so fusing it on the ref side only creates new mismatches. Convergence beats
> linguistic correctness here, so W4 deliberately leaves digit-adjacent hyphens for E5 (the
> number stays a clean token E8 verbalizes; the suffix stays separate).

---

## Missing — primarily affects **CER** (character-level residue)

Reduces genuine residual error; most-per-line-of-code wins first. **Fix difficulty** uses
the same ★–★★★★ scale as the WER table above. Rows ordered easiest-first (C7 is resolved,
listed last).

C6/C2/C1 are **implemented**, applied symmetrically to ref and hyp. **C6 and C2 run in both
modes** (unambiguous); **C1 is eval-only** (a symmetric scoring fold that can touch a genuine
`-u` vowel, so it is not applied to reader-facing inference output). There is no flag — these
run automatically. C3/C4/C5 remain gaps.

| # | Item | Why it hurts CER | Fix difficulty | Status / suggested fix |
|---|------|------------------|:--------------:|---------------|
| C6 | **Visual-lookalike letters/digits** (`ഠ` used as zero in a number) | Homoglyph substitution inside a number. | **★ trivial** | **Implemented — both modes.** `_fix_homoglyphs`: `ഠ`→`0` only when digit-adjacent (`1ഠ0`→`100`); a standalone `ഠ` or one inside a word is untouched. Fires 0× in the current corpus but harmless. Latin `O`/`0` deliberately **not** mapped (too ambiguous). |
| C2 | **Anusvara vs nasal spelling** (word-final `മ്` vs `ം`) | Per-character substitution surviving current pipeline. | **★★ easy** | **Implemented — both modes.** `_fold_final_nasal`: word-final `മ്`→`ം` (`വരാമ്`→`വരാം`; tolerates trailing punctuation in inference, `വരാമ്.`→`വരാം.`). Only the labial `മ്`; the `ന്` nasal is left alone. Unambiguous. |
| C1 | **Samvruthokaram / final-vowel drift** (`നാല്`/`നാലു`; final `ു്`/`ു` → `്`) | Hits nearly every word ending → large, broad CER contributor. | **★★ easy** | **Implemented — `eval` mode only** (not `train`, not `inference`). `_fold_samvruthokaram`: word-final consonant+`ു്`/`ു` → consonant+`്` (`നാലു`→`നാല്`). **Symmetric scoring fold, not a spelling corrector** — it can't tell a samvruthokaram `ു` from a real `-u` vowel, so it also folds short words (`ഒരു`→`ഒര്`); fine for symmetric scoring but wrong for a reader, hence eval-only. It is the main CER win (see impact note). |
| C3 | **Conjunct / reph / `്യ`·`്വ` ordering variants** not unified by NFC | Multiple valid encodings of the same conjunct. | **★★★ moderate** | *Gap.* Mostly handled by indic-nlp already; the residual needs targeted per-conjunct rules found by audit — not a single regex. |
| C5 | **Foreign-word transliteration variants** (`ഫോൺ`/`ഫോണ്`, `ബസ്`/`ബസ്സ്`) | Same borrowed word, different spelling. | **★★★ moderate** | *Gap.* Curated canonical-spelling lexicon for frequent borrowings — inherently a hand-built list, no regex. |
| C4 | **Long-vowel / gemination spelling drift** (`ാ` vs `ാാ`, `ക്ക` vs `ക`) | Casual-transcription doubling/undoubling. | **★★★★ hard** | *Gap.* Gemination is *contrastive* in Malayalam (`കല` art vs `കല്ല` stone), so no blanket `CC→C` rule is safe — needs a conservative per-word whitelist. |
| <a name="c8"></a>C8 | **Bare (ZWJ-less) word-final chillu** — `consonant + virama` word-finally may not unify with atomic chillu. | Legacy/IME chillu without ZWJ mismatches an atomic-chillu reference. | **★ trivial** | *Not implemented (dropped by request).* Would be a 6-entry word-final map (`ന്`→`ൻ`, …), the inverse of E12. |
| <a name="c7"></a>C7 | ~~indic-nlp fallback parity gap~~ **RESOLVED** — colon→visarga and two-part vowel-sign composition used to fire only when indic-nlp-library was installed. | (was) an environment-dependent CER difference. | — (done) | **Fixed** by making `indic-nlp-library` a **hard dependency**. Single code path now; no parity gap remains. |

> **C1/C2 measured impact.** Unlike the W-family, the endings folds **help on every measured
> cell**: dataset **WER −1.09**, CER −0.14; calls WER −0.21, CER −0.06 (per-rule, C1 does most
> of it; C2 adds a little on dataset; C6 is 0×-but-safe here). They run automatically — no flag.

---

## Corpus-measured spoken-form variants (new — from ref↔hyp diff analysis)

These were found by diffing the reference and hypothesis of the real call + dataset
corpora (`call_ref/hyp.txt`, `dataset_ref/hyp.txt`), taking every token pair that **still
differs after normalization**, and classifying whether the two forms are the *same word*.
The pairs below are recurring, systematic same-word variants **not** covered by E1–E12 /
W1–W5 / C1–C8. Occurrence counts are over both corpora combined (≈2,000 distinct
post-normalization substitution types; ~48% were genuine ASR errors and excluded).

The dominant axis here is **spoken vs written Malayalam**: the reference is written-standard,
the hypothesis (ASR of conversational speech) captures the colloquial pronunciation. These
are a *different class* from the encoding/spelling items above — they are dialectal/register
variation, so canonicalizing them shades toward the "opinionated correction" line the
[Recommended order](#recommended-order) note warns against. Each is legitimate **only** as a
curated, conservative, symmetric rule (or a whitelist), never a blanket transform, and only
where the two forms are truly the same lexeme. Ranked by measured frequency.

Same difficulty scale as the tables above (**★ trivial** … **★★★★ hard**). These are
*register* rules, so even the regex-shaped ones carry the C4/C5 caveat below — cheap to
*write* is not the same as safe to *ship*. Rows ordered easiest-first.

| # | Item | Example (ref → hyp) | Occ. | Fix difficulty | Notes / suggested fix |
|---|------|--------------------|------|:--------------:|-----------------------|
| <a name="c14"></a>C14 | **English `-ing`: `-ംഗ്` ↔ `-ങ്`** | `ട്രെയിനിംഗ്`↔`ട്രെയിനിങ്`, `വെഡ്ഡിംഗ്`↔`വെഡ്ഡിങ്` | 48 | ★ trivial | Word-final `ംഗ്`↔`ങ്` is a single fixed substitution. A **C5 sub-case** (loanword spelling); fold into the borrowing lexicon. |
| <a name="c13"></a>C13 | **Colloquial nominalizer** `-ന്നത്` → `-ന്നേ` | `വരുന്നത്`→`വരുന്നേ`, `ചെയ്യുന്നത്`→`ചെയ്യുന്നേ` | 48 | ★★ easy | One word-final suffix regex `ന്നത്`→`ന്നേ` on a closed verbal ending; few guard cases. |
| <a name="c11"></a>C11 | **Copula / question reduction** `-ആണ്`/`-ആണോ` → `-ആ` | `എന്താണ്`→`എന്താ`, `എങ്ങനെയാണ്`→`എങ്ങനെയാ`, `ശരിയാണ്`→`ശരിയാ` | 142 | ★★ easy | Small closed-set ending rule on the `āṇ(u)` copula; canonicalize one direction. Same lexeme. |
| <a name="c15"></a>C15 | **Nasal assimilation `ൻപ`/`ൻ` ↔ `മ്പ`/`മ്`** | `ഒൻപത്`↔`ഒമ്പത്`, `മുൻപ്`↔`മുമ്പ്`, `മുൻപോട്ട്`↔`മുമ്പോട്ട്` | 45 | ★★ easy | Small closed map/regex (`ൻപ`↔`മ്പ`); both spellings current. A handful of high-frequency stems covers most of it. |
| <a name="c16"></a>C16 | **`യ്ക്ക` ↔ `ക്ക` sandhi** (dative/verb) | `ഉച്ചയ്ക്ക്`↔`ഉച്ചക്ക്`, `അടയ്ക്കാൻ`↔`അടക്കാൻ`, `അയയ്ക്കുക`↔`അയക്കുക` | 26 | ★★ easy | Regex to drop/insert glide `യ` before geminate `ക്ക`; small guard set. Related to C4. |
| <a name="c10"></a>C10 | **Final vowel-sign length `െ` ↔ `േ`** (short↔long e) | `ഓക്കെ`→`ഓക്കേ`, `പക്ഷെ`→`പക്ഷേ`, `അല്ലെ`→`അല്ലേ`, `അതേ`→`അതെ` | **645** | ★★★ moderate | Second largest. The rewrite is a one-line word-final `െ`↔`േ` regex, but real minimal pairs exist (particles vs. loanwords vs. inflections), so it needs a whitelist + guards, not a blanket rule. Sibling of C1 on the e-axis. |
| <a name="c9"></a>C9 | **Colloquial temporal contraction** `-പ്പോൾ`/`-പ്പം` → `-പ്പൊ`, `-മ്പോൾ` → `-മ്പം`/`-മ്പൊ` | `അപ്പോൾ`→`അപ്പൊ`, `ഇപ്പോൾ`→`ഇപ്പൊ`, `ചിലപ്പോൾ`→`ചിലപ്പം` | **944** | ★★★ moderate | Largest single gap, but the target forms scatter (`-പ്പൊ`/`-പ്പം`/`-മ്പം`) so it's a curated map over the *appōḷ/ippōḷ* closed word class canonicalizing to one form, not a single regex. |
| <a name="c12"></a>C12 | **Verb `വ` ↔ `ക` alternation** | `ഉണ്ടാവും`↔`ഉണ്ടാകും`, `പോവാൻ`↔`പോകാൻ`, `ആവും`↔`ആകും` | 98 | ★★★ moderate | Regular *va/ka* stem variation, but `വ`↔`ക` can't be rewritten blindly (distinct verbs collide) — needs a curated per-stem map with morphological guards. |

**Longer tail (lower frequency, same character):** loanword vowel `a↔e` drift
(`പേരന്റ്`/`പേരെന്റ്` — extends C5, ≈34); colloquial pronoun forms (`നമ്മൾക്ക്`/`നമുക്ക്`,
`എല്ലാരും`/`എല്ലാവരും` — borderline lexical, ≈51); dialectal `വെക്ക/വച്ച` *ai↔e*
(`വച്ച്`/`വെച്ച്` — borderline, ≈67); `-ിയിട്ട്`→`-ീട്ട്` contraction; `-ിയിൽ`→`-ീൽ`
contraction (`രീതിയിൽ`/`രീതില്`); `കൈയിൽ`/`കയ്യിൽ` metathesis; `ആഴ്ച്ച`/`ആഴ്ച`;
`പെട്ടന്ന്`/`പെട്ടെന്ന്`; `ശെരി`/`ശരി`; `എന്തേലും`/`എന്തെങ്കിലും`. These are candidates for
the same curated-whitelist treatment only if error analysis shows they matter.

> **Caveat (unchanged policy).** C9–C16 are *spoken-form/dialectal* canonicalizations, not
> encoding fixes. They collapse a real register distinction, so — like C4/C5 — they are
> double-edged for scoring: safe **only** as conservative, symmetric, whitelist-scoped rules
> on closed word classes, and only where the pair is unambiguously the same lexeme. Do **not**
> implement any of them as a general character rewrite. Measure prevalence (the analyzer's gap
> detectors) and CER impact before building. The two big ones (C9, C10) are the only ones
> whose volume clearly justifies the work today.

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

0. **Guardrail test** (above) — cheap, protects the wins already banked; prevents silently
   reintroducing the vanishing-vowels bug. (C7 fallback-parity is already **resolved** —
   indic-nlp-library is now a hard dependency, so there is a single code path.)
1. **W1 + W3** (postposition split-merge + acronym/letter-sequence collapse) — biggest WER-side wins.
2. **C1 + C2** (samvruthokaram + anusvara) — biggest CER-side wins per line of code, since they
   touch most word endings.
3. **C9 + C10** (colloquial temporal contraction + final `െ`/`േ` drift) — the two largest
   *measured* residual-CER contributors in the real corpora (≈944 + ≈645 occurrences), each a
   closed-class curated rule. See [Corpus-measured spoken-form variants](#corpus-measured-spoken-form-variants-new--from-refhyp-diff-analysis).
4. Remaining items (W2, W4, W5, C3–C6, C8, C11–C16) as they surface in error analysis.

> A **spacing-insensitive CER** was trialled as a metric-side way to neutralize W1 boundary
> noise, but on these corpora it moved the rate in both directions (stripping matched spaces
> shrinks the reference-length denominator), so it was dropped — prefer the W1 split/merge map.

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
| two-part vowel signs (`ൊ ോ ൌ`) | ✅ via indic-nlp (required) | ✅ | ✅ | ✅ |
| colon → visarga | ✅ via indic-nlp (required) | ✅ | ✅ | ✅ |
| Indic danda `।॥` removed (E5) | ✅ | ➖ | ❌ ASCII-only | ➖ |
| Indic digit → ASCII (E6) | ✅ | ❌ | ❌ | ❌ |
| **digit verbalization (E8)** | ✅ **unique** | ❌ | ❌ | ❌ |
| **English↔Malayalam number canon. (E9)** | ✅ **unique** | ❌ | ❌ | ❌ |
| Mark-class stripping (the bug) | ❌ correctly avoids | ❌ correctly avoids | ❌ correctly avoids | ⚠️ in non-Indic base path |
| opinionated typo/spelling "fix" | ❌ correctly avoids | ➖ opt-in | ⚠️ **on by default** | ➖ |

Notes:
- **indic-nlp-library** (Anoop Kunchukuttan, MIT) — the de-facto standard and our required
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
