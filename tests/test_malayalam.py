"""Tests for the Malayalam normalizer — including the crown-jewel regression pins
(E12 conjunct fix, E8/E9 number canonicalization) and the vanishing-vowels guardrail.
"""

import pytest

from indic_normalizer import get_normalizer

norm, source = get_normalizer("ml")


def test_registered():
    assert source == "indic_normalizer.malayalam"
    for alias in ("ml", "mal", "malayalam", "Malayalam", " ML "):
        fn, src = get_normalizer(alias)
        assert src == "indic_normalizer.malayalam"


# --- E12: mis-encoded chillu conjunct (ൻ്റ -> ന്റ), NFC-invisible ---------------------
def test_chillu_conjunct_fix():
    # chillu-n + virama + rra  vs  na + virama + rra — same word, must normalize equal
    wrong = "വിവേകാനന്ദൻ്റെ"   # contains ൻ + ് + റ
    right = "വിവേകാനന്ദന്റെ"    # contains ന + ് + റ
    assert norm(wrong) == norm(right)


def test_genuine_word_final_chillu_untouched():
    # a real word-final chillu (no following virama) must survive, not be rewritten
    out = norm("അവൻ")
    assert "ൻ" in out  # chillu-n preserved


# --- E8 (eval): digits -> native words; train leaves numbers as written --------------
@pytest.mark.parametrize("digits,expected", [
    ("24", "ഇരുപത്തിനാല്"),
    ("500", "അഞ്ഞൂറ്"),
    ("21", "ഇരുപത്തിയൊന്ന്"),          # sandhi glide യ
    ("007", "പൂജ്യം പൂജ്യം ഏഴ്"),       # zero-padded / id-like: digit by digit
])
def test_eval_digit_verbalization(digits, expected):
    assert norm(digits) == expected


@pytest.mark.parametrize("text", ["24", "500", "007", "ഇരുപത്തിനാല്", "ട്വൻറ്റി ഫോർ"])
def test_train_leaves_numbers_as_written(text):
    assert train(text) == text


def test_indic_digits_folded():
    assert train("൨൪") == "24"          # E6 folds Malayalam digits in every mode
    assert norm("൨൪") == norm("24")


# --- E9: English-style number words -> canonical Malayalam ----------------------------
def test_english_number_canon_by_mode():
    # eval: digits are verbalized, but English-style number words are left as written
    assert norm("24") == norm("ഇരുപത്തിനാല്") == "ഇരുപത്തിനാല്"
    assert norm("ട്വൻറ്റി ഫോർ") == "ട്വൻറ്റി ഫോർ"
    # train: left as written
    assert train("ട്വൻറ്റി ഫോർ") == "ട്വൻറ്റി ഫോർ"
    # inference: all three meet as digits
    assert infer("ട്വൻറ്റി ഫോർ") == infer("24") == infer("ഇരുപത്തിനാല്") == "24"


def test_group_comma_collapsed():
    assert norm("70,000") == norm("70000")


def test_ambiguous_to_for_left_alone():
    # bare ടു / ഫോർ (English to/for) with no unambiguous number word must not convert to a
    # NUMBER word (E9). (C1 does fold the final ു -> ട്, but that's a separate rule; the point
    # here is only that it is NOT turned into രണ്ട്.)
    assert infer("പ്ലസ് ടു") == "പ്ലസ് ടു"      # E9 (inference) leaves bare ടു alone


# --- Guardrail: vanishing-vowels must NOT happen (ACL-2024 identity check) ------------
def test_no_vanishing_vowels():
    # a plain Malayalam sentence keeps its word count and its vowel signs
    s = "ഇതൊരു ഉദാഹരണമാണ്"
    out = norm(s)
    assert len(out.split()) == len(s.split())   # word count preserved (no vowel-split)
    for vowel_sign in ("ൊ", "ാ"):
        assert vowel_sign in out                # mātras survive


def test_punctuation_and_unk_removed():
    assert "!" not in norm("പൗരൻ!!")
    assert "⁇" not in norm("foo ⁇ bar")


# --- Mode selector: eval (default, verbalize) vs inference (ITN) ----------------------
from indic_normalizer import normalize as _normalize
from indic_normalizer.languages import malayalam as _mal

infer, infer_src = get_normalizer("ml", mode="inference")
train, _ = get_normalizer("ml", mode="train")


def test_inference_registered_same_source():
    assert infer_src == "indic_normalizer.malayalam"


def test_default_mode_is_eval():
    # get_normalizer("ml") with no mode == eval: punctuation stripped, numbers verbalized
    assert norm("24 രൂപ.") == "ഇരുപത്തിനാല് രൂപ"


@pytest.mark.parametrize("mode", ["eval", "evaluation"])
def test_eval_aliases(mode):
    fn, _ = get_normalizer("ml", mode=mode)
    assert fn("24 രൂപ.") == "ഇരുപത്തിനാല് രൂപ"


@pytest.mark.parametrize("mode", ["train", "training", "fine-tune", "finetune"])
def test_train_aliases(mode):
    fn, _ = get_normalizer("ml", mode=mode)
    assert fn("24 രൂപ.") == "24 രൂപ"        # same canonicalization, numbers untouched


# inference: every number convention -> digits
@pytest.mark.parametrize("text,expected", [
    ("24", "24"),
    ("ഇരുപത്തിനാല്", "24"),          # native words -> digits (ITN)
    ("ട്വൻറ്റി ഫോർ", "24"),          # English-style words -> digits (E9 still runs, then ITN)
    ("അഞ്ഞൂറ്", "500"),
    ("ഇരുപത്തിയൊന്ന്", "21"),        # sandhi-fused form reversed
])
def test_inference_numbers_to_digits(text, expected):
    assert infer(text) == expected


def test_inference_digit_by_digit_run():
    assert infer("പൂജ്യം പൂജ്യം ഏഴ്") == "007"       # id-like run reversed


def test_inference_preserves_punctuation_and_case():
    # E5 (punctuation) and E10 (case) are OFF in inference -> readable transcript
    assert infer("ഇരുപത്തിനാല്, ശരി?") == "24, ശരി?"
    assert infer("Hello World") == "Hello World"      # case kept (eval would lowercase)


def test_inference_number_adjacent_to_punct():
    assert infer("(ഏഴ്)") == "(7)"
    assert infer("അമ്പത് രൂപ.") == "50 രൂപ."


def test_inference_leaves_non_numbers_untouched():
    assert infer("ഒരു കോൾ ഇരുപത്തിനാല് മണിക്കൂർ") == "ഒരു കോൾ 24 മണിക്കൂർ"


def test_eval_and_inference_share_encoding_base():
    # the E12 chillu fix (shared) applies in BOTH modes
    wrong, right = "വിവേകാനന്ദൻ്റെ", "വിവേകാനന്ദന്റെ"
    assert infer(wrong) == infer(right)


def test_verbalize_itn_roundtrip_all_values():
    # ITN is the exact inverse of verbalization for every covered value, by construction
    for n in range(0, 10000):
        assert _mal._to_digits(_mal._num_words(n)) == str(n)


def test_one_shot_normalize_mode():
    assert _normalize("24 രൂപ.", lang="ml") == "ഇരുപത്തിനാല് രൂപ"      # eval
    assert _normalize("24 രൂപ.", lang="ml", mode="train") == "24 രൂപ"
    assert _normalize("24 രൂപ.", lang="ml", mode="inference") == "24 രൂപ."


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        get_normalizer("ml", mode="bogus")


# --- C-family CER folds: C6/C2 in BOTH modes; C1 eval-only ----------------------------

# C6 homoglyph: ഠ used as zero inside a number -> 0
def test_c6_homoglyph_in_number():
    assert train("1ഠ0") == "100"
    assert norm("1ഠ0") == "നൂറ്"                 # eval then verbalizes the 100


def test_c6_standalone_letter_untouched():
    # ഠ not adjacent to a digit is the real letter -> left alone (inside a word)
    assert "ഠ" in norm("ഠമരം")


def test_c6_runs_in_both_modes():
    # C6 is unambiguous -> applies in inference too (1ഠ0 -> 100, then kept as digits)
    assert infer("1ഠ0") == "100"


# C2 word-final മ് -> ം (both modes)
def test_c2_final_ma_to_anusvara():
    assert norm("വരാമ്") == "വരാം"
    assert norm("വരാമ്") == norm("വരാം")          # convergence


def test_c2_runs_in_both_modes():
    # C2 is unambiguous -> also folds in inference, and tolerates trailing punctuation there
    assert infer("വരാമ്") == "വരാം"
    assert infer("വരാമ്.") == "വരാം."             # punctuation kept, മ് still folded


def test_c2_leaves_chillu_n_alone():
    assert "ൻ" in norm("അവൻ") or "ന്" in norm("അവന്")   # C2 does not touch the n-nasal


# C1 samvruthokaram / final-vowel drift (EVAL ONLY)
def test_c1_samvruthokaram_folded():
    assert norm("നാലു") == norm("നാല്")          # ...ു -> ...് ; same word
    assert norm("അവനു") == "അവന്"


def test_c1_only_touches_consonant_u_ending():
    # C1 fires only on a CONSONANT + word-final ു; an ി-ending word is untouched.
    assert norm("പോയി") == "പോയി"


def test_c1_is_a_symmetric_scoring_fold():
    # HONESTY: C1 cannot orthographically tell a samvruthokaram ു from a genuine -u vowel,
    # so it also folds real short words (ഒരു -> ഒര്). That's acceptable because it is applied
    # SYMMETRICALLY to ref and hyp — both sides fold identically, so they still match and the
    # measured WER/CER improves. It is a scoring canonicalization, not a spelling corrector.
    assert norm("ഒരു") == "ഒര്"
    assert norm("ഒരു") == norm("ഒര്")          # ref/hyp still converge either way


def test_c1_is_eval_only():
    # The C1 samvruthokaram FOLD must not run in inference on a non-number word: അവനു
    # ("to him") is not a numeral, so inference leaves it untouched (eval would fold -> അവന്).
    assert infer("അവനു") == "അവനു"
    assert norm("അവനു") == "അവന്"       # eval C1 fold still applies
    # (നാലു IS the number four; inference ITN converts it to 4 — see test_itn_fused_*)


# --- Inference ITN: fused numeral + suffix (greedy number-word parser) ----------------
def test_itn_pure_numeral_to_digits():
    assert infer("ഇരുപത്തിനാല്") == "24"
    assert infer("അഞ്ഞൂറ്") == "500"
    assert infer("നാലു") == "4"              # "four" with spoken samvruthokaram -> 4, no dangling -ു


def test_itn_fused_ordinal():
    assert infer("നാലാം") == "4-ാം"            # fourth
    assert infer("മൂന്നാം") == "3-ാം"           # third
    assert infer("പത്താം") == "10-ാം"           # tenth
    assert infer("രണ്ടാമത്തെ") == "2-ാമത്തെ"    # second
    assert infer("ഇരുപത്തിയൊന്നാം") == "21-ാം"  # 21st: ordinal on the last unit, after a glide


def test_itn_spoken_u_numeral():
    assert infer("രണ്ടു") == "2"               # രണ്ടു ദിവസം "2 days"
    assert infer("പത്തു") == "10"


def test_itn_non_ordinal_fusions_left_as_words():
    # ORDINALS ONLY: numeral-prefixed idioms are words, not numbers -> untouched
    for w in ["ഒന്നും", "രണ്ടും", "ഒന്നുമില്ല", "ഒന്നിച്ച്", "ഒന്നുകിൽ", "ഒന്നെങ്കിൽ",
              "രണ്ടുമൂന്ന്", "ഒന്നാണ്", "മുപ്പത്തിയല്ലേ", "ഒന്നുകൂടെ"]:
        assert infer(w) == w, f"{w!r} should stay a word, got {infer(w)!r}"


def test_itn_does_not_corrupt_real_words():
    # words that merely CONTAIN a number substring but don't START with a number unit
    for w in ["ഫ്രണ്ടിൽ", "ഫ്രണ്ടിലോട്ട്", "ആയിരിക്കും", "തമ്പാനൂറിന്റെ", "ഹലോ", "ഷെയറിങ്"]:
        assert infer(w) == w, f"{w!r} should be untouched, got {infer(w)!r}"


def test_itn_runon_components_summed():
    # a run-together multi-component number composes additively (deterministic)
    assert infer("ഇരുനൂറ്റിയെണ്ണൂറ്റിയൊമ്പത്") == "1009"   # 200+800+9


# --- W3/W4/W5: segmentation rules (eval-only, opt-in via segmentation=True) -----------
# OFF by default (measured net-neutral-to-negative on the real corpora), so they only fire
# when explicitly enabled.
def segnorm(t):
    return _mal.normalize(t, mode="eval", segmentation=True)


def test_segmentation_off_by_default():
    # default eval leaves the boundary variants alone
    assert norm("e-mail") == norm("e mail")                # hyphen -> space via E5, not joined
    assert norm("എ ടി എം") == "എ ടി എം"                    # single letters NOT collapsed
    assert norm("വേഗം വേഗം") == "വേഗം വേഗം"                # duplicate NOT collapsed


# W4 intra-word hyphen join
def test_w4_letter_hyphen_joined():
    assert segnorm("e-mail") == "email"
    assert segnorm("സ്കിൽ-ബേസ്ഡ്") == "സ്കിൽബേസ്ഡ്"
    assert segnorm("വൺ-ഓൺ-വൺ") == "വൺഓൺവൺ"


def test_w4_hyphenated_variants_converge():
    assert segnorm("സ്കിൽ-ബേസ്ഡ്") == segnorm("സ്കിൽബേസ്ഡ്")


def test_w4_digit_hyphen_not_joined():
    # a numeral + suffix (24-ന്) must NOT fuse; the hyphen falls through to E5 as a separator
    assert segnorm("24-ന്") == "ഇരുപത്തിനാല് ന്"           # eval: number verbalized, suffix separate
    assert _mal.normalize("24-ന്", mode="train", segmentation=True) == "24-ന്"   # train keeps it (T1)


# W3 single-letter acronym run collapse
def test_w3_letter_run_collapsed():
    assert segnorm("എ ടി എം") == "എടിഎം"
    assert segnorm("ഒ ടി പി വേണം") == "ഒടിപി വേണം"


def test_w3_acronym_variants_converge():
    assert segnorm("എ ടി എം") == segnorm("എടിഎം")


def test_w3_lone_word_untouched():
    # W3 must not collapse a lone multi-letter word (each is >1 grapheme). Both words survive
    # as separate tokens; C1 folds the final ു of ഒരു (-> ഒര്), which is fine — the point is W3
    # did not merge them.
    out = _mal.normalize("ഒരു ദിവസം", mode="eval", segmentation=True)
    assert len(out.split()) == 2 and out.endswith("ദിവസം")


# W5 reduplication collapse
def test_w5_adjacent_identical_collapsed():
    assert segnorm("വേഗം വേഗം വരൂ") == "വേഗം വരൂ"
    assert segnorm("പെട്ടെന്ന് പെട്ടെന്ന്") == "പെട്ടെന്ന്"


def test_w5_reduplication_variants_converge():
    assert segnorm("വേഗം വേഗം") == segnorm("വേഗം")


def test_w5_non_adjacent_repeat_kept():
    # non-adjacent repeat (അവൻ … അവൻ) is a normal sentence, not reduplication -> both kept.
    # (C1 folds വന്നു->വന്ന്; the assertion checks W5 kept both അവൻ, not the -u ending.)
    out = _mal.normalize("അവൻ വന്നു അവൻ പോയി", mode="eval", segmentation=True)
    assert out.split().count("അവൻ") == 2      # both occurrences survive (not collapsed)


def test_w5_does_not_eat_repeated_digit_words():
    # a spoken digit-by-digit id ("zero zero seven") must keep both പൂജ്യം
    assert segnorm("പൂജ്യം പൂജ്യം ഏഴ്") == "പൂജ്യം പൂജ്യം ഏഴ്"
    assert segnorm("007") == "പൂജ്യം പൂജ്യം ഏഴ്"     # eval's own digit-by-digit output too


def test_segmentation_ignored_in_inference():
    # segmentation has no effect in inference mode (reader-facing transcript kept intact)
    assert _mal.normalize("എ ടി എം വേഗം വേഗം", mode="inference", segmentation=True) \
        == "എ ടി എം വേഗം വേഗം"


# --- Colon vs visarga, and inference leaving digits alone ------------------------------
def test_punctuation_colon_is_not_visarga():
    assert norm("ആണ്: ഷാങ്ഹായ്") == "ആണ് ഷാങ്ഹായ്"     # eval strips the colon
    assert infer("പറ:") == "പറ:"                         # inference keeps it
    assert "ഃ" not in infer("ആണ്: ഷാങ്ഹായ്")


def test_midword_colon_is_visarga():
    assert norm("ദു:ഖം") == "ദുഃഖം"                      # ':' typed for ഃ inside a word


def test_inference_keeps_existing_digits():
    assert infer("4:00-5:00") == "4:00-5:00"
    assert infer("2950 × 18%") == "2950 × 18%"


def test_train_mode_canonical_name():
    assert _mal._resolve_mode("training") == "train"
    assert _mal._resolve_mode("fine-tune") == "train"


# --- Lakh grouping (E7), all modes ---------------------------------------------------
def test_lakh_grouping_commas_removed_all_modes():
    assert train("1,75,000 രൂപ") == "175000 രൂപ"
    assert train("12,50,000") == "1250000"
    assert train("70,000") == "70000"
    assert train("1,000,000") == "1000000"
    assert infer("1,75,000 രൂപ") == "175000 രൂപ"
    assert train("1,2,3") == "1 2 3"          # a list, not a grouping


# --- Train: C1 off, number-internal punctuation kept (T1) ---------------------------
def test_train_keeps_samvruthokaram():
    assert train("ഒരു കാര്യം പറഞ്ഞു") == "ഒരു കാര്യം പറഞ്ഞു"
    assert norm("ഒരു കാര്യം പറഞ്ഞു") == "ഒര് കാര്യം പറഞ്ഞ്"


def test_train_keeps_number_internal_punctuation():
    assert train("4:00-5:00 വരെ.") == "4:00-5:00 വരെ"
    assert train("2.5 ലക്ഷം, 70% ആയി") == "2.5 ലക്ഷം 70% ആയി"
    assert train("23-ന് 8-9 ദിവസം") == "23-ന് 8-9 ദിവസം"
    assert train("1/2") == "1/2"
    assert train("ശരി. - അതെ") == "ശരി അതെ"     # ordinary punctuation still stripped


def test_train_keeps_tags():
    assert train("<unintelligible> ഹലോ") == "unintelligible ഹലോ"


# --- Eval: large numbers read as numbers, tags removed ------------------------------
def test_eval_thousands_and_lakhs():
    assert norm("12000") == "പന്ത്രണ്ടായിരം"
    assert norm("10000") == "പതിനായിരം"
    assert norm("20000") == "ഇരുപതിനായിരം"
    assert norm("25000") == "ഇരുപത്തിയഞ്ചായിരം"
    assert norm("90000") == "തൊണ്ണൂറായിരം"
    assert norm("12500") == "പന്ത്രണ്ടായിരത്തിയഞ്ഞൂറ്"
    assert norm("200000") == "രണ്ട് ലക്ഷം"
    assert norm("1,75,000") == "ഒര് ലക്ഷത്തി എഴുപത്തിയഞ്ചായിരം"   # ഒരു -> ഒര് via C1


def test_eval_ids_still_digit_by_digit():
    assert norm("12345678") == "ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച് ആറ് ഏഴ് എട്ട്"
    assert norm("01234") == "പൂജ്യം ഒന്ന് രണ്ട് മൂന്ന് നാല്"


def test_eval_and_inference_remove_tags():
    assert norm("<unintelligible> ഹലോ") == "ഹലോ"
    assert infer("<unintelligible> ഹലോ.") == "ഹലോ."
    assert norm("<noise> ശരി <unk>") == "ശരി"


# --- Inference: lone ഒന്ന് only in numeric context (I1) ------------------------------
def test_inference_lone_one_is_a_word():
    assert infer("ഒന്ന് നോക്കിയിട്ട് പറയൂ") == "ഒന്ന് നോക്കിയിട്ട് പറയൂ"
    assert infer("ഒന്നു ചെക്ക് ചെയ്യ്.") == "ഒന്നു ചെക്ക് ചെയ്യ്."


def test_inference_lone_one_numeric_context():
    assert infer("ഒന്ന് രൂപ") == "1 രൂപ"
    assert infer("ഒന്ന് ഇയർ ആയി") == "1 ഇയർ ആയി"
    assert infer("ഒന്ന് രണ്ട് മൂന്ന്") == "123"
    assert infer("വൺ") == "1"


# --- Train: AU length mark kept (T2), C2 not after a virama ------------------------
def test_train_keeps_au_length_mark():
    assert train("അക്കൗണ്ട്") == "അക്കൗണ്ട്"          # U+0D57 kept
    assert train("അക്കൌണ്ട്") == "അക്കൗണ്ട്"          # two-part sign -> modern form
    assert norm("അക്കൗണ്ട്") == norm("അക്കൌണ്ട്")     # eval: still the same either way


def test_train_c2_not_after_virama():
    assert train("ഹ്മ്") == "ഹ്മ്"
    assert train("അവിടെയാണ് പണം") == "അവിടെയാണ് പണം"
    assert train("വേണം വേണമ്") == "വേണം വേണം"         # ordinary word-final മ് still folds


# --- Inference ITN v2: inverse of the digits generator (malayalam_itn) ---------------
@pytest.mark.parametrize("text,expected", [
    # lakhs / crores in Indian grouping
    ("രണ്ട് ലക്ഷത്തി ഇരുപത്തിയഞ്ചായിരം", "2,25,000"),
    ("ടു ലാക് ട്വന്റി ഫൈവ് തൗസന്റ്", "2,25,000"),
    ("വൺ തൗസന്റ് ടു ഹൺഡ്രഡ്", "1200"),
    # suffixes: 10 and up -> 27-ന്
    ("ഇരുപത്തിയേഴിന്", "27-ന്"),
    ("എയ്റ്റിലേക്കാണ്", "8-ലേക്കാണ്"),
    ("ട്വന്റി എയ്റ്റിലാണ്", "28-ലാണ്"),
    ("ട്വന്റിയിലേക്ക്", "20-ലേക്ക്"),
    # halves / time
    ("മൂന്നരയ്ക്ക് വരാം", "3:30-ന് വരാം"),
    ("ഒരു മണി", "1 മണി"),
    # percent, k, ambiguous ടു / ഫോർ with a numeric neighbour
    ("ടു പെർസെന്റ്", "2%"),
    ("ട്വന്റി കെ", "20k"),
    ("ടു കെ", "2k"),
    ("ഫോർ ഓ ക്ലോക്ക്", "4 ഓ ക്ലോക്ക്"),
    # ranges stay two numbers
    ("ടെൻ തൗസന്റ് ഫിഫ്റ്റീൻ തൗസന്റ്", "10000 15000"),
])
def test_itn_v2_conversions(text, expected):
    assert infer(text) == expected


@pytest.mark.parametrize("text", [
    "ആയിരിക്കും", "ആയിരക്കണക്കിന്", "ഡബിൾ റൂം", "ഫസ്റ്റ് ടൈം",
    "ഒരു കോൾ", "ഒന്ന് നോക്കട്ടെ", "ടു ദി", "ഫോർ യു",
])
def test_itn_v2_guards_left_as_words(text):
    assert infer(text) == text


def test_itn_v2_digits_roundtrip_native():
    from indic_normalizer.languages import malayalam_itn as _itn
    from indic_normalizer.digits import ml as _ml
    for n in list(range(0, 1000)) + [1200, 9999, 12500, 45678, 99999]:
        assert _itn.itn(_ml.words(n)) == _itn._fmt(n), n
        if n not in (2, 4):                                   # lone ടു / ഫോർ = "to" / "for"
            assert _itn.itn(_ml.en_words(n)) == _itn._fmt(n), n
