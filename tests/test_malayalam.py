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


# --- E8: native digit verbalization --------------------------------------------------
@pytest.mark.parametrize("digits,expected", [
    ("24", "ഇരുപത്തിനാല്"),
    ("500", "അഞ്ഞൂറ്"),
    ("21", "ഇരുപത്തിയൊന്ന്"),   # sandhi glide യ
])
def test_digit_verbalization(digits, expected):
    assert norm(digits) == expected


def test_id_like_digits_read_digit_by_digit():
    # zero-padded / long runs -> digit by digit
    assert norm("007") == "പൂജ്യം പൂജ്യം ഏഴ്"


def test_indic_digits_folded_then_verbalized():
    assert norm("൨൪") == norm("24")  # Malayalam 24 -> same as ASCII 24


# --- E9: English-style number words -> canonical Malayalam ----------------------------
def test_english_number_canonicalization():
    # "24", English-in-Malayalam-script, and native Malayalam all collapse to one form
    canonical = norm("24")
    assert norm("ട്വൻറ്റി ഫോർ") == canonical
    assert norm("ഇരുപത്തിനാല്") == canonical


def test_group_comma_collapsed():
    assert norm("70,000") == norm("70000")


def test_ambiguous_to_for_left_alone():
    # bare ടു / ഫോർ (English to/for) with no unambiguous number word must not convert to a
    # NUMBER word (E9). (C1 does fold the final ു -> ട്, but that's a separate rule; the point
    # here is only that it is NOT turned into രണ്ട്.)
    out = norm("പ്ലസ് ടു")
    assert "രണ്ട്" not in out    # not converted to the number two
    assert out.split()[-1] == "ട്"   # E9 left it as the (C1-folded) word, not a numeral


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


def test_inference_registered_same_source():
    assert infer_src == "indic_normalizer.malayalam"


def test_default_mode_is_eval():
    # get_normalizer("ml") with no mode == the eval verbalizing normalizer
    assert norm("24") == "ഇരുപത്തിനാല്"


@pytest.mark.parametrize("mode", ["eval", "evaluation", "training", "fine-tune"])
def test_eval_aliases_verbalize(mode):
    fn, _ = get_normalizer("ml", mode=mode)
    assert fn("24") == "ഇരുപത്തിനാല്"        # all eval aliases -> native words


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
    assert _normalize("24 രൂപ.", lang="ml", mode="inference") == "24 രൂപ."


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        get_normalizer("ml", mode="bogus")


# --- C-family CER folds: C6/C2 in BOTH modes; C1 eval-only ----------------------------

# C6 homoglyph: ഠ used as zero inside a number -> 0 (then verbalized)
def test_c6_homoglyph_in_number():
    assert norm("1ഠ0") == norm("100")            # ഠ -> 0 -> നൂറ്
    assert norm("1ഠ0") == "നൂറ്"


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
    assert infer("നാലാം") == "4-ാം"          # fourth
    assert infer("മൂന്നാം") == "3-ാം"         # third


def test_itn_fused_particle():
    assert infer("മുപ്പത്തിയല്ലേ") == "30-അല്ലേ"   # "thirty, isn't it" (reverse glide restores അ)
    assert infer("ഒന്നുകൂടെ") == "1-കൂടെ"         # stranded samvruthokaram dropped, suffix kept


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
    # a numeral + suffix (24-ന്) must NOT fuse — the number stays its own token so E8 can
    # verbalize it cleanly (measured to hurt WER otherwise). Hyphen falls through to E5.
    out = segnorm("24-ന്")
    assert out == "ഇരുപത്തിനാല് ന്"          # number verbalized, suffix separate
    assert "ഇരുപത്തിനാല്ന്" not in out        # NOT glued


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
    # E8 spells 007 digit-by-digit; W5 must NOT collapse the repeated പൂജ്യം
    assert segnorm("007") == "പൂജ്യം പൂജ്യം ഏഴ്"


def test_segmentation_ignored_in_inference():
    # segmentation has no effect in inference mode (reader-facing transcript kept intact)
    assert _mal.normalize("എ ടി എം വേഗം വേഗം", mode="inference", segmentation=True) \
        == "എ ടി എം വേഗം വേഗം"
