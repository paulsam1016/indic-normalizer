"""Browser verifier for verbalize runs (``indic-verifier``): check the numbers verbalize was unsure
about, change numbers already decided, and fix the text of any sentence.

Standard library only (no GPU, torch or numpy). It works on a run dir written by
``indic_normalizer.digits`` (``indic-digits verbalize``); see ``server`` for the folders and files.
"""
