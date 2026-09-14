"""Per-language normalizers. Each module exposes ``normalize(text) -> str``.

Add a language by dropping ``<lang>.py`` here and registering it in
``indic_normalizer.registry`` (alias table + a branch in ``_load``).
"""
