"""Moved to ``indic_normalizer.verifier``; this alias keeps the old import path working."""
import sys

from ... import verifier as _verifier
from ...verifier import server, ui  # noqa: F401

sys.modules[__name__ + ".server"] = server
sys.modules[__name__ + ".ui"] = ui
__doc__ = _verifier.__doc__
