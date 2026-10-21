"""Audio-based choice of number readings (RNN-T/TDT or CTC scoring with a NeMo model).

Needs the optional extra:  pip install "indic-normalizer[digits]"   (numpy, PyYAML, soundfile;
NeMo + torch for the model — install the NeMo build that matches your CUDA).

    from indic_normalizer.digits.scoring import load_config, NemoScorer, verbalize_dir, compare_dir
    cfg = load_config("digits_scoring.yaml")
    scorer = NemoScorer(cfg)
    verbalize_dir("eval_frozen", "audio/eval_frozen", scorer, cfg)
    compare_dir("eval_frozen", "audio/eval_frozen", scorer, cfg,
                hyps_dirs=["base_hypotheses-streaming", "base_hypotheses-segment"])

Command line: ``indic-digits verbalize|compare --config ... DATA_DIR AUDIO_DIR``.
"""

from .config import ScoringConfig, load_config


def __getattr__(name):            # lazy: importing the package never pulls torch / NeMo
    if name == "NemoScorer":
        from .nemo_scorer import NemoScorer
        return NemoScorer
    if name in ("verbalize_dir", "verbalize_document", "plan_document"):
        from . import pipeline
        return getattr(pipeline, name)
    if name == "compare_dir":
        from .compare import compare_dir
        return compare_dir
    raise AttributeError(name)


__all__ = ["ScoringConfig", "load_config", "NemoScorer", "verbalize_dir", "verbalize_document",
           "plan_document", "compare_dir"]
