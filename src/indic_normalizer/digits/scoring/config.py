"""Runtime configuration for audio-based number scoring (YAML or JSON file + CLI overrides).

Example (``examples/digits_scoring.yaml``)::

    model_path: /models/sravaani/v2.nemo   # NeMo hybrid RNN-T(TDT)/CTC checkpoint
    pretrained_name: null               # or a NeMo pretrained name instead of model_path
    cuda: null                          # int = that CUDA device; null = auto; negative = CPU
    allow_mps: true
    compute_dtype: float32              # float32 | bfloat16 | float16 (null = bf16 if available)
    matmul_precision: high              # highest | high | medium
    batch_size: 16                      # (audio, candidate) pairs per forward pass
    tdt_loss: torch                     # torch (pure torch, no numba/libNVVM) | nemo (NeMo's kernel)
    joint_chunk_frames: 16              # encoder frames per joint evaluation (memory bound)
    encoder_batch_s: 30                 # encoder batch budget: clips x longest clip <= this (padded s)
    group_audio_s: 300                  # recordings pooled per scoring group (s of audio; 0 = per file)
    decoding:                           # transcribe() only (compare --transcribe); merged over the
      greedy:                           #   model's own decoding config
        use_cuda_graph_decoder: false   # default false: CUDA graphs crashed after direct scoring
    head: tdt                           # tdt (RNN-T head) | ctc
    unit: segment                       # segment | word (window encoded alone) | window (sentence
                                        #   encoded, number's frames scored)
    pad_s: 0.25                         # audio padding around a segment / word window, seconds
    context_words: 1                    # word unit: neighbouring words included on each side
    max_candidates: 12                  # readings scored per number (first = default)
    style: contracted                   # default native compound style
    margin_threshold: 1.0               # nats; best-vs-best DIFFERENT reading below this -> needs_review
    style_margin: 2.0                   # nats; the audio's spelling (contracted / glide / spaced) is kept
                                        #   only if it beats the other spellings by this; else `style`
    dominant_language_share: 0.8        # a file with more than this share in one language reads every
                                        #   number in it; else per sentence (isolated switches -> review)
    hyp_pad_s: 0.5                      # compare only: hypothesis text within sentence +- this (s)
    length_norm: false                  # compare NLL per token instead of total NLL
    sample_rate: 16000
    out_dir: null                       # verbalize root: OUT_DIR/<data dir>/{verbalized,verify,failed}/<stem>.json + details
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

HEADS = ("tdt", "ctc")
UNITS = ("segment", "word", "window")


@dataclass
class ScoringConfig:
    model_path: Optional[str] = None
    pretrained_name: Optional[str] = None
    cuda: Optional[int] = None
    allow_mps: bool = True
    compute_dtype: Optional[str] = "float32"
    matmul_precision: str = "high"
    batch_size: int = 16
    tdt_loss: str = "torch"
    joint_chunk_frames: int = 16
    encoder_batch_s: float = 30.0
    group_audio_s: float = 300.0
    decoding: Optional[dict] = None
    head: str = "tdt"
    unit: str = "segment"
    pad_s: float = 0.25
    context_words: int = 1
    max_candidates: int = 12
    style: str = "contracted"
    margin_threshold: float = 1.0
    style_margin: float = 2.0
    dominant_language_share: float = 0.8    # see latin.sentence_languages
    hyp_pad_s: float = 0.5
    length_norm: bool = False
    sample_rate: int = 16000
    out_dir: Optional[str] = None

    def validate(self) -> "ScoringConfig":
        if self.head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {self.head!r}")
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, got {self.unit!r}")
        if self.tdt_loss not in ("torch", "nemo"):
            raise ValueError("tdt_loss must be torch | nemo")
        if self.hyp_pad_s < 0 or self.pad_s < 0:
            raise ValueError("pad_s and hyp_pad_s must be >= 0")
        if self.encoder_batch_s <= 0 or self.group_audio_s < 0:
            raise ValueError("encoder_batch_s must be > 0 and group_audio_s >= 0")
        if not 0.5 <= self.dominant_language_share <= 1:
            raise ValueError("dominant_language_share must be between 0.5 and 1")
        if self.style_margin < 0 or self.margin_threshold < 0:
            raise ValueError("style_margin and margin_threshold must be >= 0")
        if self.batch_size < 1 or self.max_candidates < 1:
            raise ValueError("batch_size and max_candidates must be >= 1")
        if self.compute_dtype not in (None, "float32", "bfloat16", "float16"):
            raise ValueError("compute_dtype must be float32 | bfloat16 | float16 | null")
        from ..ml import STYLES
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {self.style!r}")
        if self.joint_chunk_frames < 1:
            raise ValueError("joint_chunk_frames must be >= 1")
        if self.context_words < 0:
            raise ValueError("context_words must be >= 0")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[str] = None, **overrides) -> ScoringConfig:
    """Read a YAML/JSON config file (optional) and apply non-None overrides."""
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover
                raise ImportError("YAML config needs PyYAML: pip install 'indic-normalizer[digits]'") from e
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"config {path} must be a mapping")
    for old in ("output_suffix", "output_format", "report_dir"):    # removed options: ignore
        data.pop(old, None)
    if "output_dir" in data:                                        # old name of out_dir
        data.setdefault("out_dir", data.pop("output_dir"))
    known = {f.name for f in fields(ScoringConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    if data.get("out_dir"):
        data["out_dir"] = os.path.expanduser(str(data["out_dir"]))
    if data.get("model_path"):
        data["model_path"] = os.path.expanduser(str(data["model_path"]))
    return ScoringConfig(**data).validate()
