"""Score candidate texts against audio with a NeMo hybrid RNN-T (TDT) / CTC model.

Model loading follows AetherTranscribe's SegmentASR / load_base_asr_model (NeMo Speech 3.0
public APIs): ``transcribe_utils.setup_model`` restores the .nemo on CPU, ``model.freeze()``,
cast to ``compute_dtype`` on the host, then one move to the device chosen by
``get_inference_device(cuda, allow_mps)``; grad off and TF32 matmul precision set globally.

The score is the negative log-likelihood -log p(text | audio) in nats, from either head:
    tdt  the RNN-T/TDT transducer head (model.decoder + model.joint + model.loss)
    ctc  the auxiliary CTC head (model.ctc_decoder + CTC loss)
The transducer NLL is computed by ``tdt_loss`` in plain torch by default (``tdt_loss: torch``) —
NeMo's own RNNT/TDT loss is a numba CUDA kernel that needs libNVVM from a full CUDA toolkit;
``tdt_loss: nemo`` uses it anyway (includes TDT's training-time sigma). The joint is evaluated
``joint_chunk_frames`` frames at a time so the full (T, U, V) tensor is never held.
Lower is better. Both are exact sequence likelihoods (summed over all alignments), so
candidates of different length are comparable on the same audio; ``length_norm`` in the
config switches the comparison to NLL per token. Joint / CTC logits are taken to float32
before the loss even when the model runs in bfloat16.

``transcribe()`` runs the model's normal greedy decoding (as SegmentASR does) — used by
``compare`` to measure how often each pick matches the model's own transcript.

Any object with the same ``score(pairs, head)`` method can replace this class (tests use a
fake scorer).
"""

from __future__ import annotations

import contextlib
from typing import List, Sequence, Tuple


def _load_model(cfg, device, dtype):
    from omegaconf import OmegaConf
    try:
        from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
        mcfg = OmegaConf.create({"model_path": cfg.model_path, "pretrained_name": cfg.pretrained_name})
        model, _ = setup_model(mcfg, __import__("torch").device("cpu"))
    except ImportError:                                    # older NeMo without setup_model
        from nemo.collections.asr.models import ASRModel
        if cfg.model_path:
            model = ASRModel.restore_from(cfg.model_path, map_location="cpu")
        else:
            model = ASRModel.from_pretrained(cfg.pretrained_name, map_location="cpu")
    model.freeze()
    return model.to(dtype).to(device).eval()


class NemoScorer:
    def __init__(self, cfg):
        if not cfg.model_path and not cfg.pretrained_name:
            raise ValueError("set model_path (or pretrained_name) in the config file")
        try:
            import torch
            from nemo.collections.asr.parts.utils.transcribe_utils import (
                get_inference_device, get_inference_dtype)
        except ImportError as e:
            raise ImportError("RNN-T scoring needs NeMo + torch: pip install 'indic-normalizer[digits]'") from e
        self.torch = torch
        self.cfg = cfg
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        self.device = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)
        self.dtype = get_inference_dtype(cfg.compute_dtype, device=self.device)
        self.model = _load_model(cfg, self.device, self.dtype)
        self._set_decoding(getattr(cfg, "decoding", None) or {})
        self.batch_size = cfg.batch_size
        self.sample_rate = int(self.model.cfg.preprocessor["sample_rate"])
        if self.sample_rate != int(cfg.sample_rate):
            raise ValueError(f"model expects {self.sample_rate} Hz audio, config says {cfg.sample_rate}")
        self.heads = ["tdt"] if getattr(self.model, "joint", None) is not None else []
        if getattr(self.model, "ctc_decoder", None) is not None:
            self.heads.append("ctc")
        self.tokenizer = self.model.tokenizer
        from collections import Counter
        self.stats = Counter()                  # batching / timing counters (see stats_report)
        self.tdt_loss = getattr(cfg, "tdt_loss", "torch")
        self.chunk = int(getattr(cfg, "joint_chunk_frames", 32))
        self.durations = self._durations()
        self.blank = int(getattr(self.model.decoder, "blank_idx", self.tokenizer.vocab_size)) \
            if "tdt" in self.heads else None

    def _set_decoding(self, overrides: dict):
        """Greedy decoding for transcribe(), as AetherTranscribe's SegmentASR sets it up, but with
        the CUDA-graph decoder OFF by default: it can hit 'illegal memory access' when the decoder /
        joint were also run directly (scoring) on the same model. Config ``decoding`` overrides."""
        if not hasattr(self.model, "change_decoding_strategy"):
            return
        from omegaconf import OmegaConf, open_dict
        dec = OmegaConf.create(OmegaConf.to_container(self.model.cfg.decoding, resolve=True))
        base = {"strategy": "greedy_batch",
                "greedy": {"loop_labels": True, "use_cuda_graph_decoder": False,
                           "preserve_alignments": False}}
        with open_dict(dec):
            dec = OmegaConf.merge(dec, OmegaConf.create(base), OmegaConf.create(overrides))
        try:
            self.model.change_decoding_strategy(dec, decoder_type="rnnt", verbose=False)
        except TypeError:
            try:
                self.model.change_decoding_strategy(dec, decoder_type="rnnt")
            except TypeError:
                self.model.change_decoding_strategy(dec)

    def _durations(self):
        joint = getattr(self.model, "joint", None)
        n_extra = int(getattr(joint, "num_extra_outputs", 0) or 0) if joint is not None else 0
        if not n_extra:
            return None
        c = self.model.cfg
        for get in (lambda: c.model_defaults.tdt_durations, lambda: c.loss.tdt_kwargs.durations,
                    lambda: c.decoding.durations):
            try:
                d = list(get())
                if len(d) == n_extra:
                    return [int(x) for x in d]
            except Exception:
                continue
        raise ValueError(f"model has {n_extra} duration outputs but no tdt_durations in its config")

    # ------------------------------------------------------------------ helpers
    def n_tokens(self, text: str) -> int:
        return max(1, len(self.tokenizer.text_to_ids(text)))

    @contextlib.contextmanager
    def _loss_reduction_none(self):
        loss = self.model.loss
        old = getattr(loss, "reduction", "mean_batch")
        loss.reduction = None
        try:
            yield loss
        finally:
            loss.reduction = old

    def _encode(self, A, a_len):
        return self.model.forward(input_signal=A, input_signal_length=a_len)

    def _nll_tdt(self, enc, enc_len, T, t_len):
        if self.tdt_loss == "torch":
            return self._nll_tdt_torch(enc, enc_len, T, t_len)
        return self._nll_tdt_nemo(enc, enc_len, T, t_len)

    def _nll_tdt_torch(self, enc, enc_len, T, t_len):
        from .tdt_loss import cell_logprobs, lattice_nll
        dec, _, _ = self.model.decoder(targets=T, target_length=t_len)    # B, H, U+1
        f, g = enc.transpose(1, 2), dec.transpose(1, 2)                  # B, T, H / B, U+1, H
        D = len(self.durations) if self.durations else 0
        parts = [cell_logprobs(self.model.joint.joint(f[:, a:a + self.chunk], g), T, self.blank, D)
                 for a in range(0, f.shape[1], self.chunk)]
        b, l, d = (self.torch.cat(x, dim=1) for x in zip(*parts))
        return lattice_nll(b, l, d, enc_len, t_len, self.durations).tolist()

    def _nll_tdt_nemo(self, enc, enc_len, T, t_len):
        dec, _, _ = self.model.decoder(targets=T, target_length=t_len)
        joint = self.model.joint.joint(enc.transpose(1, 2), dec.transpose(1, 2))   # B,T,U+1,V(+dur)
        with self._loss_reduction_none() as loss:
            nll = loss(log_probs=joint.float(), targets=T, input_lengths=enc_len, target_lengths=t_len)
        nll = nll.reshape(-1)
        if nll.numel() != T.shape[0]:   # loss ignored reduction=None -> one utterance at a time
            return [self._nll_tdt_nemo(enc[i:i + 1, :, :int(enc_len[i])], enc_len[i:i + 1],
                                  T[i:i + 1, :int(t_len[i])], t_len[i:i + 1])[0]
                    for i in range(T.shape[0])]
        return nll.tolist()

    def _nll_ctc(self, enc, enc_len, T, t_len):
        F = self.torch.nn.functional
        logp = self.model.ctc_decoder(encoder_output=enc).float().log_softmax(-1)   # B,T,C
        blank = logp.shape[-1] - 1
        return F.ctc_loss(logp.transpose(0, 1), T, enc_len, t_len, blank=blank,
                          reduction="none", zero_infinity=True).tolist()

    # ------------------------------------------------------------------ batching
    # Every candidate of a number is scored on the SAME audio clip (the sentence, or the number's
    # word window), so the expensive part — the Conformer encoder — runs once per unique clip,
    # not once per (clip, candidate). Clips are deduplicated by object identity (the pipeline
    # reuses one array per sentence / word window), encoded ``batch_size`` clips at a time
    # longest-first (as SegmentASR does, so a batch pads to a similar length), and then the
    # (clip, candidate-text) pairs of those clips go through the decoder + joint + loss
    # ``batch_size`` pairs at a time, reusing the encoder output (index_select per pair).
    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _pad_audio(self, audios):
        torch = self.torch
        ts = [torch.as_tensor(a, dtype=torch.float32) for a in audios]
        a_len = torch.tensor([len(a) for a in ts], dtype=torch.long)
        A = torch.zeros(len(ts), int(a_len.max()))
        for i, a in enumerate(ts):
            A[i, :len(a)] = a
        return A.to(self.device, self.dtype), a_len.to(self.device)

    def _pad_tokens(self, texts):
        torch = self.torch
        ids = [self.tokenizer.text_to_ids(t) or [0] for t in texts]
        t_len = torch.tensor([len(x) for x in ids], dtype=torch.long)
        T = torch.zeros(len(ids), int(t_len.max()), dtype=torch.long)
        for i, x in enumerate(ids):
            T[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        return T.to(self.device), t_len.to(self.device)

    def _frame_windows(self, enc, enc_len, fps, rows, wins):
        """Per pair: the encoder frames of its (start_s, end_s) window inside its clip's encoding
        (None = all frames), stacked and padded -> (B, D, T_max), lengths."""
        torch = self.torch
        parts, lens = [], []
        for r, w in zip(rows, wins):
            n = int(enc_len[r])
            if w is None:
                a, b = 0, n
            else:
                a = min(max(0, int(w[0] * fps[r])), n - 1)
                b = min(n, max(a + 1, int(-(-w[1] * fps[r] // 1))))      # ceil
            parts.append(enc[r, :, a:b])
            lens.append(b - a)
        T = max(lens)
        out = enc.new_zeros(len(parts), enc.shape[1], T)
        for i, p in enumerate(parts):
            out[i, :, :p.shape[1]] = p
        return out, torch.tensor(lens, dtype=enc_len.dtype, device=enc.device)

    def _encoder_batches(self, order, clips):
        """Duration-budget batches (as Lhotse/NeMo dynamic batching): clips are longest-first,
        so a batch's padded size is n_clips x its first clip; add clips while that stays within
        ``encoder_batch_s`` (at least 1, at most 256 clips)."""
        budget = float(getattr(self.cfg, "encoder_batch_s", 240.0)) * self.sample_rate
        i = 0
        while i < len(order):
            longest = max(1, len(clips[order[i]]))
            n = int(max(1, min(256, budget // longest)))
            yield order[i:i + n]
            i += n

    # ------------------------------------------------------------------ API
    def score_heads(self, pairs: Sequence[Tuple[object, str]], heads=("tdt", "ctc")):
        """{head: [nll per pair]} — one encoder pass per unique clip, shared by its candidates
        and by both heads."""
        import time
        heads = [h for h in heads if h in self.heads]
        if not heads:
            raise ValueError(f"none of {heads} available in this model (has {self.heads})")
        clip_of, clips, pair_clip = {}, [], []
        for a, *_ in pairs:
            k = id(a)
            if k not in clip_of:
                clip_of[k] = len(clips)
                clips.append(a)
            pair_clip.append(clip_of[k])
        by_clip = [[] for _ in clips]
        for j, c in enumerate(pair_clip):
            by_clip[c].append(j)
        order = sorted(range(len(clips)), key=lambda c: len(clips[c]), reverse=True)
        res = {h: [None] * len(pairs) for h in heads}
        st = self.stats
        with self.torch.inference_mode():
            for cb in self._encoder_batches(order, clips):
                t0 = time.perf_counter()
                A, a_len = self._pad_audio([clips[c] for c in cb])
                enc, enc_len = self._encode(A, a_len)                   # B_clips, D, T
                fps = (enc_len.float() / (a_len.float() / self.sample_rate)).tolist()   # frames / s
                self._sync()
                t1 = time.perf_counter()
                st["encoder_batches"] += 1
                st["clips_encoded"] += len(cb)
                st["audio_seconds_encoded"] += float(a_len.sum()) / self.sample_rate
                st["encode_s"] += t1 - t0
                row = {c: r for r, c in enumerate(cb)}
                pj = [j for c in cb for j in by_clip[c]]
                for k in range(0, len(pj), self.batch_size):
                    sel = pj[k:k + self.batch_size]
                    rows = self.torch.tensor([row[pair_clip[j]] for j in sel], device=self.device)
                    if any(len(pairs[j]) > 2 and pairs[j][2] is not None for j in sel):
                        e, e_len = self._frame_windows(enc, enc_len, fps, [row[pair_clip[j]] for j in sel],
                                                       [pairs[j][2] if len(pairs[j]) > 2 else None
                                                        for j in sel])
                    else:
                        e_len = enc_len.index_select(0, rows)
                        e = enc.index_select(0, rows)[:, :, :int(e_len.max())]
                    T, t_len = self._pad_tokens([pairs[j][1] for j in sel])
                    for h in heads:
                        vals = (self._nll_tdt if h == "tdt" else self._nll_ctc)(e, e_len, T, t_len)
                        for j, v in zip(sel, vals):
                            res[h][j] = float(v)
                    st["decoder_batches"] += 1
                    st["pairs_scored"] += len(sel)
                self._sync()
                st["decode_s"] += time.perf_counter() - t1
        return res

    def score(self, pairs, head: str = "tdt") -> List[float]:
        return self.score_heads(pairs, (head,))[head]

    def transcribe(self, audios) -> List[str]:
        """Greedy transcripts of numpy audio arrays (the model's own decoding). NeMo's
        model.transcribe batches internally (``batch_size`` clips, longest-first)."""
        import time
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]), reverse=True)
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            hyps = self.model.transcribe(audio=[audios[i] for i in order], batch_size=self.batch_size,
                                         num_workers=0, verbose=False)
        self._sync()
        self.stats["transcribe_s"] += time.perf_counter() - t0
        self.stats["clips_transcribed"] += len(audios)
        self.stats["audio_seconds_transcribed"] += sum(len(a) for a in audios) / self.sample_rate
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        out = [None] * len(audios)
        for pos, i in enumerate(order):
            out[i] = getattr(hyps[pos], "text", hyps[pos]) or ""
        return out
