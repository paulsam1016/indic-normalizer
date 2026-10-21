"""Pure-PyTorch transducer (RNN-T / TDT) negative log-likelihood for scoring.

NeMo's RNN-T/TDT losses run as numba CUDA kernels, which need libNVVM from a full CUDA
toolkit (``NvvmSupportError: libNVVM cannot be found`` on a runtime-only box). Scoring only
needs the forward variable, so this computes it directly in torch — no numba, any device —
and gives the model's actual likelihood (no TDT training-time ``sigma`` undernormalization).

Lattice (per utterance, T encoder frames, U target tokens), states (t, u), t < T:
    label  y_{u+1} with duration d (d may be 0): (t, u) -> (t + d, u + 1)
    blank            with duration d >= 1      : (t, u) -> (t + d, u)
    end   blank with duration d from (T - d, U)  [exactly reaching T]
logp(label) and logp(blank) come from log_softmax over the vocab+blank logits, logp(d) from
log_softmax over the duration logits (TDT). A plain RNN-T is the special case durations
label = {0}, blank = {1}, logp(d) = 0. The forward pass runs over anti-diagonals t + u = k
(every dependency has a smaller k), vectorised across the batch and the diagonal.
Matches NeMo's TDTLossPytorch reference recursion (with sigma = 0).

Memory: only per-cell values (blank, next label, durations) are kept, never the full
(T, U+1, V) log-softmax; ``cell_logprobs`` takes the joint one chunk of frames at a time.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

NEG = -1e30


def cell_logprobs(joint: torch.Tensor, targets: torch.Tensor, blank: int,
                  n_durations: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """From joint logits (B, t, U+1, V+1[+D]) of a chunk of frames ->
    (blank_lp, label_lp, dur_lp) of shapes (B, t, U+1), (B, t, U+1), (B, t, U+1, max(D, 2)),
    float32. label_lp[:, :, u] = logp(targets[u]) (u < U; -inf at u = U)."""
    j = joint.float()
    B, n, U1, C = j.shape
    D, U = n_durations, U1 - 1
    tok = j[..., :C - D] if D else j
    lse = tok.logsumexp(-1)
    blank_lp = tok[..., blank] - lse
    label_lp = torch.full((B, n, U1), NEG, dtype=torch.float32, device=j.device)
    if U > 0:
        g = tok[:, :, :U, :].gather(-1, targets[:, None, :U, None].expand(B, n, U, 1)).squeeze(-1)
        label_lp[:, :, :U] = g - lse[:, :, :U]
    if D:
        dur = j[..., C - D:].log_softmax(-1)
    else:
        dur = torch.zeros(B, n, U1, 2, dtype=torch.float32, device=j.device)
    return blank_lp, label_lp, dur


def lattice_nll(blank_lp: torch.Tensor, label_lp: torch.Tensor, dur: torch.Tensor,
                frame_lens: torch.Tensor, target_lens: torch.Tensor,
                durations: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Forward algorithm over per-cell log-probs (B, T, U+1) -> -log p per utterance."""
    if durations:
        lab_d = list(enumerate(durations))
        blk_d = [(n, d) for n, d in enumerate(durations) if d > 0]
    else:
        lab_d, blk_d = [(0, 0)], [(1, 1)]
    B, T, U1 = blank_lp.shape
    U = U1 - 1
    dev = blank_lp.device
    # All incoming transitions of a cell, as rows: (duration d, duration-logit index n, kind,
    # du) with kind 0 = blank (from (t-d, u)), 1 = label (from (t-d, u-1)). One gather + one
    # logsumexp per anti-diagonal for ALL transitions (few kernel launches per step).
    rows = [(d, n, 0, 0) for n, d in blk_d] + [(d, n, 1, 1) for n, d in lab_d]
    R_d = torch.tensor([r[0] for r in rows], device=dev)[:, None]
    R_n = torch.tensor([r[1] for r in rows], device=dev)[:, None]
    R_k = torch.tensor([r[2] for r in rows], device=dev)[:, None]
    R_du = torch.tensor([r[3] for r in rows], device=dev)[:, None]
    emit = torch.stack([blank_lp, label_lp], dim=-1)               # B, T, U1, 2
    alpha = torch.full((B, T, U1), NEG, dtype=torch.float32, device=dev)
    alpha[:, 0, 0] = 0.0
    for k in range(1, T + U):
        t_lo, t_hi = max(0, k - U), min(T - 1, k)
        if t_lo > t_hi:
            continue
        ts = torch.arange(t_lo, t_hi + 1, device=dev)[None, :]   # 1, L
        us = k - ts
        tp, up = ts - R_d, us - R_du                              # R, L
        ok = (tp >= 0) & (up >= 0)
        tpc, upc = tp.clamp(min=0), up.clamp(min=0)
        kk, nn = R_k.expand_as(tpc), R_n.expand_as(tpc)
        v = alpha[:, tpc, upc] + emit[:, tpc, upc, kk] + dur[:, tpc, upc, nn]   # B, R, L
        v = torch.where(ok, v, torch.full_like(v, NEG))
        alpha[:, ts[0], us[0]] = torch.logsumexp(v, dim=1)

    bi = torch.arange(B, device=dev)
    Ub = target_lens.long().to(dev)
    out = torch.full((B,), NEG, dtype=torch.float32, device=dev)
    for n, d in blk_d:                                            # end: blank from (T_b - d, U_b)
        tp = frame_lens.long().to(dev) - d
        ok = tp >= 0
        tpc = tp.clamp(min=0)
        v = alpha[bi, tpc, Ub] + blank_lp[bi, tpc, Ub] + dur[bi, tpc, Ub, n]
        out = torch.logaddexp(out, torch.where(ok, v, torch.full_like(v, NEG)))
    return -out


def transducer_nll(joint: torch.Tensor, targets: torch.Tensor, frame_lens: torch.Tensor,
                   target_lens: torch.Tensor, blank: int,
                   durations: Optional[Sequence[int]] = None, chunk_frames: int = 32) -> torch.Tensor:
    """-log p(targets | audio) per utterance from a full joint tensor.

    joint        (B, T, U+1, V+1[+D]) raw joint logits; the first V+1 are vocab + blank,
                 the last D = len(durations) are TDT duration logits (absent for plain RNN-T)
    targets      (B, U) token ids (padded); frame_lens (B,), target_lens (B,)
    blank        blank id within the V+1 token logits (NeMo: V, i.e. the last)
    durations    TDT durations, e.g. [0, 1, 2, 3, 4]; None = plain RNN-T
    """
    D = len(durations) if durations else 0
    parts = [cell_logprobs(joint[:, a:a + chunk_frames], targets, blank, D)
             for a in range(0, joint.shape[1], chunk_frames)]
    b, l, d = (torch.cat(x, dim=1) for x in zip(*parts))
    return lattice_nll(b, l, d, frame_lens, target_lens, durations)
