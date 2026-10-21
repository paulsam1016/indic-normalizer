"""Pure-torch transducer NLL vs brute-force path enumeration (runs where torch is installed)."""
from functools import lru_cache

import pytest

torch = pytest.importorskip("torch")

from indic_normalizer.digits.scoring.tdt_loss import transducer_nll  # noqa: E402


def _brute(joint, y, T, blank, durs):
    U = len(y)
    if durs:
        D = len(durs)
        tok, dur = joint[..., :-D].log_softmax(-1), joint[..., -D:].log_softmax(-1)
        lab, blk = list(enumerate(durs)), [(n, d) for n, d in enumerate(durs) if d > 0]
    else:
        tok, dur = joint.log_softmax(-1), torch.zeros(*joint.shape[:2], 2)
        lab, blk = [(0, 0)], [(1, 1)]

    @lru_cache(None)
    def f(t, u):
        tot = []
        for n, d in blk:
            if t + d == T and u == U:
                tot.append(tok[t, u, blank] + dur[t, u, n])
            elif t + d < T:
                tot.append(tok[t, u, blank] + dur[t, u, n] + f(t + d, u))
        if u < U:
            for n, d in lab:
                if t + d < T:
                    tot.append(tok[t, u, y[u]] + dur[t, u, n] + f(t + d, u + 1))
        return torch.logsumexp(torch.stack(tot), 0) if tot else torch.tensor(-1e30)
    return -f(0, 0)


@pytest.mark.parametrize("durs", [[0, 1, 2, 3, 4], [0, 1, 2], None])
def test_matches_brute_force(durs):
    torch.manual_seed(0)
    V, B, T, U = 6, 3, 7, 4
    D = len(durs) if durs else 0
    joint = torch.randn(B, T, U + 1, V + 1 + D) * 2
    tg = torch.randint(0, V, (B, U))
    fl, tl = torch.tensor([7, 5, 6]), torch.tensor([4, 2, 3])
    got = transducer_nll(joint, tg, fl, tl, V, durs, chunk_frames=3)
    exp = torch.stack([_brute(joint[b, :fl[b], :tl[b] + 1], tuple(tg[b, :tl[b]].tolist()), int(fl[b]), V, durs)
                       for b in range(B)])
    assert torch.allclose(got, exp, atol=1e-3)


def test_bfloat16_joint_close_to_float32():
    torch.manual_seed(1)
    j = torch.randn(2, 9, 4, 12)
    tg = torch.randint(0, 6, (2, 3))
    fl, tl = torch.tensor([9, 8]), torch.tensor([3, 2])
    a = transducer_nll(j, tg, fl, tl, 6, [0, 1, 2, 3, 4])
    b = transducer_nll(j.bfloat16(), tg, fl, tl, 6, [0, 1, 2, 3, 4])
    assert torch.allclose(a, b, atol=0.2)
