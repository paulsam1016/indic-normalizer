"""NemoScorer batching (shared encoder per clip, duration-budget encoder batches, pair batches)
must give exactly the per-pair scores. Uses a tiny fake model; runs where torch is installed."""
import collections
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from indic_normalizer.digits.scoring.nemo_scorer import NemoScorer  # noqa: E402

V, D, H = 20, 5, 8


class _Tok:
    vocab_size = V

    def text_to_ids(self, t):
        return [ord(c) % V for c in t if not c.isspace()]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(160, H)          # frame-wise -> padding-independent encoder
        self.emb = torch.nn.Embedding(V + 1, H)
        self.jl = torch.nn.Linear(H, V + 1 + D)
        self.cl = torch.nn.Linear(H, V + 1)

    def encode(self, input_signal, input_signal_length):
        T = input_signal.shape[1] // 160
        x = input_signal[:, :T * 160].reshape(input_signal.shape[0], T, 160)
        return self.lin(x).transpose(1, 2), input_signal_length // 160

    def dec(self, targets, target_length):
        y = torch.cat([torch.full((targets.shape[0], 1), V), targets], 1)
        return self.emb(y).cumsum(1).transpose(1, 2), target_length, None

    def joint(self, f, g):
        return self.jl(torch.tanh(f[:, :, None, :] + g[:, None, :, :]))

    def ctc(self, encoder_output):
        return self.cl(encoder_output.transpose(1, 2))


def _scorer(batch_size, encoder_batch_s):
    torch.manual_seed(0)
    m = _Model().eval()
    sc = object.__new__(NemoScorer)
    sc.torch, sc.device, sc.dtype = torch, torch.device("cpu"), torch.float32
    sc.cfg = types.SimpleNamespace(encoder_batch_s=encoder_batch_s)
    sc.model = types.SimpleNamespace(forward=m.encode, decoder=m.dec,
                                     joint=types.SimpleNamespace(joint=m.joint), ctc_decoder=m.ctc)
    sc.batch_size, sc.sample_rate, sc.heads = batch_size, 16000, ["tdt", "ctc"]
    sc.tokenizer, sc.stats = _Tok(), collections.Counter()
    sc.tdt_loss, sc.chunk, sc.durations, sc.blank = "torch", 4, [0, 1, 2, 3, 4], V
    return sc


def test_batched_scores_equal_per_pair_scores():
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(int(16000 * s)).astype("float32") for s in (0.5, 0.9, 0.3, 0.7, 0.6)]
    texts = ["abc", "abd", "xy", "hello", "hi", "abcd", "q", "zz", "mmm", "n o p", "rs", "tuv"]
    pairs = [(clips[i % 5], t) for i, t in enumerate(texts)]
    sc = _scorer(batch_size=3, encoder_batch_s=2.0)
    batched = sc.score_heads(pairs)
    assert sc.stats["clips_encoded"] == 5                 # each clip encoded once
    assert sc.stats["pairs_scored"] == len(pairs)
    assert sc.stats["encoder_batches"] < 5                # several clips per encoder batch
    for h in ("tdt", "ctc"):
        single = [_scorer(1, 1.0).score_heads([p], [h])[h][0] for p in pairs]
        assert max(abs(a - b) for a, b in zip(batched[h], single)) < 1e-4


def test_window_scoring_equals_scoring_the_cut_audio():
    """With a frame-wise encoder, scoring frames [a, b) of a clip's encoding must equal scoring the
    audio cut to those frames (160 samples = 1 frame = 0.01 s here)."""
    rng = np.random.default_rng(1)
    clip = rng.standard_normal(16000).astype("float32")                 # 100 frames
    sc = _scorer(batch_size=4, encoder_batch_s=5.0)
    wins = [(0.20, 0.50), (0.0, 0.31), (0.55, 1.0)]
    texts = ["abc", "hello", "xy"]
    got = sc.score_heads([(clip, t, w) for t, w in zip(texts, wins)])
    assert sc.stats["clips_encoded"] == 1
    for h in ("tdt", "ctc"):
        for (t, w), g in zip(zip(texts, wins), got[h]):
            a, b = int(w[0] * 100), int(-(-w[1] * 100 // 1))
            ref = _scorer(1, 5.0).score_heads([(clip[a * 160:b * 160].copy(), t)], [h])[h][0]
            assert abs(g - ref) < 1e-4
