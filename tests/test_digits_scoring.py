import json
import os
import wave

import pytest

np = pytest.importorskip("numpy")

from indic_normalizer.digits.scoring import load_config  # noqa: E402
from indic_normalizer.digits.scoring.audio import load_wav, slice_audio  # noqa: E402
from indic_normalizer.digits.scoring.compare import compare_dir  # noqa: E402
from indic_normalizer.digits.scoring.pipeline import (  # noqa: E402
    output_path, plan_document, verbalize_dir)


class FakeScorer:
    """Deterministic stand-in for NemoScorer: likes English for tdt, Malayalam for ctc,
    and records the audio length it was given."""
    heads = ["tdt", "ctc"]

    def transcribe(self, audios):
        return ["ഫീസ് ഇരുപത്തഞ്ച് ആണ്" if len(a) > 24400 else "ശരി ഫോർ പേര്" for a in audios]

    def __init__(self):
        self.lengths = []

    def n_tokens(self, text):
        return max(1, len(text.split()))

    def score(self, pairs, head="tdt"):
        out = []
        for audio, text, *_ in pairs:
            self.lengths.append(len(audio))
            en = any(w in text for w in ("ട്വന്റി", "ഫൈവ്", "തൗസന്റ്", "ഫോർ"))
            base = len(text) / 100.0
            split = any(w in text for w in ("ടു ഫൈവ്", "രണ്ട് അഞ്ച്"))        # 25 digit by digit: never said
            out.append(base + (0.0 if (en if head == "tdt" else not en) else 3.0) + (5.0 if split else 0.0))
        return out


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)         # default report dirs (outputs/...) go to the temp dir


def _wav(path, seconds, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.zeros(int(seconds * sr)) + 100).astype("<i2").tobytes())


DOC = {
    "transcription": "ഫീസ് 25 ആണ്. ശരി, 4 പേര്.",
    "words": [
        {"word": "ഫീസ്", "start_time": 0.0, "end_time": 0.4},
        {"word": "25", "start_time": 0.4, "end_time": 1.0},
        {"word": "ആണ്.", "start_time": 1.0, "end_time": 1.3},
        {"word": "ശരി,", "start_time": 2.0, "end_time": 2.3},
        {"word": "4", "start_time": 2.3, "end_time": 2.6},
        {"word": "പേര്.", "start_time": 2.6, "end_time": 3.0},
    ],
    "sentences": [
        {"text": "ഫീസ് 25 ആണ്.", "start_time": 0.0, "end_time": 1.3, "speaker": "1"},
        {"text": "ശരി, 4 പേര്.", "start_time": 2.0, "end_time": 3.0, "speaker": "2"},
        {"text": "ഓക്കേ.", "start_time": 3.2, "end_time": 3.5, "speaker": "1"},
    ],
}


@pytest.fixture()
def dataset(tmp_path):
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    (data / "rec1.json").write_text(json.dumps(DOC, ensure_ascii=False), encoding="utf-8")
    _wav(audio / "rec1.wav", 4.0)
    return data, audio, tmp_path


def test_config_file_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model_path: ~/m.nemo\nbatch_size: 4\nhead: ctc\ncuda: 0\ncompute_dtype: bfloat16\n",
                 encoding="utf-8")
    cfg = load_config(str(p), batch_size=8)
    assert cfg.batch_size == 8 and cfg.head == "ctc" and cfg.model_path.endswith("m.nemo")
    assert cfg.cuda == 0 and cfg.compute_dtype == "bfloat16" and cfg.hyp_pad_s == 0.5
    assert load_config(str(p), hyp_pad_s=1.5).hyp_pad_s == 1.5
    assert "~" not in cfg.model_path
    p.write_text("bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))
    with pytest.raises(ValueError):
        load_config(None, head="rnnt")


def test_audio_load_and_slice(dataset):
    _, audio, _ = dataset
    x = load_wav(str(audio / "rec1.wav"), 16000)
    assert x.dtype == np.float32 and len(x) == 64000
    assert len(slice_audio(x, 1.0, 2.0, 16000, pad_s=0.1)) == int(1.2 * 16000)


def test_plan_segment_and_word_jobs(dataset):
    data, audio, _ = dataset
    cfg = load_config(None, max_candidates=3, pad_s=0.15)
    x = load_wav(str(audio / "rec1.wav"), 16000)
    plans = plan_document(DOC, x, cfg, units=("segment", "word"))
    nums = [n for p in plans for n in p["numbers"]]
    assert [n.span.core for n in nums] == ["25", "4"]
    n = nums[0]
    # segment: whole sentence with the candidate substituted
    assert [t for _, t in n.jobs["segment"]] == ["ഫീസ് ഇരുപത്തഞ്ച് ആണ്", "ഫീസ് ഇരുപത്തിയഞ്ച് ആണ്",
                                                 "ഫീസ് ട്വന്റി ഫൈവ് ആണ്"]
    # word: the number word +- 1 neighbour, cut by word timestamps
    assert n.unit_used["word"] == "word" and n.word_idx == [1]
    assert n.jobs["word"][2][1] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ്"
    assert len(n.jobs["word"][0][0]) == int((1.3 + 0.15) * 16000)


def test_word_unit_falls_back_without_matching_words(dataset):
    data, audio, _ = dataset
    doc = dict(DOC, words=[])
    plans = plan_document(doc, load_wav(str(audio / "rec1.wav")), load_config(None), units=("word",))
    n = plans[0]["numbers"][0]
    assert n.unit_used["word"] == "segment(fallback)"


def test_verbalize_dir_writes_same_schema(dataset):
    data, audio, tmp = dataset
    cfg = load_config(None, head="tdt", unit="segment", margin_threshold=0.0, out_dir=str(tmp / "out"))
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), cfg, log=lambda *_: None)
    assert tot["files"] == 1 and tot["numbers"] == 2 and tot["verbalized"] == 1
    dst = output_path(str(data / "rec1.json"), cfg, str(data))
    assert dst == os.path.join(str(tmp / "out"), "data", "verbalized", "rec1.json")
    out = json.load(open(dst, encoding="utf-8"))
    s0, s1, s2 = out["sentences"]
    assert s0["text"] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ്"          # tdt fake prefers English
    assert s1["text"] == "ശരി ഫോർ പേര്"
    assert s2["text"] == "ഓക്കേ"
    assert out["transcription"] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ് ശരി ഫോർ പേര് ഓക്കേ"
    rd = tmp / "out" / "data"
    for f in ("verbalized.tsv", "verify.tsv", "failed.tsv", "summary.json", "report.html"):
        assert (rd / f).exists()
    from indic_normalizer.digits.scoring.details import read_tsv
    det = read_tsv(str(rd / "verbalized.tsv"))
    assert [r["number"] for r in det] == ["25", "4"] and det[0]["file"] == "rec1" and det[0]["status"] == "auto"
    assert det[0]["model_pick"] == det[0]["final_pick"] == "ട്വന്റി ഫൈവ്"
    assert len(det[0]["candidates"].split(" | ")) == len(det[0]["scores"].split(" | ")) >= 2
    assert open(rd / "verbalized.tsv", "rb").read(3) == b"\xef\xbb\xbf"           # BOM for Excel
    assert not list((rd / "verify").iterdir()) and not list((rd / "failed").iterdir())


def test_needs_review_when_margin_small(dataset):
    data, audio, tmp = dataset

    class Flat(FakeScorer):
        def score(self, pairs, head="tdt"):
            return [1.0 for _ in pairs]

    cfg = load_config(None, out_dir=str(tmp / "out"))
    tot = verbalize_dir(str(data), str(audio), Flat(), cfg, log=lambda *_: None)
    assert tot["needs_review"] == 2 and tot["verify"] == 1 and tot["verbalized"] == 0
    rd = tmp / "out" / "data"
    copied = json.load(open(rd / "verify" / "rec1.json", encoding="utf-8"))
    assert [x["text"] for x in copied["sentences"]] == ["ഫീസ് 25 ആണ്", "ശരി 4 പേര്", "ഓക്കേ"]  # uncertain: kept
    from indic_normalizer.digits.scoring.details import read_tsv
    det = read_tsv(str(rd / "verify.tsv"))
    assert [r["number"] for r in det] == ["25", "4"] and all(r["status"] == "pending" for r in det)
    assert "need review" in det[0]["file_reasons"] and det[0]["needs_review"] == "true"
    assert det[0]["model_pick"] == "ഇരുപത്തഞ്ച്" and det[0]["final_pick"] == ""      # tie -> default
    # a later confident run moves the file from verify/ to verbalized/
    verbalize_dir(str(data), str(audio), FakeScorer(), load_config(None, out_dir=str(tmp / "out"),
                  margin_threshold=0.0), log=lambda *_: None)
    assert (rd / "verbalized" / "rec1.json").exists() and not (rd / "verify" / "rec1.json").exists()
    assert read_tsv(str(rd / "verify.tsv")) == [] and len(read_tsv(str(rd / "verbalized.tsv"))) == 2


def test_missing_audio_is_skipped(dataset):
    data, audio, tmp = dataset
    os.remove(audio / "rec1.wav")
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), load_config(None, out_dir=str(tmp / "o")),
                        log=lambda *_: None)
    assert tot["failed"] == 1 and tot["verbalized"] == 0
    rd = tmp / "o" / "data"
    assert (rd / "failed" / "rec1.json").exists()
    from indic_normalizer.digits.scoring.details import read_tsv
    f = read_tsv(str(rd / "failed.tsv"))[0]
    assert f["file"] == "rec1" and f["stage"] == "find_audio" and "rec1" in f["error"] and f["status"] == "pending"


def test_compare_report(dataset):
    data, audio, tmp = dataset
    hyps = tmp / "hyps"
    hyps.mkdir()
    (hyps / "rec1.jsonl").write_text(
        json.dumps({"start_time": 0.0, "end_time": 1.3, "text": "ഫീസ് ട്വന്റി ഫൈവ് ആണ്"}, ensure_ascii=False)
        + "\n" + json.dumps({"start_time": 2.0, "end_time": 3.0, "text": "ശരി നാല് പേര്"}, ensure_ascii=False)
        + "\n", encoding="utf-8")
    seg = tmp / "segs"
    seg.mkdir()
    (seg / "rec1.jsonl").write_text(
        json.dumps({"start_time": 0.0, "end_time": 1.3, "text": "फीस पच्चीस"}, ensure_ascii=False) + "\n"
        + json.dumps({"start_time": 2.0, "end_time": 3.0, "text": "ശരി ഫോർ പേര്"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    rep = compare_dir(str(data), str(audio), FakeScorer(), load_config(None), out_root=str(tmp / "o"),
                      hyps_dirs=[str(hyps), str(seg)], log=lambda *_: None)
    assert rep["numbers"] == 2
    assert set(rep["methods"]) == {"tdt/segment", "ctc/segment", "tdt/word", "ctc/word",
                                   "tdt/window", "ctc/window"}
    assert rep["methods"]["tdt/segment"]["lang"] == {"en": 2}
    assert rep["methods"]["ctc/segment"]["lang"] == {"ml": 2}
    assert rep["agreement"]["tdt/segment vs ctc/segment"]["same"] == 0
    assert rep["references"]["hyps"]["oracle"] == 2
    assert rep["references"]["segs"]["oracle"] == 1                     # Devanagari row can't match
    assert rep["references"]["segs"]["hyp_malayalam_pct"] == 50.0
    assert rep["methods"]["tdt/segment"]["match"]["hyps"]["match"] == 1  # ട്വന്റി ഫൈവ് yes, ഫോർ no
    assert rep["methods"]["ctc/segment"]["match"]["hyps"]["match"] == 1  # നാല് yes
    assert rep["methods"]["tdt/segment"]["match"]["segs"] == {"match": 1, "pct": 50.0, "given_oracle_pct": 100.0}

    rep2 = compare_dir(str(data), str(audio), FakeScorer(), load_config(None), out_root=str(tmp / "o2"),
                       transcribe=True, log=lambda *_: None)
    assert rep2["references"]["model"]["oracle"] == 2 and rep2["references"]["model"]["default_match"] == 1
    assert rep2["methods"]["tdt/segment"]["match"]["model"]["match"] == 1     # ഫോർ
    assert rep2["methods"]["ctc/segment"]["match"]["model"]["match"] == 1     # ഇരുപത്തഞ്ച്
    md = open(os.path.join(rep2["out_dir"], "report.html"), encoding="utf-8").read()
    assert "Matches <code>model</code>" in md and "Columns of numbers.tsv" in md and ">Timing<" in md
    assert 'id="numbers-data"' in md and 'class="sortable"' in md
    import re as _re
    data = json.loads(_re.search(r'id="numbers-data">(.*?)</script>', md, _re.S).group(1))
    assert len(data["rows"]) == 2 and "tdt/segment" in data["methods"] and "model" in data["refs"]
    assert rep2["timing"]["numbers"] == 2 and "transcribe" in rep2["timing"]["stages_s"]
    assert rep2["timing"]["audio_hours"] > 0 and rep2["timing"]["rtf_total"] is not None

    import csv
    rows = list(csv.DictReader(open(os.path.join(rep["out_dir"], "numbers.tsv"), encoding="utf-8"), delimiter="\t"))
    assert rows[0]["sent_start"] == "0.0" and rows[0]["sent_end"] == "1.3"
    assert rows[0]["word_start"] == "0.4" and rows[0]["word_end"] == "1.0"
    assert rows[0]["candidates"].split(" | ")[0] == rows[0]["default"]
    assert rows[0]["hyp:hyps"] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ്"
    assert rows[0]["tdt/segment:match:hyps"] == "True" and rows[0]["oracle:segs"] == "False"
    for f in ("report.html", "report.json", "numbers.tsv"):
        assert os.path.exists(os.path.join(rep["out_dir"], f))


def test_cli_text(capsys):
    from indic_normalizer.digits.cli import main
    main(["text", "ഫീസ് 1,75,000 ആണ്.", "--candidates"])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "ഫീസ് ഒരു ലക്ഷത്തി എഴുപത്തയ്യായിരം ആണ്"
    assert "[integer]" in out          # train mode already dropped the grouping commas


def test_hyp_pad_changes_window(dataset):
    data, audio, tmp = dataset
    hyps = tmp / "hp"
    hyps.mkdir()
    # hypothesis for sentence 0 sits 0.8 s after the sentence ends (1.3 s)
    (hyps / "rec1.jsonl").write_text(json.dumps({"start_time": 2.1, "end_time": 2.4, "text": "ട്വന്റി ഫൈവ്"},
                                                ensure_ascii=False) + "\n", encoding="utf-8")
    narrow = compare_dir(str(data), str(audio), FakeScorer(), load_config(None, hyp_pad_s=0.5),
                         out_root=str(tmp / "n"), hyps_dirs=[str(hyps)], log=lambda *_: None)
    wide = compare_dir(str(data), str(audio), FakeScorer(), load_config(None, hyp_pad_s=1.0),
                       out_root=str(tmp / "w"), hyps_dirs=[str(hyps)], log=lambda *_: None)
    assert narrow["hyp_pad_s"] == 0.5 and wide["hyp_pad_s"] == 1.0
    assert wide["references"]["hp"]["oracle"] > narrow["references"]["hp"]["oracle"]


def test_review_ignores_spelling_style_ties(dataset):
    data, audio, tmp = dataset

    class StyleTie(FakeScorer):
        # contracted and spaced (same family) nearly tie; English clearly worse
        def score(self, pairs, head="tdt"):
            return [5.0 if any(w in t for w in ("ട്വന്റി", "ഫോർ", "രണ്ട്", "ടു")) else (1.0 if " അഞ്ച്" in t else 1.1)
                    for _, t in pairs]

    cfg = load_config(None, out_dir=str(tmp / "o"))
    x = load_wav(str(audio / "rec1.wav"), 16000)
    plans = plan_document(DOC, x, cfg)
    from indic_normalizer.digits.scoring.pipeline import _run, decide
    _run(StyleTie(), plans, "segment", ("tdt",))
    n = plans[0]["numbers"][0]
    d = decide(n, "segment", "tdt", cfg)
    assert n.cands[d["model_best"]].form == "spaced" and round(d["margin"], 4) == 0.1
    assert round(d["review_margin"], 4) == 4.0 and d["needs_review"] is False
    # spelling too close (0.1 < style_margin 2): the default contracted spelling is used
    assert n.cands[d["best"]].form == "contracted" and d["style_fallback"] and round(d["style_gap"], 4) == 0.1
    d0 = decide(n, "segment", "tdt", load_config(None, style_margin=0.0))      # 0 = always the audio's
    assert n.cands[d0["best"]].form == "spaced" and not d0["style_fallback"]


def test_groups_pool_files_into_one_scoring_pass(tmp_path):
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    for k in range(3):
        (data / f"r{k}.json").write_text(json.dumps(DOC, ensure_ascii=False), encoding="utf-8")
        _wav(audio / f"r{k}.wav", 4.0)

    class Count(FakeScorer):
        def score(self, pairs, head="tdt"):
            self.calls.append(len(pairs))
            return super().score(pairs, head)

    outs = {}
    for g in (0, 1800):
        s = Count()
        s.calls = []
        od = tmp_path / f"o{g}"
        verbalize_dir(str(data), str(audio), s, load_config(None, group_audio_s=g, out_dir=str(od),
                                                           margin_threshold=0.0), log=lambda *_: None)
        outs[g] = (s.calls, [json.load(open(od / "data" / "verbalized" / f"r{k}.json", encoding="utf-8"))
                             ["transcription"] for k in range(3)])
    assert outs[0][0] == [10, 10, 10]        # per file (25: 6 readings incl. digit by digit, 4: 4)
    assert outs[1800][0] == [30]            # pooled across files
    assert outs[0][1] == outs[1800][1]      # same results


def test_window_jobs_share_the_sentence_clip(dataset):
    data, audio, _ = dataset
    cfg = load_config(None, max_candidates=3, pad_s=0.15)
    x = load_wav(str(audio / "rec1.wav"), 16000)
    plans = plan_document(DOC, x, cfg, units=("segment", "window"))
    n = plans[0]["numbers"][0]
    seg_clip = n.jobs["segment"][0][0]
    assert all(job[0] is seg_clip for job in n.jobs["window"])          # same clip -> encoded once
    assert [job[1] for job in n.jobs["window"]][2] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ്"   # window text
    w0, w1 = n.jobs["window"][0][2]                                     # words 0..2 = 0.0..1.3 s
    assert w0 == 0.0 and abs(w1 - 1.45) < 1e-9                          # clip starts at 0 (clamped)
    n2 = plans[1]["numbers"][0]                                         # sentence 2.0-3.0, clip from 1.85
    a, b = n2.jobs["window"][0][2]
    assert abs(a - (2.0 - 0.15 - 1.85)) < 1e-6 and abs(b - (3.0 + 0.15 - 1.85)) < 1e-6
    assert n2.unit_used["window"] == "window"


def test_sweep_one_at_a_time(dataset):
    from indic_normalizer.digits.scoring.sweep import parse_values, sweep
    data, audio, tmp = dataset
    grid = {"batch_size": [4, 16], "group_audio_s": [0.0, 300.0]}
    res = sweep(str(data), str(audio), FakeScorer(), load_config(None), out_root=str(tmp / "sw"),
                grid=grid, warmup_files=1, min_gain=1.0, log=lambda *_: None)   # timing noise (0.00 s runs) never wins
    labels = [r["label"] for r in res["runs"]]
    assert labels[0] == "baseline" and "batch_size_4" in labels and "group_audio_s_0" in labels
    assert all(r["changed_total"] == 0 for r in res["runs"])          # fake scorer: batching-invariant
    assert next(r for r in res["runs"] if r["label"] == "batch_size_16")["reused"]   # = baseline
    assert res["best_settings"]["batch_size"] == 16, [(r["label"], r.get("total_s"), r.get("error")) for r in res["runs"]]
    assert set(res["best_settings"]) == {"batch_size", "joint_chunk_frames", "encoder_batch_s", "group_audio_s",
                                         "compute_dtype"}
    for f in ("sweep.html", "sweep.json", "sweep.tsv", "best_config.yaml"):
        assert (tmp / "sw" / f).exists()
    assert parse_values("batch_size", "8, 16") == [8, 16]
    assert parse_values("encoder_batch_s", "60") == [60.0]


def test_sweep_stops_parameter_after_oom(dataset):
    from indic_normalizer.digits.scoring import sweep as sw
    data, audio, tmp = dataset
    real = sw.compare_dir

    def fake(data_dir, audio_dir, scorer, cfg, *a, **k):
        if cfg.batch_size >= 32:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return real(data_dir, audio_dir, scorer, cfg, *a, **k)

    sw.compare_dir = fake
    try:
        res = sw.sweep(str(data), str(audio), FakeScorer(), load_config(None), out_root=str(tmp / "sw"),
                       grid={"batch_size": [64, 8, 32, 16], "joint_chunk_frames": [16]},
                       warmup_files=0, log=lambda *_: None)
    finally:
        sw.compare_dir = real
    by = {r["label"]: r for r in res["runs"]}
    assert by["batch_size_32"]["oom"] and by["batch_size_64"].get("skipped")
    assert "joint_chunk_frames_16" in by                      # moved on to the next setting
    assert res["best_settings"]["batch_size"] in (8, 16)


def test_sweep_compute_dtype_reloads_model(dataset):
    from indic_normalizer.digits.scoring.sweep import sweep
    data, audio, tmp = dataset
    built = []

    def make(cfg):
        built.append(cfg.compute_dtype)
        return FakeScorer()

    res = sweep(str(data), str(audio), FakeScorer(), load_config(None), out_root=str(tmp / "sw"),
                grid={"compute_dtype": ["float32", "bfloat16"]}, warmup_files=0, log=lambda *_: None,
                make_scorer=make)
    by = {r["label"]: r for r in res["runs"]}
    assert built == ["bfloat16"]                     # float32 = baseline, reused; bf16 reloaded once
    assert by["compute_dtype_bfloat16"]["changed_total"] == 0
    assert by["compute_dtype_bfloat16"]["margin_drift_max"] == 0
    skipped = sweep(str(data), str(audio), FakeScorer(), load_config(None), out_root=str(tmp / "sw2"),
                    grid={"compute_dtype": ["bfloat16"]}, warmup_files=0, log=lambda *_: None)
    assert not any(r["param"] == "compute_dtype" for r in skipped["runs"])   # no make_scorer -> skipped


def test_sweep_config_file(tmp_path):
    from indic_normalizer.digits.scoring.sweep import load_sweep_config
    p = tmp_path / "s.yaml"
    p.write_text("grid:\n  batch_size: [16, 8]\n  joint_chunk_frames: none\n  compute_dtype: [float32, bfloat16]\n"
                 "greedy: true\nhyps_dirs: refs\n", encoding="utf-8")
    sc = load_sweep_config(str(p))
    assert sc["grid"] == {"batch_size": [16, 8], "compute_dtype": ["float32", "bfloat16"]}
    assert sc["greedy"] is True and sc["hyps_dirs"] == ["refs"] and sc["max_changed"] == 0
    sc = load_sweep_config(str(p), grid={"batch_size": "none", "encoder_batch_s": "30,60", "group_audio_s": None},
                           limit=5, hyps_dirs=[])
    assert sc["grid"] == {"encoder_batch_s": [30.0, 60.0], "compute_dtype": ["float32", "bfloat16"]}
    assert sc["limit"] == 5 and sc["hyps_dirs"] == ["refs"]
    assert set(load_sweep_config()["grid"]) == {"batch_size", "joint_chunk_frames", "encoder_batch_s",
                                                "group_audio_s"}          # float32 only by default
    for f in ("digits_sweep.yaml", "digits_sweep_dtype.yaml"):
        load_sweep_config(os.path.join(os.path.dirname(__file__), "..", "examples", f))
    p.write_text("grid:\n  batchsize: [8]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_sweep_config(str(p))


def test_sweep_config_base_and_examples():
    from indic_normalizer.digits.scoring.sweep import load_sweep_config
    sc = load_sweep_config(os.path.join(os.path.dirname(__file__), "..", "examples", "digits_sweep_bf16.yaml"))
    assert sc["base"]["compute_dtype"] == "bfloat16" and "compute_dtype" not in sc["grid"]
    assert load_config(None, **sc["base"]).compute_dtype == "bfloat16"


def test_cli_cfg_merges_base_and_flags(tmp_path):
    import argparse
    from indic_normalizer.digits.cli import _cfg
    a = argparse.Namespace(config=None, model_path=None, cuda=None, batch_size=None)
    assert _cfg(a, batch_size=24, compute_dtype="bfloat16").batch_size == 24      # base used
    a.batch_size = 8
    assert _cfg(a, batch_size=24).batch_size == 8                                 # flag wins
    assert _cfg(a).batch_size == 8


def test_outcomes_counts():
    from indic_normalizer.digits.scoring.compare import outcomes
    m, r = "tdt/segment", "ref"
    rows = [{f"hyp:{r}": "x", f"{m}:review": rv, f"{m}:match:{r}": mt, f"oracle:{r}": orc}
            for rv, mt, orc in [("False", "True", "True"), ("False", "True", "True"), ("True", "False", "True"),
                                ("False", "False", "True"), ("False", "False", "False")]]
    o = outcomes(rows, m, r)
    assert (o["success"], o["verify"], o["failed"], o["unchecked"], o["n"]) == (2, 1, 1, 1, 5)
    assert o["success_pct"] == 40.0 and o["success_of_checked_pct"] == 66.67


def test_html_report_md_subset():
    from indic_normalizer.digits.scoring.html_report import md_to_html, page
    body, toc = md_to_html("# T\n\n## Sec `a`\n\n| A | B `x | y` |\n|---|---|\n| 1 (5%) | **z** <b> |\n\n- item\n")
    assert "<th>B <code>x | y</code></th>" in body and '<td class="num">1 (5%)</td>' in body
    assert "<b>z</b> &lt;b&gt;" in body and toc == [("sec-a", "Sec a")] and "<li>item</li>" in body
    doc = page("t", "# T", {"columns": ["x"], "rows": [["</script><b>"]], "methods": [], "refs": [], "default_method": ""})
    assert doc.count("</script>") == 2          # data block + JS, the payload's </ is escaped


def test_verbalize_same_format(dataset):
    data, audio, tmp = dataset
    cfg = load_config(None, out_dir=str(tmp / "same"), margin_threshold=0.0)
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), cfg, log=lambda *_: None)
    src = json.load(open(next(data.glob("*.json")), encoding="utf-8"))
    out = json.load(open(tmp / "same" / "data" / "verbalized" / next(data.glob("*.json")).name, encoding="utf-8"))
    assert set(out) == set(src)                                       # same schema, same keys
    assert [set(s) for s in out["sentences"]] == [set(s) for s in src["sentences"]]
    assert all(set(w) == set(src["words"][0]) for w in out["words"])
    assert out["sentences"][0]["text"] == "ഫീസ് ട്വന്റി ഫൈവ് ആണ്"       # FakeScorer prefers English (tdt)
    assert out["transcription"] == " ".join(s["text"] for s in out["sentences"])
    ws = [w["word"] for w in out["words"]]
    assert ws[:4] == ["ഫീസ്", "ട്വന്റി", "ഫൈവ്", "ആണ്"]                  # 25 -> two words
    w25 = [w for w in out["words"] if w["word"] in ("ട്വന്റി", "ഫൈവ്")]
    assert w25[0]["start_time"] == 0.4 and w25[-1]["end_time"] == 1.0 and w25[0]["end_time"] == w25[1]["start_time"]
    assert tot["word_fallbacks"] == 0 and tot["word_text_mismatch"] == 0
    html = open(tmp / "same" / "data" / "report.html", encoding="utf-8").read()
    assert "Picks by form" in html and 'id="numbers-data"' in html and "text_verbalized" in html
    from indic_normalizer.digits.scoring.pipeline import run_dir
    with pytest.raises(ValueError):                                  # never write into the input dir
        run_dir(str(tmp / "verbalized" / "verbalized"), load_config(None, out_dir=str(tmp)))
    with pytest.raises(ValueError):
        verbalize_dir(str(data), str(audio), FakeScorer(), load_config(None), log=lambda *_: None)


def test_config_old_output_dir_name():
    assert load_config(None, out_dir="a").out_dir == "a"
    import tempfile, os as _os
    p = _os.path.join(tempfile.mkdtemp(), "c.yaml")
    open(p, "w").write("output_dir: /x\nreport_dir: null\noutput_format: same\n")
    assert load_config(p).out_dir == "/x"


def test_verify_copy_has_confident_numbers_verbalized(dataset):
    data, audio, tmp = dataset

    class Mixed(FakeScorer):          # 25: a tie (uncertain); 4: clearly Malayalam
        def score(self, pairs, head="tdt"):
            return [1.0 if "ശരി" not in t else (0.0 if "നാല്" in t else 5.0) for _, t, *r in pairs]

    tot = verbalize_dir(str(data), str(audio), Mixed(), load_config(None, out_dir=str(tmp / "o")), log=lambda *_: None)
    assert tot["verify"] == 1 and tot["needs_review"] == 1
    rd = tmp / "o" / "data"
    doc = json.load(open(rd / "verify" / "rec1.json", encoding="utf-8"))
    assert [x["text"] for x in doc["sentences"]] == ["ഫീസ് 25 ആണ്", "ശരി നാല് പേര്", "ഓക്കേ"]
    assert "നാല്" in [w["word"] for w in doc["words"]] and "25" in [w["word"] for w in doc["words"]]
    from indic_normalizer.digits.scoring.details import read_tsv
    rows = read_tsv(str(rd / "verify.tsv"))
    assert [(r["number"], r["status"], r["final_pick"]) for r in rows] == [("25", "pending", ""), ("4", "auto", "നാല്")]


def test_spelling_rule_keeps_clear_audio_spelling_and_alt_words():
    from indic_normalizer.digits.scoring.pipeline import Number, decide
    from indic_normalizer.digits import find, candidates
    sp = find("98000")[0]
    cands = candidates(sp)
    forms = [c.form for c in cands]
    n = Number(0, sp, cands, "")
    cfg = load_config(None)

    def score(pref):
        n.scores[("segment", "tdt")] = [pref.get(c.form, 10.0) for c in cands]
        return decide(n, "segment", "tdt", cfg)

    d = score({"glide": 1.0, "contracted": 4.0, "spaced": 5.0})          # glide clearly wins (gap 3)
    assert cands[d["best"]].form == "glide" and not d["style_fallback"]
    d = score({"glide_alt": 1.0, "contracted_alt": 1.5, "contracted": 3.5})  # എട്ടായിരം word wins, spelling close
    assert cands[d["best"]].form == "contracted_alt" and d["style_fallback"]
    assert cands[d["best"]].text == "തൊണ്ണൂറ്റെട്ടായിരം" and "contracted_alt" in forms


def _suffix_number(core, scores_by_form):
    from types import SimpleNamespace
    from indic_normalizer.digits import ml
    from indic_normalizer.digits.scoring.pipeline import Number
    cands = ml.candidates(core)
    n = Number(0, SimpleNamespace(core=core, cls="num_suffix", text=core, token_index=0), cands, cands[0].text)
    n.scores[("segment", "tdt")] = [scores_by_form.get(c.form, 9.0) for c in cands]
    return n


def test_num_suffix_has_bare_candidates():
    from indic_normalizer.digits import ml
    forms = {c.form: c.text for c in ml.candidates("27-ന്")}
    assert forms["contracted"] == "ഇരുപത്തേഴിന്" and forms["contracted+nosuffix"] == "ഇരുപത്തേഴ്"
    assert "english+nosuffix" in forms
    c = ml.candidates("55000-യുടെ")
    fs = [x.form for x in c]
    assert fs.index("english+nosuffix") < fs.index("contracted_alt") < 12
    assert ml.suffix_issue("55000-യുടെ") and ml.suffix_issue("1000-ഉം")
    assert ml.suffix_issue("250യിൽ") is None and ml.suffix_issue("27-ന്") is None
    assert ml.suffix_issue("1000-ന്") is None          # ആയിരത്തിന് is normal


def test_suffix_issue_forces_review():
    from indic_normalizer.digits.scoring.pipeline import decide
    cfg = load_config(None)
    # the bare number wins clearly -> reading is confident but still sent to verify
    n = _suffix_number("27-ന്", {"contracted+nosuffix": 1.0, "contracted": 8.0})
    d = decide(n, "segment", "tdt", cfg)
    assert n.cands[d["model_best"]].form.endswith("+nosuffix")
    assert d["suffix_issue"] and d["needs_review"] and d["review_margin"] > cfg.margin_threshold
    # the suffixed reading wins clearly -> no issue
    n = _suffix_number("27-ന്", {"contracted": 1.0})
    d = decide(n, "segment", "tdt", cfg)
    assert d["suffix_issue"] is None and not d["needs_review"]
    # -ം + vowel suffix is flagged even when the audio agrees with the text
    n = _suffix_number("55000-യുടെ", {"contracted": 1.0})
    d = decide(n, "segment", "tdt", cfg)
    assert "dropped noun" in d["suffix_issue"] and d["needs_review"]


def test_split_digit_readings():
    from indic_normalizer.digits import ml
    assert ml.SPLIT_MIN_DIGITS == 2
    ph = ml.candidates("9876543210")                       # phone: digits first, then the value
    assert ph[0].form == "digit_by_digit" and ph[0].text.startswith("ഒമ്പത് എട്ട്") and ph[1].lang == "en"
    assert any(c.text == "നൂറ് കോടി" for c in ml.candidates("1000000000"))   # up to 999 crore
    c = {(x.lang, x.form): x.text for x in ml.candidates("9876543210-ൽ")}   # suffixed: words + split
    assert ("ml", "digit_by_digit") in c and ("ml", "contracted") in c
    c = {(x.lang, x.form): x.text for x in ml.candidates("94")}               # ordinary numbers too
    assert c[("ml", "digit_by_digit")] == "ഒമ്പത് നാല്" and c[("en", "digit_by_digit")] == "നൈൻ ഫോർ"
    assert ml.candidates("94")[0].text == "തൊണ്ണൂറ്റിനാല്" and not any(x.form == "digit_by_digit" for x in ml.candidates("7"))



def test_scorer_error_fails_its_group_only(tmp_path):
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    for k in range(3):
        (data / f"r{k}.json").write_text(json.dumps(DOC, ensure_ascii=False), encoding="utf-8")
        _wav(audio / f"r{k}.wav", 4.0)
    calls = {"n": 0}

    class Flaky(FakeScorer):
        def score(self, pairs, head="tdt"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("CUDA out of memory")
            return super().score(pairs, head)
    cfg = load_config(None, out_dir=str(tmp_path / "o"), group_audio_s=0)      # one file per group
    tot = verbalize_dir(str(data), str(audio), Flaky(), cfg, log=lambda *_: None)
    assert tot["failed"] == 1 and tot["verbalized"] + tot["verify"] == 2
    rd = tmp_path / "o" / "data"
    from indic_normalizer.digits.scoring.details import read_tsv
    failed = read_tsv(str(rd / "failed.tsv"))
    assert [(r["file"], r["stage"]) for r in failed] == [("r1", "score")] and "out of memory" in failed[0]["error"]
    assert (rd / "failed" / "r1.json").exists() and (rd / "summary.json").exists()


def test_config_rejects_bad_values():
    for kw in ({"style": "contractd"}, {"joint_chunk_frames": 0}, {"context_words": -1}, {"sample_rate": 0}):
        with pytest.raises(ValueError):
            load_config(None, **kw)


def test_decide_nan_and_cache():
    import math
    from indic_normalizer.digits.scoring.pipeline import decide, _round
    n = _suffix_number("27-ന്", {"contracted": float("nan"), "glide": 2.0})
    d = decide(n, "segment", "tdt", load_config(None))
    assert n.cands[d["model_best"]].form != "contracted" and decide(n, "segment", "tdt", load_config(None)) is d
    assert _round(float("nan")) is None and _round(math.inf) is None


def test_english_pairs_reading():
    from indic_normalizer.digits import ml
    assert ml.en_pairs(428) == "ഫോർ ട്വന്റി എയ്റ്റ്" and ml.en_pairs(405) == "ഫോർ ഓ ഫൈവ്"
    assert ml.en_pairs(2500) == "ട്വന്റി ഫൈവ് ഹൺഡ്രഡ്" and ml.en_pairs(400) is None and ml.en_pairs(99) is None
    c = {x.text: x.form for x in ml.candidates("428")}
    assert c["ഫോർ ട്വന്റി എയ്റ്റ്"] == "pairs" and c["ഫോർ ഹൺഡ്രഡ് ട്വന്റി എയ്റ്റ്"] == "english"


def test_hundred_percentage_reading():
    from indic_normalizer.digits import ml
    c = {x.text: x.form for x in ml.candidates("100")}
    assert c["ഹൺഡ്രഡ് പെർസെന്റേജ്"] == "percentage_said" and "ഹൺഡ്രഡ്" in c
    assert ml.candidates("100")[0].text == "നൂറ്"
    assert "ഹൺഡ്രഡ് പെർസെന്റേജ്" in {x.text for x in ml.candidates("100%")}
    assert {x.text: x.form for x in ml.candidates("1000")}["തൗസന്റ്"] == "english_bare"


def test_currency_sign_written_apart_needs_review():
    from types import SimpleNamespace
    from indic_normalizer.digits import find
    from indic_normalizer.digits.scoring.pipeline import context_issue, decide, Number
    from indic_normalizer.languages.malayalam import normalize
    for text, flagged in (("വില ₹ 500 ആണ്", True), ("വില Rs. 500 ആണ്", True), ("INR 500", True),
                          ("വില 500 ₹ ആണ്", True), ("വില ₹500 ആണ്", False), ("500 രൂപ", False), ("rs500", False)):
        t = normalize(text, mode="train")
        sp = find(t)[0]
        assert bool(context_issue(t, sp)) == flagged, text
    t = normalize("വില Rs. 500 ആണ്", mode="train")
    sp = find(t)[0]
    from indic_normalizer.digits import candidates
    cands = candidates(sp)
    n = Number(0, sp, cands, "")
    n.text_issue = context_issue(t, sp)
    n.scores[("segment", "tdt")] = [1.0] + [20.0] * (len(cands) - 1)     # very confident
    d = decide(n, "segment", "tdt", load_config(None))
    assert d["text_issue"] and d["needs_review"] and d["review_margin"] > 5


def test_ordinal_glide_variant():
    from indic_normalizer.digits import ml
    from indic_normalizer.digits.scoring.pipeline import family
    c = {x.text: x for x in ml.candidates("30-ന്")}
    assert "തേർട്ടിയത്തിന്" in c and family(c["തേർട്ടിയത്തിന്"]) == family(c["തേർട്ടിത്തിന്"])
    assert ml.en_ordinal_ya(20) == "ട്വന്റിയത്ത്" and ml.en_ordinal_ya(27) is None


def test_doubled_digits():
    from indic_normalizer.digits import ml
    assert ml.en_doubled("55") == "ഡബിൾ ഫൈവ്" and ml.en_doubled("555") == "ട്രിപ്പിൾ ഫൈവ്"
    assert ml.en_doubled("98100") == "നൈൻ എയ്റ്റ് വൺ ഡബിൾ സീറോ" and ml.en_doubled("123") is None
    assert ("doubled", "ഡബിൾ ഫൈവ്") in [(x.form, x.text) for x in ml.candidates("55")]
    assert ml.candidates("55")[0].text == "അമ്പത്തഞ്ച്"
    assert "ഡബിൾ ഫൈവിന്" in {x.text for x in ml.candidates("55-ന്")}
    assert any(x.form == "doubled" for x in ml.candidates("9988776655"))
    assert not any(x.form == "doubled" for x in ml.candidates("56"))


def test_english_unit_word():
    from indic_normalizer.digits import ml
    c = {x.text: x.form for x in ml.candidates("5-മാസം")}
    assert c["ഫൈവ് മന്ത്"] == "english+en_unit" and c["അഞ്ച് മാസം"] == "contracted"
    assert not any("മന്ത്" in x.text for x in ml.candidates("5-മാസത്തിൽ"))     # case ending: no translation
    assert "ടു പോയിന്റ് ഫൈവ് ലാക്" in {x.text for x in ml.candidates("2.5-ലക്ഷം")}
    assert not any(x.form == "digit_by_digit" for x in ml.candidates("30"))      # round numbers not split


class Prefers(FakeScorer):
    """Confident fake: the sentence reading the given words scores 0, every other one 5."""
    def __init__(self, *words):
        super().__init__()
        self.words = words

    def score(self, pairs, head="tdt"):
        return [0.0 if any(f" {w} " in f" {t} " for w in self.words) else 5.0 for _, t, *_ in pairs]


def test_reverify(dataset, monkeypatch):
    from indic_normalizer.digits.scoring.details import NUMBER_COLS, read_tsv, write_tsv
    from indic_normalizer.digits.scoring.reverify import reverify
    data, audio, tmp = dataset
    cfg = load_config(None, out_dir=str(tmp / "o"))
    english = Prefers("ട്വന്റി ഫൈവ്", "ഫോർ")
    verbalize_dir(str(data), str(audio), english, cfg, log=lambda *_: None)
    rd = tmp / "o" / "data"
    v = read_tsv(str(rd / "verbalized.tsv"))
    assert [(r["number"], r["status"], r["final_pick"]) for r in v] == [("25", "auto", "ട്വന്റി ഫൈവ്"), ("4", "auto", "ഫോർ")]
    # same scorer: nothing changes
    tot = reverify(str(rd), str(audio), english, cfg, log=lambda *_: None)
    assert tot["files_checked"] == 1 and tot["numbers_flagged"] == 0 and (rd / "verbalized" / "rec1.json").exists()

    # 4 was verified by hand as ഫോർ; 25 stays auto. Now the model prefers Malayalam strongly.
    v[1].update(status="verified", final_pick="ഫോർ")
    write_tsv(str(rd / "verbalized.tsv"), NUMBER_COLS, v)
    malayalam = Prefers("ഇരുപത്തഞ്ച്", "നാല്")
    tot = reverify(str(rd), str(audio), malayalam, cfg, dry_run=True, log=lambda *_: None)
    assert tot["numbers_flagged"] == 2 and tot["flagged_verified"] == 1 and (rd / "verbalized" / "rec1.json").exists()
    rep = read_tsv(str(rd / "reverify_report.tsv"))
    nums = read_tsv(str(rd / "reverify_numbers.tsv"))                # every checked number, measured
    assert [(r["number"], r["status"], r["agrees"]) for r in nums] == [("25", "auto", "false"), ("4", "verified", "false")]
    meas = json.load(open(rd / "reverify_report.json", encoding="utf-8"))["measure"]
    assert meas["verified_numbers"] == 1 and meas["verified_agrees"] == 0
    assert "Model vs readings verified by hand" in open(rd / "reverify_report.html", encoding="utf-8").read()
    assert {(r["number"], r["status_before"], r["action"]) for r in rep} == {("25", "auto", "dry-run"), ("4", "verified", "dry-run")}

    tot = reverify(str(rd), str(audio), malayalam, cfg, log=lambda *_: None)
    assert tot["files_moved_to_reverify"] == ["rec1"]
    assert not (rd / "verbalized" / "rec1.json").exists() and (rd / "reverify" / "rec1.json").exists()
    assert not (rd / "verify" / "rec1.json").exists()
    assert (rd / "input" / "rec1.json").exists() and read_tsv(str(rd / "verbalized.tsv")) == []
    rows = read_tsv(str(rd / "reverify.tsv"))
    assert [(r["number"], r["status"], r["previous_pick"]) for r in rows] == [("25", "pending", "ട്വന്റി ഫൈവ്"), ("4", "pending", "ഫോർ")]
    out = json.load(open(rd / "reverify" / "rec1.json", encoding="utf-8"))
    assert "25" in out["sentences"][0]["text"]                      # open again: written as digits
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    assert (s["verbalized"], s["reverify"]) == (0, 1)
    from indic_normalizer.digits.verifier import server as V        # the verifier picks it up
    monkeypatch.setattr(V, "SETTINGS", str(tmp / "settings.json"))
    st = {}
    V.api_open(st, {"run_dir": str(rd), "audio_dir": str(audio)})
    assert [(f["file"], f["kind"]) for f in V.api_files(st)["files"]] == [("rec1", "reverify")]
    d = V.api_file(st, "rec1", "reverify")
    assert [n["previous_pick"] for x in d["sentences"] for n in x["numbers"]] == ["ട്വന്റി ഫൈവ്", "ഫോർ"]
    i25, i4 = [n["id"] for x in d["sentences"] for n in x["numbers"]]
    r = V.api_save(st, {"file": "rec1", "kind": "reverify", "rev": d["rev"], "picks": {i25: "ഇരുപത്തഞ്ച്", i4: "നാല് പേര്"}})
    assert r["done"] and r["typed"] == 1                              # നാല് പേര് is not a candidate
    assert (rd / "verbalized" / "rec1.json").exists() and not (rd / "reverify" / "rec1.json").exists()
    assert read_tsv(str(rd / "reverify.tsv")) == [] and len(read_tsv(str(rd / "verbalized.tsv"))) == 2
    typed = read_tsv(str(rd / "typed_readings.tsv"))
    assert [(t["number"], t["typed"], t["previous_pick"], t["kind"]) for t in typed] == [("4", "നാല് പേര്", "ഫോർ", "reverify")]
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    assert (s["verbalized"], s["reverify"]) == (1, 0)


def test_suffix_gap_measured():
    from indic_normalizer.digits.scoring.pipeline import decide
    cfg = load_config(None)
    n = _suffix_number("27-ന്", {"contracted+nosuffix": 1.0, "contracted": 4.5})
    d = decide(n, "segment", "tdt", cfg)
    assert round(d["suffix_gap"], 4) == 3.5 and d["suffix_bare"] == "ഇരുപത്തേഴ്" and d["suffix_with"] == "ഇരുപത്തേഴിന്"
    n = _suffix_number("27-ന്", {"contracted": 1.0, "contracted+nosuffix": 3.0})
    assert round(decide(n, "segment", "tdt", cfg)["suffix_gap"], 4) == -2.0
    from types import SimpleNamespace
    from indic_normalizer.digits import ml
    from indic_normalizer.digits.scoring.pipeline import Number
    c = ml.candidates("25")
    m = Number(0, SimpleNamespace(core="25", cls="integer", text="25", token_index=0), c, c[0].text)
    m.scores[("segment", "tdt")] = [1.0] * len(c)
    assert decide(m, "segment", "tdt", cfg)["suffix_gap"] is None


def test_reverify_measures_suffix_choices(tmp_path):
    from indic_normalizer.digits.scoring.details import NUMBER_COLS, read_tsv, write_tsv
    from indic_normalizer.digits.scoring.reverify import reverify
    doc = {"transcription": "", "words": [],
           "sentences": [{"text": "ജൂലൈ 27-ന് വരും", "start_time": 0.0, "end_time": 2.0}]}
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    (data / "r.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _wav(audio / "r.wav", 3.0)
    cfg = load_config(None, out_dir=str(tmp_path / "o"))
    bare = Prefers("ഇരുപത്തേഴ്")                    # audio: no suffix
    verbalize_dir(str(data), str(audio), bare, cfg, log=lambda *_: None)
    rd = tmp_path / "o" / "data"
    # the person dropped the suffix in the verifier: stored as verified + bare reading, in verbalized/
    rows = read_tsv(str(rd / "verify.tsv"))
    rows[0].update(status="verified", final_pick="ഇരുപത്തേഴ്")
    write_tsv(str(rd / "verbalized.tsv"), NUMBER_COLS, rows)
    write_tsv(str(rd / "verify.tsv"), NUMBER_COLS, [])
    os.replace(rd / "verify" / "r.json", rd / "verbalized" / "r.json")
    tot = reverify(str(rd), str(audio), bare, cfg, dry_run=True, log=lambda *_: None)
    n = read_tsv(str(rd / "reverify_numbers.tsv"))[0]
    assert n["human_choice"] == "bare" and float(n["suffix_gap"]) > 0
    band = [b for b in tot["measure"]["verified_num_suffix_by_suffix_gap"] if b["bare_wins"]["n"]][0]
    assert band["bare_wins"]["human_bare"] == 1


def test_reverify_collects_typed_readings(tmp_path):
    from indic_normalizer.digits.scoring.details import NUMBER_COLS, read_tsv, write_tsv
    from indic_normalizer.digits.scoring.reverify import reverify
    doc = {"transcription": "", "words": [],
           "sentences": [{"text": "വില 500 ആണ്", "start_time": 0.0, "end_time": 2.0}]}
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    (data / "r.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _wav(audio / "r.wav", 3.0)
    cfg = load_config(None, out_dir=str(tmp_path / "o"))
    sc = Prefers("അഞ്ഞൂറ്")
    verbalize_dir(str(data), str(audio), sc, cfg, log=lambda *_: None)
    rd = tmp_path / "o" / "data"
    rows = read_tsv(str(rd / "verbalized.tsv"))
    rows[0].update(status="verified", final_pick="അഞ്ഞൂറ് രൂപ", verified_at="2026-10-01T10:00:00")
    write_tsv(str(rd / "verbalized.tsv"), NUMBER_COLS, rows)
    tot = reverify(str(rd), str(audio), sc, cfg, dry_run=True, log=lambda *_: None)
    assert tot["typed_readings_found"] == 1 and tot["typed_readings_added"] == 1
    t = read_tsv(str(rd / "typed_readings.tsv"))
    assert [(x["number"], x["typed"], x["kind"], x["saved_at"]) for x in t] == \
        [("500", "അഞ്ഞൂറ് രൂപ", "verbalized", "2026-10-01T10:00:00")]
    tot = reverify(str(rd), str(audio), sc, cfg, dry_run=True, log=lambda *_: None)
    assert tot["typed_readings_added"] == 0 and len(read_tsv(str(rd / "typed_readings.tsv"))) == 1   # no duplicates


def test_reverify_refreshes_waiting_files(dataset):
    from indic_normalizer.digits.scoring.details import NUMBER_COLS, read_tsv, write_tsv
    from indic_normalizer.digits.scoring.reverify import reverify
    data, audio, tmp = dataset
    cfg = load_config(None, out_dir=str(tmp / "o"))
    verbalize_dir(str(data), str(audio), FakeScorer(), cfg, log=lambda *_: None)    # -> verify/, both pending
    rd = tmp / "o" / "data"
    rows = read_tsv(str(rd / "verify.tsv"))
    rows[0].update(status="verified", final_pick="ട്വന്റി ഫൈവ്")                    # 25 decided by hand
    rows[1].update(candidates="old", scores="", model_pick="old")                   # 4: stale candidates
    write_tsv(str(rd / "verify.tsv"), NUMBER_COLS, rows)
    before = (rd / "verify" / "rec1.json").read_bytes()
    tot = reverify(str(rd), str(audio), Prefers("ഇരുപത്തഞ്ച്", "നാല്"), cfg, dry_run=True, log=lambda *_: None)
    assert tot["waiting"]["files"] == 1 and tot["waiting"]["numbers_refreshed"] == 1
    assert read_tsv(str(rd / "verify.tsv"))[1]["model_pick"] == "old"                # dry run: unchanged
    tot = reverify(str(rd), str(audio), Prefers("ഇരുപത്തഞ്ച്", "നാല്"), cfg, log=lambda *_: None)
    r0, r1 = read_tsv(str(rd / "verify.tsv"))
    assert (r0["status"], r0["final_pick"]) == ("verified", "ട്വന്റി ഫൈവ്")         # decided: untouched
    assert r1["status"] == "pending" and r1["model_pick"] == "നാല്" and "ഫോർ" in r1["candidates"] and r1["scores"]
    assert (rd / "verify" / "rec1.json").read_bytes() == before                     # the copy is not touched
    tot = reverify(str(rd), str(audio), Prefers("ഇരുപത്തഞ്ച്", "നാല്"), cfg, refresh=False, log=lambda *_: None)
    assert "waiting" not in tot


def test_verbalize_resume(tmp_path):
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    for k in range(3):
        (data / f"r{k}.json").write_text(json.dumps(DOC, ensure_ascii=False), encoding="utf-8")
        _wav(audio / f"r{k}.wav", 4.0)
    cfg = load_config(None, out_dir=str(tmp_path / "o"))
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), cfg, limit=1, log=lambda *_: None)
    assert tot["this_run"]["files"] == 1
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), cfg, resume=True, log=lambda *_: None)
    assert tot["resumed_skipped"] == 1 and tot["this_run"]["files"] == 2 and tot["report"].startswith("report_resume_")
    s = json.load(open(tmp_path / "o" / "data" / "summary.json", encoding="utf-8"))
    assert s["files"] == 3 and s["verify"] + s["verbalized"] + s["failed"] == 3 and s["this_run"]["files"] == 2
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), cfg, resume=True, log=lambda *_: None)
    assert tot["this_run"]["files"] == 0 and tot["resumed_skipped"] == 3
    assert (tmp_path / "o" / "data" / "report.html").exists()


def test_verbalize_retry_failed(tmp_path):
    from indic_normalizer.digits.scoring.details import read_tsv
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    for k in range(2):
        (data / f"r{k}.json").write_text(json.dumps(DOC, ensure_ascii=False), encoding="utf-8")
    _wav(audio / "r0.wav", 4.0)                                  # r1: no audio yet -> failed
    cfg = load_config(None, out_dir=str(tmp_path / "o"))
    rd = tmp_path / "o" / "data"
    verbalize_dir(str(data), str(audio), FakeScorer(), cfg, log=lambda *_: None)
    assert (rd / "failed" / "r1.json").exists()
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), cfg, resume=True, log=lambda *_: None)
    assert tot["this_run"]["files"] == 0                         # --resume: failed counts as done
    _wav(audio / "r1.wav", 4.0)
    tot = verbalize_dir(str(data), str(audio), FakeScorer(), cfg, retry_failed=True, log=lambda *_: None)
    assert tot["this_run"]["files"] == 1 and tot["resumed_skipped"] == 1
    assert not (rd / "failed" / "r1.json").exists() and not read_tsv(str(rd / "failed.tsv"))
    assert (rd / "verbalized" / "r1.json").exists() or (rd / "verify" / "r1.json").exists()
    s = json.load(open(rd / "summary.json", encoding="utf-8"))
    assert s["failed"] == 0 and s["files"] == 2


def test_old_rows_without_currency_sign_still_align(tmp_path):
    from indic_normalizer.digits.scoring.details import NUMBER_COLS, read_tsv, write_tsv
    from indic_normalizer.digits.scoring.reverify import reverify
    data, audio = tmp_path / "data", tmp_path / "audio"
    data.mkdir()
    audio.mkdir()
    (data / "a.json").write_text(json.dumps({"sentences": [{"text": "ഫീസ് ₹ 500 ആണ്", "start_time": 0, "end_time": 2}],
                                             "words": []}, ensure_ascii=False), encoding="utf-8")
    _wav(audio / "a.wav", 3.0)
    cfg = load_config(None, out_dir=str(tmp_path / "o"))
    sc = Prefers("അഞ്ഞൂറ്")
    verbalize_dir(str(data), str(audio), sc, cfg, log=lambda *_: None)
    rd = tmp_path / "o" / "data"
    rows = read_tsv(str(rd / "verify.tsv"))
    assert rows[0]["number"] == "₹500"
    rows[0]["number"] = "500"                                  # as an older run wrote it
    write_tsv(str(rd / "verify.tsv"), NUMBER_COLS, rows)
    tot = reverify(str(rd), str(audio), sc, cfg, log=lambda *_: None)
    assert tot["waiting"]["files"] == 1 and tot["waiting"]["skipped_misaligned"] == 0
